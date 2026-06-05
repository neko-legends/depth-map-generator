use base64::{engine::general_purpose, Engine as _};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{ChildStdin, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_opener::OpenerExt;
use uuid::Uuid;

#[cfg(unix)]
use std::os::unix::process::CommandExt;
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

const IMAGE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "jpe", "jfif", "jpf", "jp2", "j2k", "webp", "bmp", "tif", "tiff",
];
#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;
const EMBEDDED_WORKER: &str = include_str!("../../python/da3_worker.py");

#[derive(Debug, Serialize)]
struct RuntimeStatus {
    #[serde(rename = "pythonFound")]
    python_found: bool,
    #[serde(rename = "pythonCommand")]
    python_command: Option<String>,
    #[serde(rename = "appDataDir")]
    app_data_dir: String,
    #[serde(rename = "workerPath")]
    worker_path: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GpuVram {
    name: String,
    memory_used_mb: u64,
    memory_total_mb: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VramStatus {
    available: bool,
    gpus: Vec<GpuVram>,
    message: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelStatus {
    repo: String,
    name: String,
    installed: bool,
    model_path: String,
    size_bytes: u64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct DepthOptions {
    device: String,
    model_repo: String,
    output_format: String,
    generate_depth: bool,
    convert_webp: bool,
    webp_quality: u8,
    invert: bool,
    recursive: bool,
    overwrite: bool,
    output_dir: String,
    suffix: String,
    resize_mode: String,
    resize_value: u32,
    process_res: u32,
    process_res_method: String,
    normalize_low: f32,
    normalize_high: f32,
    depth_gamma: f32,
    keep_model_loaded: bool,
}

#[derive(Debug, Serialize)]
struct WorkerJob {
    paths: Vec<String>,
    options: DepthOptions,
    app_data_dir: String,
    env_dir: String,
    model_dir: String,
    download_only: bool,
    preload_only: bool,
}

#[derive(Clone, Default)]
struct ActiveJobState {
    inner: Arc<Mutex<Option<ActiveJob>>>,
}

struct ActiveJob {
    id: String,
    pid: u32,
    cancel_requested: bool,
}

impl ActiveJobState {
    fn is_busy(&self) -> Result<bool, String> {
        self.inner
            .lock()
            .map(|job| job.is_some())
            .map_err(|_| "Unable to lock active job state.".to_string())
    }

    fn start(&self, id: String, pid: u32) -> Result<(), String> {
        let mut active = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock active job state.".to_string())?;
        if active.is_some() {
            return Err("Another job is already running.".to_string());
        }
        *active = Some(ActiveJob {
            id,
            pid,
            cancel_requested: false,
        });
        Ok(())
    }

    fn request_cancel(&self) -> Result<u32, String> {
        let mut active = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock active job state.".to_string())?;
        let Some(job) = active.as_mut() else {
            return Err("No active job to cancel.".to_string());
        };
        job.cancel_requested = true;
        Ok(job.pid)
    }

    fn finish(&self, id: &str) -> Result<bool, String> {
        let mut active = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock active job state.".to_string())?;
        let Some(job) = active.as_ref() else {
            return Ok(false);
        };
        if job.id != id {
            return Ok(false);
        }
        let canceled = job.cancel_requested;
        *active = None;
        Ok(canceled)
    }

    fn finish_current(&self) -> Result<Option<bool>, String> {
        let mut active = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock active job state.".to_string())?;
        let canceled = active.as_ref().map(|job| job.cancel_requested);
        *active = None;
        Ok(canceled)
    }

    fn finish_pid(&self, pid: u32) -> Result<Option<bool>, String> {
        let mut active = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock active job state.".to_string())?;
        let Some(job) = active.as_ref() else {
            return Ok(None);
        };
        if job.pid != pid {
            return Ok(None);
        }
        let canceled = job.cancel_requested;
        *active = None;
        Ok(Some(canceled))
    }
}

#[derive(Clone, Default)]
struct PersistentWorkerState {
    inner: Arc<Mutex<Option<PersistentWorker>>>,
}

struct PersistentWorker {
    pid: u32,
    stdin: Arc<Mutex<ChildStdin>>,
}

fn worker_path(app: &AppHandle) -> Result<PathBuf, String> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf));
    let resource_dir = app.path().resource_dir().ok();
    let candidates = [
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("python")
            .join("da3_worker.py"),
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("resources")
            .join("python")
            .join("da3_worker.py"),
        exe_dir
            .unwrap_or_else(|| PathBuf::from("."))
            .join("python")
            .join("da3_worker.py"),
        resource_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("_up_")
            .join("python")
            .join("da3_worker.py"),
        resource_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("python")
            .join("da3_worker.py"),
        resource_dir
            .unwrap_or_else(|| PathBuf::from("."))
            .join("da3_worker.py"),
    ];
    let checked = candidates
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");

    if let Some(path) = candidates.into_iter().find(|path| path.exists()) {
        return Ok(path);
    }

    write_embedded_worker(app).map_err(|error| {
        format!(
            "Could not find python/da3_worker.py. Checked: {checked}. Embedded fallback failed: {error}"
        )
    })
}

fn write_embedded_worker(app: &AppHandle) -> Result<PathBuf, String> {
    let worker_dir = app_data_dir(app)?.join("python");
    fs::create_dir_all(&worker_dir)
        .map_err(|error| format!("Unable to create embedded worker dir: {error}"))?;
    let worker = worker_dir.join("da3_worker.py");
    let needs_write = fs::read_to_string(&worker)
        .map(|content| content != EMBEDDED_WORKER)
        .unwrap_or(true);
    if needs_write {
        fs::write(&worker, EMBEDDED_WORKER)
            .map_err(|error| format!("Unable to write embedded worker: {error}"))?;
    }
    Ok(worker)
}

fn app_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Unable to resolve app data dir: {error}"))?;
    fs::create_dir_all(&dir).map_err(|error| format!("Unable to create app data dir: {error}"))?;
    Ok(dir)
}

fn parse_python_version(output: &str) -> Option<(u32, u32, u32)> {
    let version = output
        .split_whitespace()
        .find(|part| part.chars().next().is_some_and(|ch| ch.is_ascii_digit()))?;
    let mut parts = version.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor, patch))
}

fn supported_python_version(version: (u32, u32, u32)) -> bool {
    version.0 == 3 && (10..=12).contains(&version.1)
}

fn python_version(spec: &str) -> Option<(u32, u32, u32)> {
    let mut parts = spec.split_whitespace();
    let executable = parts.next()?;
    let args: Vec<_> = parts.collect();
    let mut command = Command::new(executable);
    command.args(args).arg("--version");
    hide_command_window(&mut command);
    let output = command.output().ok()?;
    if !output.status.success() {
        return None;
    }
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    parse_python_version(&combined)
}

fn find_python() -> Option<String> {
    let candidates = if cfg!(target_os = "windows") {
        vec![
            "py -3.11", "py -3.12", "py -3.10", "python", "python3", "py -3",
        ]
    } else {
        vec![
            "python3.11",
            "python3.12",
            "python3.10",
            "python3",
            "python",
        ]
    };

    for candidate in candidates {
        if python_version(candidate).is_some_and(supported_python_version) {
            return Some(candidate.to_string());
        }
    }
    None
}

fn hide_command_window(command: &mut Command) {
    #[cfg(target_os = "windows")]
    {
        command.creation_flags(CREATE_NO_WINDOW);
    }
}

fn command_from_python_spec(spec: &str) -> Command {
    let mut parts = spec.split_whitespace();
    let executable = parts.next().unwrap_or("python");
    let mut command = Command::new(executable);
    for arg in parts {
        command.arg(arg);
    }
    hide_command_window(&mut command);
    command
}

fn configure_worker_command(command: &mut Command) {
    hide_command_window(command);
    #[cfg(unix)]
    {
        command.process_group(0);
    }
}

fn kill_process_tree(pid: u32) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut command = Command::new("taskkill");
        command.args(["/PID", &pid.to_string(), "/T", "/F"]);
        command
    };

    #[cfg(not(target_os = "windows"))]
    let mut command = {
        let mut command = Command::new("kill");
        command.args(["-TERM", &format!("-{pid}")]);
        command
    };

    hide_command_window(&mut command);
    match command.output() {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => {
            #[cfg(not(target_os = "windows"))]
            {
                let fallback = Command::new("kill")
                    .args(["-TERM", &pid.to_string()])
                    .output();
                if matches!(fallback, Ok(output) if output.status.success()) {
                    return Ok(());
                }
            }
            Err(format!(
                "Unable to cancel job: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ))
        }
        Err(error) => Err(format!("Unable to cancel job: {error}")),
    }
}

fn managed_python_path(env_dir: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        env_dir.join("Scripts").join("python.exe")
    } else {
        env_dir.join("bin").join("python")
    }
}

fn installed_packages_report(env_dir: &Path) -> String {
    let python_path = managed_python_path(env_dir);
    if !python_path.exists() {
        return "Installed packages: app-managed Python environment has not been created yet.\n"
            .to_string();
    }

    let mut command = Command::new(python_path);
    command.args(["-m", "pip", "freeze", "--all"]);
    hide_command_window(&mut command);

    match command.output() {
        Ok(output) if output.status.success() => format!(
            "Installed packages from app-managed environment:\n\n{}",
            String::from_utf8_lossy(&output.stdout)
        ),
        Ok(output) => format!(
            "Installed packages: unable to read pip freeze.\n\n{}",
            String::from_utf8_lossy(&output.stderr)
        ),
        Err(error) => format!("Installed packages: unable to run pip freeze: {error}\n"),
    }
}

fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            IMAGE_EXTENSIONS
                .iter()
                .any(|candidate| candidate.eq_ignore_ascii_case(ext))
        })
        .unwrap_or(false)
}

fn collect_images(path: &Path, recursive: bool, output: &mut Vec<String>) {
    if path.is_file() {
        if is_image_file(path) {
            output.push(path.display().to_string());
        }
        return;
    }

    if !path.is_dir() {
        return;
    }

    let Ok(entries) = fs::read_dir(path) else {
        return;
    };

    for entry in entries.flatten() {
        let entry_path = entry.path();
        if entry_path.is_file() && is_image_file(&entry_path) {
            output.push(entry_path.display().to_string());
        } else if recursive && entry_path.is_dir() {
            collect_images(&entry_path, recursive, output);
        }
    }
}

fn model_cache_name(model_repo: &str) -> Result<&'static str, String> {
    match model_repo {
        "sy000/MDA/mda_mog_sky_l2" => Ok("MDA-DA3-MoG-Sky"),
        "depth-anything/DA3-LARGE-1.1" => Ok("DA3-LARGE-1.1"),
        "depth-anything/DA3-SMALL" => Ok("DA3-SMALL"),
        _ => Err(format!("Unsupported depth model: {model_repo}")),
    }
}

fn model_size_bytes(model_repo: &str) -> Result<u64, String> {
    match model_repo {
        "sy000/MDA/mda_mog_sky_l2" => Ok(5_423_360_059),
        "depth-anything/DA3-LARGE-1.1" => Ok(1_643_843_860),
        "depth-anything/DA3-SMALL" => Ok(137_000_000),
        _ => Err(format!("Unsupported depth model: {model_repo}")),
    }
}

fn supported_models() -> [&'static str; 3] {
    [
        "sy000/MDA/mda_mog_sky_l2",
        "depth-anything/DA3-LARGE-1.1",
        "depth-anything/DA3-SMALL",
    ]
}

fn model_dir_for(app_data: &Path, model_repo: &str) -> Result<PathBuf, String> {
    Ok(app_data.join("models").join(model_cache_name(model_repo)?))
}

fn model_file_for(model_dir: &Path, model_repo: &str) -> Result<PathBuf, String> {
    match model_repo {
        "sy000/MDA/mda_mog_sky_l2" => Ok(model_dir
            .join("source")
            .join("checkpoints")
            .join("MDA")
            .join("DA3_MOG_Sky_LogL2.ckpt")),
        "depth-anything/DA3-LARGE-1.1" | "depth-anything/DA3-SMALL" => {
            Ok(model_dir.join("model.safetensors"))
        }
        _ => Err(format!("Unsupported depth model: {model_repo}")),
    }
}

fn terminal_worker_event(value: &serde_json::Value) -> bool {
    value
        .get("type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|event_type| matches!(event_type, "done" | "error" | "canceled"))
}

fn persistent_worker(
    app: &AppHandle,
    persistent_workers: &PersistentWorkerState,
    active_jobs: &ActiveJobState,
) -> Result<(u32, Arc<Mutex<ChildStdin>>), String> {
    if let Some(worker) = persistent_workers
        .inner
        .lock()
        .map_err(|_| "Unable to lock persistent worker state.".to_string())?
        .as_ref()
    {
        return Ok((worker.pid, worker.stdin.clone()));
    }

    let python = find_python()
        .ok_or_else(|| "Python 3.10, 3.11, or 3.12 was not found on PATH.".to_string())?;
    let worker = worker_path(app)?;
    let app_data = app_data_dir(app)?;
    let jobs_dir = app_data.join("jobs");
    let env_dir = app_data.join("python-env");
    let model_dir = model_dir_for(&app_data, "sy000/MDA/mda_mog_sky_l2")?;
    fs::create_dir_all(&jobs_dir).map_err(|error| format!("Unable to create jobs dir: {error}"))?;

    let job_id = Uuid::new_v4().to_string();
    let job_path = jobs_dir.join(format!("persistent-{job_id}.json"));
    let options = DepthOptions {
        device: "auto".to_string(),
        model_repo: "sy000/MDA/mda_mog_sky_l2".to_string(),
        output_format: "png8".to_string(),
        generate_depth: true,
        convert_webp: false,
        webp_quality: 92,
        invert: true,
        recursive: true,
        overwrite: false,
        output_dir: String::new(),
        suffix: "_depth".to_string(),
        resize_mode: "none".to_string(),
        resize_value: 1024,
        process_res: 504,
        process_res_method: "upper_bound_resize".to_string(),
        normalize_low: 1.0,
        normalize_high: 99.0,
        depth_gamma: 1.0,
        keep_model_loaded: true,
    };
    let job = WorkerJob {
        paths: Vec::new(),
        options,
        app_data_dir: app_data.display().to_string(),
        env_dir: env_dir.display().to_string(),
        model_dir: model_dir.display().to_string(),
        download_only: false,
        preload_only: false,
    };
    fs::write(
        &job_path,
        serde_json::to_string_pretty(&job)
            .map_err(|error| format!("Unable to serialize persistent worker config: {error}"))?,
    )
    .map_err(|error| format!("Unable to write persistent worker config: {error}"))?;

    let mut command = command_from_python_spec(&python);
    configure_worker_command(&mut command);
    command
        .arg(worker)
        .arg("--daemon")
        .arg("--job")
        .arg(job_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| format!("Unable to start persistent Python worker: {error}"))?;
    let child_pid = child.id();
    let Some(stdin) = child.stdin.take() else {
        let _ = kill_process_tree(child_pid);
        return Err("Unable to open persistent worker input pipe.".to_string());
    };
    let stdin = Arc::new(Mutex::new(stdin));
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let app_for_stdout = app.clone();
    let app_for_stderr = app.clone();
    let app_for_wait = app.clone();
    let active_jobs_for_stdout = (*active_jobs).clone();
    let active_jobs_for_wait = (*active_jobs).clone();
    let persistent_for_wait = (*persistent_workers).clone();

    thread::spawn(move || {
        if let Some(stdout) = stdout {
            let reader = BufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
                    let terminal = terminal_worker_event(&value);
                    let _ = app_for_stdout.emit("depth-worker-event", value);
                    if terminal {
                        let _ = active_jobs_for_stdout.finish_current();
                    }
                } else {
                    let _ = app_for_stdout.emit(
                        "depth-worker-event",
                        serde_json::json!({ "type": "setup", "message": line }),
                    );
                }
            }
        }
    });

    thread::spawn(move || {
        if let Some(stderr) = stderr {
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                let _ = app_for_stderr.emit(
                    "depth-worker-event",
                    serde_json::json!({ "type": "setup", "message": line }),
                );
            }
        }
    });

    thread::spawn(move || {
        let status = child.wait();
        if let Ok(mut worker) = persistent_for_wait.inner.lock() {
            if worker
                .as_ref()
                .is_some_and(|worker| worker.pid == child_pid)
            {
                *worker = None;
            }
        }
        let canceled = active_jobs_for_wait
            .finish_pid(child_pid)
            .ok()
            .flatten()
            .unwrap_or(false);
        if canceled {
            let _ = app_for_wait.emit(
                "depth-worker-event",
                serde_json::json!({
                    "type": "canceled",
                    "message": "Job canceled."
                }),
            );
        } else if !matches!(status, Ok(status) if status.success()) {
            let _ = app_for_wait.emit(
                "depth-worker-event",
                serde_json::json!({
                    "type": "error",
                    "message": "Persistent depth worker exited with an error."
                }),
            );
        }
    });

    let mut slot = persistent_workers
        .inner
        .lock()
        .map_err(|_| "Unable to lock persistent worker state.".to_string())?;
    if let Some(existing) = slot.as_ref() {
        let _ = kill_process_tree(child_pid);
        return Ok((existing.pid, existing.stdin.clone()));
    }
    *slot = Some(PersistentWorker {
        pid: child_pid,
        stdin: stdin.clone(),
    });
    Ok((child_pid, stdin))
}

fn enqueue_persistent_job(
    app: AppHandle,
    persistent_workers: State<'_, PersistentWorkerState>,
    active_jobs: State<'_, ActiveJobState>,
    paths: Vec<String>,
    options: DepthOptions,
    preload_only: bool,
) -> Result<String, String> {
    if active_jobs.is_busy()? {
        return Err("Another job is already running.".to_string());
    }

    let app_data = app_data_dir(&app)?;
    let env_dir = app_data.join("python-env");
    let model_dir = model_dir_for(&app_data, &options.model_repo)?;
    let job_id = Uuid::new_v4().to_string();
    let job = WorkerJob {
        paths,
        options,
        app_data_dir: app_data.display().to_string(),
        env_dir: env_dir.display().to_string(),
        model_dir: model_dir.display().to_string(),
        download_only: false,
        preload_only,
    };
    let job_json = serde_json::to_string(&job)
        .map_err(|error| format!("Unable to serialize persistent job: {error}"))?;
    let (pid, stdin) = persistent_worker(&app, &persistent_workers, &active_jobs)?;
    if let Err(error) = active_jobs.start(job_id.clone(), pid) {
        return Err(error);
    }
    let send_result = stdin
        .lock()
        .map_err(|_| "Unable to lock persistent worker input.".to_string())
        .and_then(|mut input| {
            writeln!(input, "{job_json}")
                .and_then(|_| input.flush())
                .map_err(|error| format!("Unable to send job to persistent worker: {error}"))
        });
    if let Err(error) = send_result {
        let _ = active_jobs.finish(&job_id);
        return Err(error);
    }
    Ok(job_id)
}

fn image_mime_type(path: &Path) -> Result<&'static str, String> {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => Ok("image/png"),
        Some("jpg") | Some("jpeg") | Some("jpe") | Some("jfif") => Ok("image/jpeg"),
        Some("webp") => Ok("image/webp"),
        Some("bmp") => Ok("image/bmp"),
        Some("tif") | Some("tiff") => Ok("image/tiff"),
        Some(extension) => Err(format!("Unsupported preview image extension: {extension}")),
        None => Err("Preview image has no file extension.".to_string()),
    }
}

#[tauri::command]
fn check_runtime(app: AppHandle) -> Result<RuntimeStatus, String> {
    let app_data = app_data_dir(&app)?;
    let worker = worker_path(&app)?;
    let python = find_python();
    Ok(RuntimeStatus {
        python_found: python.is_some(),
        python_command: python,
        app_data_dir: app_data.display().to_string(),
        worker_path: worker.display().to_string(),
    })
}

#[tauri::command]
fn get_model_status(app: AppHandle) -> Result<Vec<ModelStatus>, String> {
    let app_data = app_data_dir(&app)?;
    supported_models()
        .into_iter()
        .map(|repo| {
            let model_dir = model_dir_for(&app_data, repo)?;
            let model_path = model_file_for(&model_dir, repo)?;
            Ok(ModelStatus {
                repo: repo.to_string(),
                name: model_cache_name(repo)?.to_string(),
                installed: model_path.exists(),
                model_path: model_path.display().to_string(),
                size_bytes: model_size_bytes(repo)?,
            })
        })
        .collect()
}

#[tauri::command]
fn open_model_folder(app: AppHandle) -> Result<(), String> {
    let models_dir = app_data_dir(&app)?.join("models");
    fs::create_dir_all(&models_dir)
        .map_err(|error| format!("Unable to create model folder: {error}"))?;
    app.opener()
        .open_path(models_dir.display().to_string(), None::<&str>)
        .map_err(|error| format!("Unable to open model folder: {error}"))
}

#[tauri::command]
fn open_package_report(app: AppHandle) -> Result<(), String> {
    let app_data = app_data_dir(&app)?;
    let env_dir = app_data.join("python-env");
    let report_path = app_data.join("package-report.txt");
    let report = format!(
        "\
DepthMap AI Generator package report
====================================

This file lists the Python packages this app installs into its own app-managed
virtual environment, plus the currently installed package versions when that
environment exists.

App data folder:
{app_data}

Managed Python environment:
{env_dir}

Direct install commands used by the app
---------------------------------------

Base setup:
python -m pip install --upgrade pip==26.1.1 setuptools==70.2.0 wheel==0.47.0

WebP conversion:
python -m pip install pillow==12.2.0

Depth generation:
python -m pip install huggingface_hub==1.15.0 requests==2.34.2 numpy==1.26.4 pillow==12.2.0 opencv-python==4.11.0.86 safetensors==0.7.0 addict==2.4.0
python -m pip install --upgrade torch==2.11.0 torchvision==0.26.0
NVIDIA machines:
python -m pip install --upgrade --force-reinstall torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --upgrade numpy==1.26.4
python -m pip install git+https://github.com/ByteDance-Seed/depth-anything-3.git@41736238f5bced4debf3f2a12375d2466874866d

MDA runtime, installed only when the MDA model is selected:
python -m pip install PyYAML==6.0.3 omegaconf==2.3.0 einops==0.8.2 scipy==1.17.1 tqdm==4.67.3 sympy==1.14.0 lightning-bolts==0.7.0

Model files are downloaded separately from:
- https://huggingface.co/sy000/MDA
- https://github.com/biansy000/MDA/archive/0a4ae22e6994a9794dfb24a1275c7fbf59112f2a.zip
- https://huggingface.co/depth-anything/DA3-LARGE-1.1
- https://huggingface.co/depth-anything/DA3-SMALL

{installed}
",
        app_data = app_data.display(),
        env_dir = env_dir.display(),
        installed = installed_packages_report(&env_dir)
    );

    fs::write(&report_path, report)
        .map_err(|error| format!("Unable to write package report: {error}"))?;
    app.opener()
        .open_path(report_path.display().to_string(), None::<&str>)
        .map_err(|error| format!("Unable to open package report: {error}"))
}

#[tauri::command]
fn open_containing_folder(app: AppHandle, path: String) -> Result<(), String> {
    let requested_path = PathBuf::from(path);
    let folder = if requested_path.is_dir() {
        requested_path
    } else {
        requested_path
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| "Unable to resolve containing folder.".to_string())?
    };
    if !folder.is_dir() {
        return Err(format!("Folder does not exist: {}", folder.display()));
    }
    app.opener()
        .open_path(folder.display().to_string(), None::<&str>)
        .map_err(|error| format!("Unable to open folder: {error}"))
}

#[tauri::command]
fn preview_image_data_url(path: String) -> Result<String, String> {
    let image_path = PathBuf::from(path);
    if !image_path.is_file() {
        return Err("Preview image was not found.".to_string());
    }
    let mime_type = image_mime_type(&image_path)?;
    let bytes =
        fs::read(&image_path).map_err(|error| format!("Unable to read preview image: {error}"))?;
    Ok(format!(
        "data:{mime_type};base64,{}",
        general_purpose::STANDARD.encode(bytes)
    ))
}

#[tauri::command]
fn resolve_inputs(paths: Vec<String>, recursive: bool) -> Vec<String> {
    let mut output = Vec::new();
    for path in paths {
        collect_images(Path::new(&path), recursive, &mut output);
    }
    output.sort();
    output.dedup();
    output
}

#[tauri::command]
fn get_vram_status() -> VramStatus {
    let mut command = Command::new("nvidia-smi");
    command.args([
        "--query-gpu=name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]);
    hide_command_window(&mut command);

    let output = match command.output() {
        Ok(output) if output.status.success() => output,
        Ok(output) => {
            return VramStatus {
                available: false,
                gpus: Vec::new(),
                message: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            };
        }
        Err(error) => {
            return VramStatus {
                available: false,
                gpus: Vec::new(),
                message: format!("VRAM unavailable: {error}"),
            };
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let gpus = stdout
        .lines()
        .filter_map(|line| {
            let parts: Vec<_> = line.split(',').map(str::trim).collect();
            if parts.len() < 3 {
                return None;
            }
            Some(GpuVram {
                name: parts[0].to_string(),
                memory_used_mb: parts[1].parse().ok()?,
                memory_total_mb: parts[2].parse().ok()?,
            })
        })
        .collect::<Vec<_>>();

    VramStatus {
        available: !gpus.is_empty(),
        message: if gpus.is_empty() {
            "No NVIDIA GPUs reported.".to_string()
        } else {
            "OK".to_string()
        },
        gpus,
    }
}

#[tauri::command]
fn start_depth_job(
    app: AppHandle,
    persistent_workers: State<'_, PersistentWorkerState>,
    active_jobs: State<'_, ActiveJobState>,
    paths: Vec<String>,
    options: DepthOptions,
) -> Result<String, String> {
    if paths.is_empty() {
        return Err("No input images were provided.".to_string());
    }
    if active_jobs.is_busy()? {
        return Err("Another job is already running.".to_string());
    }
    if options.keep_model_loaded && options.generate_depth {
        return enqueue_persistent_job(app, persistent_workers, active_jobs, paths, options, false);
    }

    let python = find_python()
        .ok_or_else(|| "Python 3.10, 3.11, or 3.12 was not found on PATH.".to_string())?;
    let worker = worker_path(&app)?;
    let app_data = app_data_dir(&app)?;
    let jobs_dir = app_data.join("jobs");
    let env_dir = app_data.join("python-env");
    let model_dir = model_dir_for(&app_data, &options.model_repo)?;
    fs::create_dir_all(&jobs_dir).map_err(|error| format!("Unable to create jobs dir: {error}"))?;

    let job_id = Uuid::new_v4().to_string();
    let job_path = jobs_dir.join(format!("{job_id}.json"));
    let job = WorkerJob {
        paths,
        options,
        app_data_dir: app_data.display().to_string(),
        env_dir: env_dir.display().to_string(),
        model_dir: model_dir.display().to_string(),
        download_only: false,
        preload_only: false,
    };
    let job_json = serde_json::to_string_pretty(&job)
        .map_err(|error| format!("Unable to serialize job: {error}"))?;
    fs::write(&job_path, job_json).map_err(|error| format!("Unable to write job: {error}"))?;

    let mut command = command_from_python_spec(&python);
    configure_worker_command(&mut command);
    command
        .arg(worker)
        .arg("--job")
        .arg(job_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| format!("Unable to start Python worker: {error}"))?;
    let child_pid = child.id();
    if let Err(error) = active_jobs.start(job_id.clone(), child_pid) {
        let _ = kill_process_tree(child_pid);
        return Err(error);
    }
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let app_for_stdout = app.clone();
    let app_for_stderr = app.clone();
    let app_for_wait = app.clone();
    let active_jobs_for_wait = active_jobs.inner().clone();
    let job_id_for_wait = job_id.clone();

    thread::spawn(move || {
        if let Some(stdout) = stdout {
            let reader = BufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
                    let _ = app_for_stdout.emit("depth-worker-event", value);
                } else {
                    let _ = app_for_stdout.emit(
                        "depth-worker-event",
                        serde_json::json!({ "type": "setup", "message": line }),
                    );
                }
            }
        }
    });

    thread::spawn(move || {
        if let Some(stderr) = stderr {
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                let _ = app_for_stderr.emit(
                    "depth-worker-event",
                    serde_json::json!({ "type": "setup", "message": line }),
                );
            }
        }
    });

    thread::spawn(move || {
        let status = child.wait();
        let was_canceled = active_jobs_for_wait
            .finish(&job_id_for_wait)
            .unwrap_or(false);
        if was_canceled {
            let _ = app_for_wait.emit(
                "depth-worker-event",
                serde_json::json!({
                    "type": "canceled",
                    "message": "Job canceled."
                }),
            );
        } else if !matches!(status, Ok(status) if status.success()) {
            let _ = app_for_wait.emit(
                "depth-worker-event",
                serde_json::json!({
                    "type": "error",
                    "message": "Depth worker exited with an error."
                }),
            );
        }
    });

    Ok(job_id)
}

#[tauri::command]
fn download_model(app: AppHandle, model_repo: String) -> Result<String, String> {
    let active_jobs = app.state::<ActiveJobState>();
    if active_jobs.is_busy()? {
        return Err("Another job is already running.".to_string());
    }
    let python = find_python()
        .ok_or_else(|| "Python 3.10, 3.11, or 3.12 was not found on PATH.".to_string())?;
    let worker = worker_path(&app)?;
    let app_data = app_data_dir(&app)?;
    let jobs_dir = app_data.join("jobs");
    let env_dir = app_data.join("python-env");
    let model_dir = model_dir_for(&app_data, &model_repo)?;
    fs::create_dir_all(&jobs_dir).map_err(|error| format!("Unable to create jobs dir: {error}"))?;

    let job_id = Uuid::new_v4().to_string();
    let job_path = jobs_dir.join(format!("{job_id}.json"));
    let options = DepthOptions {
        device: "auto".to_string(),
        model_repo,
        output_format: "png8".to_string(),
        generate_depth: true,
        convert_webp: false,
        webp_quality: 92,
        invert: true,
        recursive: true,
        overwrite: false,
        output_dir: String::new(),
        suffix: "_depth".to_string(),
        resize_mode: "none".to_string(),
        resize_value: 1024,
        process_res: 504,
        process_res_method: "upper_bound_resize".to_string(),
        normalize_low: 1.0,
        normalize_high: 99.0,
        depth_gamma: 1.0,
        keep_model_loaded: false,
    };
    let job = WorkerJob {
        paths: Vec::new(),
        options,
        app_data_dir: app_data.display().to_string(),
        env_dir: env_dir.display().to_string(),
        model_dir: model_dir.display().to_string(),
        download_only: true,
        preload_only: false,
    };
    let job_json = serde_json::to_string_pretty(&job)
        .map_err(|error| format!("Unable to serialize job: {error}"))?;
    fs::write(&job_path, job_json).map_err(|error| format!("Unable to write job: {error}"))?;

    let mut command = command_from_python_spec(&python);
    configure_worker_command(&mut command);
    command
        .arg(worker)
        .arg("--job")
        .arg(job_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| format!("Unable to start Python worker: {error}"))?;
    let child_pid = child.id();
    if let Err(error) = active_jobs.start(job_id.clone(), child_pid) {
        let _ = kill_process_tree(child_pid);
        return Err(error);
    }
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let app_for_stdout = app.clone();
    let app_for_stderr = app.clone();
    let app_for_wait = app.clone();
    let active_jobs_for_wait = active_jobs.inner().clone();
    let job_id_for_wait = job_id.clone();

    thread::spawn(move || {
        if let Some(stdout) = stdout {
            let reader = BufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
                    let _ = app_for_stdout.emit("depth-worker-event", value);
                } else {
                    let _ = app_for_stdout.emit(
                        "depth-worker-event",
                        serde_json::json!({ "type": "setup", "message": line }),
                    );
                }
            }
        }
    });

    thread::spawn(move || {
        if let Some(stderr) = stderr {
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                let _ = app_for_stderr.emit(
                    "depth-worker-event",
                    serde_json::json!({ "type": "setup", "message": line }),
                );
            }
        }
    });

    thread::spawn(move || {
        let status = child.wait();
        let was_canceled = active_jobs_for_wait
            .finish(&job_id_for_wait)
            .unwrap_or(false);
        if was_canceled {
            let _ = app_for_wait.emit(
                "depth-worker-event",
                serde_json::json!({
                    "type": "canceled",
                    "message": "Job canceled."
                }),
            );
        } else if !matches!(status, Ok(status) if status.success()) {
            let _ = app_for_wait.emit(
                "depth-worker-event",
                serde_json::json!({
                    "type": "error",
                    "message": "Model download worker exited with an error."
                }),
            );
        }
    });

    Ok(job_id)
}

#[tauri::command]
fn cancel_active_job(active_jobs: State<'_, ActiveJobState>) -> Result<(), String> {
    let pid = active_jobs.request_cancel()?;
    kill_process_tree(pid)
}

#[tauri::command]
fn preload_model(
    app: AppHandle,
    persistent_workers: State<'_, PersistentWorkerState>,
    active_jobs: State<'_, ActiveJobState>,
    options: DepthOptions,
) -> Result<String, String> {
    if !options.keep_model_loaded || !options.generate_depth {
        return Ok("Model preload disabled.".to_string());
    }
    enqueue_persistent_job(
        app,
        persistent_workers,
        active_jobs,
        Vec::new(),
        options,
        true,
    )
}

#[tauri::command]
fn stop_persistent_worker(
    persistent_workers: State<'_, PersistentWorkerState>,
    active_jobs: State<'_, ActiveJobState>,
) -> Result<(), String> {
    if active_jobs.is_busy()? {
        return Ok(());
    }
    let worker = persistent_workers
        .inner
        .lock()
        .map_err(|_| "Unable to lock persistent worker state.".to_string())?
        .take();
    if let Some(worker) = worker {
        if let Ok(mut input) = worker.stdin.lock() {
            let _ = writeln!(input, "{}", serde_json::json!({ "shutdown": true }));
            let _ = input.flush();
        }
    }
    Ok(())
}

pub fn run() {
    tauri::Builder::default()
        .manage(ActiveJobState::default())
        .manage(PersistentWorkerState::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            cancel_active_job,
            check_runtime,
            get_model_status,
            get_vram_status,
            download_model,
            open_model_folder,
            open_package_report,
            open_containing_folder,
            preload_model,
            preview_image_data_url,
            resolve_inputs,
            start_depth_job,
            stop_persistent_worker
        ])
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}
