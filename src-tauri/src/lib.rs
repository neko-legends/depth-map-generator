use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
};
use tauri::{AppHandle, Emitter, Manager};
use uuid::Uuid;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

const IMAGE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "jpe", "jfif", "jpf", "jp2", "j2k", "webp", "bmp", "tif", "tiff",
];
#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

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
    suffix: String,
    resize_mode: String,
    resize_value: u32,
}

#[derive(Debug, Serialize)]
struct WorkerJob {
    paths: Vec<String>,
    options: DepthOptions,
    app_data_dir: String,
    env_dir: String,
    model_dir: String,
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
            .join("python")
            .join("da3_worker.py"),
        resource_dir
            .unwrap_or_else(|| PathBuf::from("."))
            .join("da3_worker.py"),
    ];

    candidates
        .into_iter()
        .find(|path| path.exists())
        .ok_or_else(|| {
            format!(
                "Could not find python/da3_worker.py from {}",
                app.path()
                    .app_data_dir()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|_| "unknown app data dir".to_string())
            )
        })
}

fn app_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Unable to resolve app data dir: {error}"))?;
    fs::create_dir_all(&dir).map_err(|error| format!("Unable to create app data dir: {error}"))?;
    Ok(dir)
}

fn find_python() -> Option<String> {
    let candidates = if cfg!(target_os = "windows") {
        vec!["py -3", "python", "python3"]
    } else {
        vec!["python3", "python"]
    };

    for candidate in candidates {
        let mut parts = candidate.split_whitespace();
        let executable = parts.next()?;
        let args: Vec<_> = parts.collect();
        let mut command = Command::new(executable);
        command
            .args(args)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        hide_command_window(&mut command);
        let status = command.status();
        if matches!(status, Ok(status) if status.success()) {
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

fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| IMAGE_EXTENSIONS.iter().any(|candidate| candidate.eq_ignore_ascii_case(ext)))
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
        "depth-anything/DA3-LARGE-1.1" => Ok("DA3-LARGE-1.1"),
        "depth-anything/DA3-SMALL" => Ok("DA3-SMALL"),
        _ => Err(format!("Unsupported depth model: {model_repo}")),
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
fn start_depth_job(app: AppHandle, paths: Vec<String>, options: DepthOptions) -> Result<String, String> {
    if paths.is_empty() {
        return Err("No input images were provided.".to_string());
    }

    let python = find_python().ok_or_else(|| "Python 3.10+ was not found on PATH.".to_string())?;
    let worker = worker_path(&app)?;
    let app_data = app_data_dir(&app)?;
    let jobs_dir = app_data.join("jobs");
    let env_dir = app_data.join("python-env");
    let model_dir = app_data
        .join("models")
        .join(model_cache_name(&options.model_repo)?);
    fs::create_dir_all(&jobs_dir).map_err(|error| format!("Unable to create jobs dir: {error}"))?;

    let job_id = Uuid::new_v4().to_string();
    let job_path = jobs_dir.join(format!("{job_id}.json"));
    let job = WorkerJob {
        paths,
        options,
        app_data_dir: app_data.display().to_string(),
        env_dir: env_dir.display().to_string(),
        model_dir: model_dir.display().to_string(),
    };
    let job_json = serde_json::to_string_pretty(&job)
        .map_err(|error| format!("Unable to serialize job: {error}"))?;
    fs::write(&job_path, job_json).map_err(|error| format!("Unable to write job: {error}"))?;

    let mut command = command_from_python_spec(&python);
    command
        .arg(worker)
        .arg("--job")
        .arg(job_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| format!("Unable to start Python worker: {error}"))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let app_for_stdout = app.clone();
    let app_for_stderr = app.clone();

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
        if !matches!(status, Ok(status) if status.success()) {
            let _ = app.emit(
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

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            check_runtime,
            get_vram_status,
            resolve_inputs,
            start_depth_job
        ])
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}
