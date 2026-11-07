use base64::{engine::general_purpose, Engine as _};
use serde::{Deserialize, Serialize};
use std::{
    env, fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{ChildStdin, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_opener::OpenerExt;
use uuid::Uuid;

#[cfg(unix)]
use std::os::unix::process::CommandExt;
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

mod neko_store;
use neko_store::NekoStore;

const IMAGE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "jpe", "jfif", "jpf", "jp2", "j2k", "webp", "bmp", "tif", "tiff",
];
#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;
const EMBEDDED_WORKER: &str = include_str!("../../python/da3_worker.py");
const DEFAULT_AGENT_API_PORT: u16 = 17334;
const AGENT_APP_ID: &str = "depth-map-ai-generator";
const AGENT_APP_NAME: &str = "DepthMap AI Generator";
const AGENT_API_BIND_ADDRESS: &str = "127.0.0.1";
const AGENT_API_REGISTRY_FILE: &str = "agent-api-registry.json";
const DEFAULT_WINDOW_WIDTH: u32 = 1920;
const DEFAULT_WINDOW_HEIGHT: u32 = 1080;
const MIN_WINDOW_WIDTH: u32 = 640;
const MIN_WINDOW_HEIGHT: u32 = 360;
const LANDSCAPE_WINDOW_WIDTH_RATIO: f64 = 0.5;
const PORTRAIT_WINDOW_WIDTH_RATIO: f64 = 0.9;

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

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AgentJobRequest {
    paths: Option<Vec<String>>,
    model_repo: Option<String>,
    options: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AgentServerStatus {
    enabled: bool,
    port: u16,
    url: String,
    openapi_url: String,
    busy: bool,
    active_job_id: Option<String>,
    message: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct AgentApiRegistryEntry {
    app_id: String,
    app_name: String,
    default_port: u16,
    bind_address: String,
    port: u16,
    enabled: bool,
    url: String,
    openapi_url: String,
    busy: bool,
    active_job_id: Option<String>,
    last_seen: Option<String>,
    note: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct AgentApiRegistry {
    updated_at: String,
    apps: Vec<AgentApiRegistryEntry>,
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

    fn active_job_id(&self) -> Result<Option<String>, String> {
        self.inner
            .lock()
            .map(|job| job.as_ref().map(|job| job.id.clone()))
            .map_err(|_| "Unable to lock active job state.".to_string())
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

#[derive(Clone, Default)]
struct AgentServerState {
    inner: Arc<Mutex<AgentServerControl>>,
}

#[derive(Default)]
struct AgentServerControl {
    enabled: bool,
    port: u16,
    stop: Option<Arc<AtomicBool>>,
}

impl AgentServerControl {
    fn port(&self) -> u16 {
        if self.port == 0 {
            read_registered_agent_api_port().unwrap_or(DEFAULT_AGENT_API_PORT)
        } else {
            self.port
        }
    }
}

fn default_depth_options() -> DepthOptions {
    DepthOptions {
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
        keep_model_loaded: false,
    }
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

/// App identifier used for the shared NekoLegendsAI store.
const NEKO_APP_SLUG: &str = "depthmap";

/// Resolve the shared NekoLegendsAI store for this app.
///
/// The default store volume is hinted to the executable's drive so that on a
/// portable install on `D:` the store lands at `D:\NekoLegendsAI` (next to the
/// app, on the roomy drive) rather than on the C: app-data volume.
fn neko_store(app: &AppHandle) -> Result<NekoStore, String> {
    let app_data = app_data_dir(app)?;
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    Ok(NekoStore::resolve_with_volume_hint(
        NEKO_APP_SLUG,
        &app_data,
        exe_dir.as_deref(),
    ))
}

/// The Python venv directory, preserving any pre-existing app-local install so
/// upgrades never orphan a working environment.
///
/// If a legacy `<app_data>/python-env` already exists, it keeps being used.
/// Otherwise the shared store's per-app venv is used so wheels are deduped.
fn resolve_env_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let app_data = app_data_dir(app)?;
    let legacy = app_data.join("python-env");
    if legacy.exists() {
        return Ok(legacy);
    }
    Ok(neko_store(app)?.env_dir)
}

/// The models root, preserving any pre-existing app-local models folder.
fn resolve_models_root(app: &AppHandle) -> Result<PathBuf, String> {
    let app_data = app_data_dir(app)?;
    let legacy = app_data.join("models");
    if legacy.exists() {
        return Ok(legacy);
    }
    Ok(neko_store(app)?.models_dir)
}

/// Apply shared cache env vars (HF/pip/uv/torch) to a worker command so that
/// model and wheel downloads are reused across the whole suite.
fn apply_store_env(app: &AppHandle, command: &mut Command) {
    if let Ok(store) = neko_store(app) {
        store.apply_env(command);
    }
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
    let mut command = python_spec_command(spec);
    command.arg("--version");
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

    // Rolling-release distros (e.g. Arch/Omarchy) often ship only a newer
    // Python than the ML stack supports, so fall back to a uv-managed one.
    #[cfg(not(target_os = "windows"))]
    if let Some(python) = find_uv().and_then(|uv| uv_python(&uv, false)) {
        return Some(python);
    }
    None
}

/// Like `find_python`, but on Linux/macOS asks uv to install a supported
/// Python when none is available yet. Used right before starting the worker.
fn ensure_python() -> Result<String, String> {
    if let Some(python) = find_python() {
        return Ok(python);
    }
    #[cfg(not(target_os = "windows"))]
    {
        if let Some(python) = find_uv().and_then(|uv| uv_python(&uv, true)) {
            return Ok(python);
        }
        Err("Python 3.10, 3.11, or 3.12 was not found on PATH, and uv was not available to \
             install one. Install uv (https://docs.astral.sh/uv/) or python3.12 and try again."
            .to_string())
    }
    #[cfg(target_os = "windows")]
    Err("Python 3.10, 3.11, or 3.12 was not found on PATH.".to_string())
}

/// Python version uv provides when no supported system Python exists.
#[cfg(not(target_os = "windows"))]
const UV_PYTHON_VERSION: &str = "3.12";

/// Locate `uv`: PATH first, then the usual install locations, since apps
/// started from a desktop launcher may not inherit the shell's PATH.
#[cfg(not(target_os = "windows"))]
fn find_uv() -> Option<PathBuf> {
    let mut candidates = vec![PathBuf::from("uv")];
    if let Some(home) = env::var_os("HOME").map(PathBuf::from) {
        candidates.push(home.join(".local").join("bin").join("uv"));
        candidates.push(home.join(".cargo").join("bin").join("uv"));
        candidates.push(
            home.join(".local")
                .join("share")
                .join("mise")
                .join("shims")
                .join("uv"),
        );
    }
    candidates.push(PathBuf::from("/usr/local/bin/uv"));
    candidates.push(PathBuf::from("/usr/bin/uv"));
    candidates.into_iter().find(|uv| {
        Command::new(uv)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    })
}

/// Path of a uv-managed (or uv-discovered) Python 3.10-3.12, installing
/// `UV_PYTHON_VERSION` first when `install` is set.
#[cfg(not(target_os = "windows"))]
fn uv_python(uv: &Path, install: bool) -> Option<String> {
    let find = || -> Option<String> {
        let output = Command::new(uv)
            .args(["python", "find", "--no-project", ">=3.10,<3.13"])
            .stderr(Stdio::null())
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        (!path.is_empty() && python_version(&path).is_some_and(supported_python_version))
            .then_some(path)
    };
    if let Some(path) = find() {
        return Some(path);
    }
    if !install {
        return None;
    }
    let installed = Command::new(uv)
        .args(["python", "install", UV_PYTHON_VERSION])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if installed {
        find()
    } else {
        None
    }
}

/// Build a command from a Python spec: either an interpreter path (which may
/// contain spaces) or a launcher plus arguments such as `py -3.11`.
fn python_spec_command(spec: &str) -> Command {
    if Path::new(spec).is_file() {
        return Command::new(spec);
    }
    let mut parts = spec.split_whitespace();
    let mut command = Command::new(parts.next().unwrap_or("python"));
    command.args(parts);
    command
}

fn hide_command_window(command: &mut Command) {
    #[cfg(target_os = "windows")]
    {
        command.creation_flags(CREATE_NO_WINDOW);
    }
}

fn command_from_python_spec(spec: &str) -> Command {
    let mut command = python_spec_command(spec);
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

fn model_dir_for(models_root: &Path, model_repo: &str) -> Result<PathBuf, String> {
    Ok(models_root.join(model_cache_name(model_repo)?))
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

    let python = ensure_python()?;
    let worker = worker_path(app)?;
    let app_data = app_data_dir(app)?;
    let jobs_dir = app_data.join("jobs");
    let env_dir = resolve_env_dir(app)?;
    let models_root = resolve_models_root(app)?;
    let model_dir = model_dir_for(&models_root, "sy000/MDA/mda_mog_sky_l2")?;
    fs::create_dir_all(&jobs_dir).map_err(|error| format!("Unable to create jobs dir: {error}"))?;

    let job_id = Uuid::new_v4().to_string();
    let job_path = jobs_dir.join(format!("persistent-{job_id}.json"));
    let options = DepthOptions {
        keep_model_loaded: true,
        ..default_depth_options()
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
    apply_store_env(app, &mut command);
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
    persistent_workers: &PersistentWorkerState,
    active_jobs: &ActiveJobState,
    paths: Vec<String>,
    options: DepthOptions,
    preload_only: bool,
) -> Result<String, String> {
    if active_jobs.is_busy()? {
        return Err("Another job is already running.".to_string());
    }

    let app_data = app_data_dir(&app)?;
    let env_dir = resolve_env_dir(&app)?;
    let models_root = resolve_models_root(&app)?;
    let model_dir = model_dir_for(&models_root, &options.model_repo)?;
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
    let (pid, stdin) = persistent_worker(&app, persistent_workers, active_jobs)?;
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
    let models_root = resolve_models_root(&app)?;
    supported_models()
        .into_iter()
        .map(|repo| {
            let model_dir = model_dir_for(&models_root, repo)?;
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
    let env_dir = resolve_env_dir(&app)?;
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

fn start_depth_job_core(
    app: AppHandle,
    persistent_workers: &PersistentWorkerState,
    active_jobs: &ActiveJobState,
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

    let python = ensure_python()?;
    let worker = worker_path(&app)?;
    let app_data = app_data_dir(&app)?;
    let jobs_dir = app_data.join("jobs");
    let env_dir = resolve_env_dir(&app)?;
    let models_root = resolve_models_root(&app)?;
    let model_dir = model_dir_for(&models_root, &options.model_repo)?;
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
    apply_store_env(&app, &mut command);
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
    let active_jobs_for_wait = active_jobs.clone();
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
fn start_depth_job(
    app: AppHandle,
    persistent_workers: State<'_, PersistentWorkerState>,
    active_jobs: State<'_, ActiveJobState>,
    paths: Vec<String>,
    options: DepthOptions,
) -> Result<String, String> {
    start_depth_job_core(
        app,
        persistent_workers.inner(),
        active_jobs.inner(),
        paths,
        options,
    )
}

#[tauri::command]
fn download_model(app: AppHandle, model_repo: String) -> Result<String, String> {
    let active_jobs = app.state::<ActiveJobState>();
    if active_jobs.is_busy()? {
        return Err("Another job is already running.".to_string());
    }
    let python = ensure_python()?;
    let worker = worker_path(&app)?;
    let app_data = app_data_dir(&app)?;
    let jobs_dir = app_data.join("jobs");
    let env_dir = resolve_env_dir(&app)?;
    let models_root = resolve_models_root(&app)?;
    let model_dir = model_dir_for(&models_root, &model_repo)?;
    fs::create_dir_all(&jobs_dir).map_err(|error| format!("Unable to create jobs dir: {error}"))?;

    let job_id = Uuid::new_v4().to_string();
    let job_path = jobs_dir.join(format!("{job_id}.json"));
    let options = DepthOptions {
        model_repo,
        ..default_depth_options()
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
    apply_store_env(&app, &mut command);
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
        persistent_workers.inner(),
        active_jobs.inner(),
        Vec::new(),
        options,
        true,
    )
}

fn stop_persistent_worker_core(
    persistent_workers: &PersistentWorkerState,
    active_jobs: &ActiveJobState,
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

#[tauri::command]
fn stop_persistent_worker(
    persistent_workers: State<'_, PersistentWorkerState>,
    active_jobs: State<'_, ActiveJobState>,
) -> Result<(), String> {
    stop_persistent_worker_core(persistent_workers.inner(), active_jobs.inner())
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn validate_agent_api_port(port: Option<u16>) -> Result<u16, String> {
    let port = port
        .or_else(read_registered_agent_api_port)
        .unwrap_or(DEFAULT_AGENT_API_PORT);
    if port == 0 {
        return Err("Agent API port must be between 1 and 65535.".to_string());
    }
    Ok(port)
}

fn agent_api_url(port: u16) -> String {
    format!("http://{AGENT_API_BIND_ADDRESS}:{port}")
}

fn timestamp_string() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}

fn shared_neko_legends_dir() -> Option<PathBuf> {
    let base = if cfg!(target_os = "windows") {
        env::var_os("APPDATA")
            .map(PathBuf::from)
            .or_else(|| env::var_os("USERPROFILE").map(PathBuf::from))
    } else if cfg!(target_os = "macos") {
        env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join("Library").join("Application Support"))
    } else {
        env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
    }?;
    Some(base.join("NekoLegends"))
}

fn agent_api_registry_path() -> Option<PathBuf> {
    Some(shared_neko_legends_dir()?.join(AGENT_API_REGISTRY_FILE))
}

fn read_agent_api_registry() -> AgentApiRegistry {
    let updated_at = timestamp_string();
    let Some(path) = agent_api_registry_path() else {
        return AgentApiRegistry {
            updated_at,
            apps: Vec::new(),
        };
    };
    fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or(AgentApiRegistry {
            updated_at,
            apps: Vec::new(),
        })
}

fn read_registered_agent_api_port() -> Option<u16> {
    read_agent_api_registry()
        .apps
        .into_iter()
        .find(|entry| entry.app_id == AGENT_APP_ID)
        .map(|entry| entry.port)
        .filter(|port| *port > 0)
}

fn publish_agent_api_status(status: &AgentServerStatus) {
    let Some(path) = agent_api_registry_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut registry = read_agent_api_registry();
    let updated_at = timestamp_string();
    let entry = AgentApiRegistryEntry {
        app_id: AGENT_APP_ID.to_string(),
        app_name: AGENT_APP_NAME.to_string(),
        default_port: DEFAULT_AGENT_API_PORT,
        bind_address: AGENT_API_BIND_ADDRESS.to_string(),
        port: status.port,
        enabled: status.enabled,
        url: status.url.clone(),
        openapi_url: status.openapi_url.clone(),
        busy: status.busy,
        active_job_id: status.active_job_id.clone(),
        last_seen: Some(updated_at.clone()),
        note: Some("Local Agent API.".to_string()),
    };
    if let Some(existing) = registry
        .apps
        .iter_mut()
        .find(|entry| entry.app_id == AGENT_APP_ID)
    {
        *existing = entry;
    } else {
        registry.apps.push(entry);
    }
    registry.updated_at = updated_at;
    if let Ok(raw) = serde_json::to_string_pretty(&registry) {
        let _ = fs::write(path, raw);
    }
}

fn agent_status_from(
    agent_state: &AgentServerState,
    active_jobs: &ActiveJobState,
) -> Result<AgentServerStatus, String> {
    let (enabled, port) = {
        let control = agent_state
            .inner
            .lock()
            .map_err(|_| "Unable to lock agent server state.".to_string())?;
        (control.enabled, control.port())
    };
    let active_job_id = active_jobs.active_job_id()?;
    let status = AgentServerStatus {
        enabled,
        port,
        url: agent_api_url(port),
        openapi_url: format!("{}/openapi.json", agent_api_url(port)),
        busy: active_job_id.is_some(),
        active_job_id,
        message: if enabled {
            "Agent API is enabled.".to_string()
        } else {
            "Agent API is off.".to_string()
        },
    };
    publish_agent_api_status(&status);
    Ok(status)
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest, String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|error| format!("Unable to set read timeout: {error}"))?;
    let mut data = Vec::new();
    let mut buffer = [0_u8; 4096];
    let mut expected_len: Option<usize> = None;

    loop {
        let bytes_read = stream
            .read(&mut buffer)
            .map_err(|error| format!("Unable to read agent request: {error}"))?;
        if bytes_read == 0 {
            break;
        }
        data.extend_from_slice(&buffer[..bytes_read]);
        if let Some(header_end) = find_header_end(&data) {
            if expected_len.is_none() {
                let headers = String::from_utf8_lossy(&data[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        if name.eq_ignore_ascii_case("content-length") {
                            value.trim().parse::<usize>().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                expected_len = Some(header_end + 4 + content_length);
            }
            if expected_len.is_some_and(|len| data.len() >= len) {
                break;
            }
        }
        if data.len() > 2 * 1024 * 1024 {
            return Err("Agent request is too large.".to_string());
        }
    }

    let header_end = find_header_end(&data).ok_or_else(|| "Invalid HTTP request.".to_string())?;
    let headers = String::from_utf8_lossy(&data[..header_end]);
    let mut lines = headers.lines();
    let request_line = lines
        .next()
        .ok_or_else(|| "Invalid HTTP request line.".to_string())?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let raw_path = parts.next().unwrap_or("/").to_string();
    let path = raw_path.split('?').next().unwrap_or("/").to_string();
    let body_start = header_end + 4;
    let body = if body_start <= data.len() {
        data[body_start..].to_vec()
    } else {
        Vec::new()
    };
    Ok(HttpRequest { method, path, body })
}

fn find_header_end(data: &[u8]) -> Option<usize> {
    data.windows(4).position(|window| window == b"\r\n\r\n")
}

fn write_json_response(
    stream: &mut TcpStream,
    status: &str,
    payload: serde_json::Value,
) -> Result<(), String> {
    let body = serde_json::to_vec(&payload)
        .map_err(|error| format!("Unable to serialize agent response: {error}"))?;
    let headers = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: content-type\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(headers.as_bytes())
        .and_then(|_| stream.write_all(&body))
        .map_err(|error| format!("Unable to write agent response: {error}"))
}

fn write_empty_response(stream: &mut TcpStream, status: &str) -> Result<(), String> {
    let headers = format!(
        "HTTP/1.1 {status}\r\nContent-Length: 0\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: content-type\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n"
    );
    stream
        .write_all(headers.as_bytes())
        .map_err(|error| format!("Unable to write agent response: {error}"))
}

fn agent_openapi(port: u16) -> serde_json::Value {
    serde_json::json!({
        "openapi": "3.1.0",
        "info": {
            "title": "DepthMap AI Generator Agent API",
            "version": env!("CARGO_PKG_VERSION")
        },
        "servers": [{ "url": agent_api_url(port) }],
        "paths": {
            "/health": { "get": { "summary": "Check API status" } },
            "/status": { "get": { "summary": "Check active job status" } },
            "/runtime": { "get": { "summary": "Check local runtime readiness" } },
            "/vram": { "get": { "summary": "Check NVIDIA VRAM" } },
            "/models": { "get": { "summary": "List supported model installation status" } },
            "/download-model": { "post": { "summary": "Download the selected model" } },
            "/prepare": { "post": { "summary": "Alias for model download" } },
            "/preload": { "post": { "summary": "Load the selected model into the persistent worker" } },
            "/generate": { "post": { "summary": "Start a depth generation job" } },
            "/cancel": { "post": { "summary": "Cancel the active job" } },
            "/stop-worker": { "post": { "summary": "Stop the persistent model worker" } }
        }
    })
}

fn parse_agent_job_request(body: &[u8]) -> Result<AgentJobRequest, String> {
    if body.is_empty() {
        return Ok(AgentJobRequest {
            paths: None,
            model_repo: None,
            options: None,
        });
    }
    serde_json::from_slice(body).map_err(|error| format!("Invalid JSON request: {error}"))
}

fn agent_options_from_request(options: Option<serde_json::Value>) -> Result<DepthOptions, String> {
    let mut merged = serde_json::to_value(default_depth_options())
        .map_err(|error| format!("Unable to build default options: {error}"))?;
    let Some(options) = options else {
        return serde_json::from_value(merged)
            .map_err(|error| format!("Unable to read default options: {error}"));
    };
    if options.is_null() {
        return serde_json::from_value(merged)
            .map_err(|error| format!("Unable to read default options: {error}"));
    }
    let overrides = options
        .as_object()
        .ok_or_else(|| "Agent options must be a JSON object.".to_string())?;
    let base = merged
        .as_object_mut()
        .ok_or_else(|| "Unable to merge default options.".to_string())?;
    for (key, value) in overrides {
        base.insert(key.clone(), value.clone());
    }
    serde_json::from_value(merged).map_err(|error| format!("Invalid agent options: {error}"))
}

fn agent_model_repo_from_request(request: AgentJobRequest) -> Result<String, String> {
    let AgentJobRequest {
        model_repo,
        options,
        ..
    } = request;
    if let Some(model_repo) = model_repo {
        return Ok(model_repo);
    }
    agent_options_from_request(options).map(|options| options.model_repo)
}

fn agent_options_and_paths_from_request(
    request: AgentJobRequest,
) -> Result<(Option<Vec<String>>, DepthOptions), String> {
    let AgentJobRequest {
        paths,
        model_repo,
        options,
    } = request;
    let mut options = agent_options_from_request(options)?;
    if let Some(model_repo) = model_repo {
        options.model_repo = model_repo;
    }
    Ok((paths, options))
}

fn handle_agent_route(
    request: HttpRequest,
    app: &AppHandle,
    persistent_workers: &PersistentWorkerState,
    active_jobs: &ActiveJobState,
    agent_state: &AgentServerState,
) -> Result<serde_json::Value, String> {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => Ok(serde_json::json!({
            "ok": true,
            "service": "DepthMap AI Generator",
            "version": env!("CARGO_PKG_VERSION"),
            "url": agent_status_from(agent_state, active_jobs)?.url
        })),
        ("GET", "/openapi.json") => Ok(agent_openapi(
            agent_status_from(agent_state, active_jobs)?.port,
        )),
        ("GET", "/status") => {
            let status = agent_status_from(agent_state, active_jobs)?;
            serde_json::to_value(status).map_err(|error| error.to_string())
        }
        ("GET", "/runtime") => {
            serde_json::to_value(check_runtime(app.clone())?).map_err(|error| error.to_string())
        }
        ("GET", "/vram") => {
            serde_json::to_value(get_vram_status()).map_err(|error| error.to_string())
        }
        ("GET", "/models") => {
            serde_json::to_value(get_model_status(app.clone())?).map_err(|error| error.to_string())
        }
        ("POST", "/download-model") | ("POST", "/prepare") => {
            let request = parse_agent_job_request(&request.body)?;
            let model_repo = agent_model_repo_from_request(request)?;
            let job_id = download_model(app.clone(), model_repo)?;
            Ok(serde_json::json!({ "ok": true, "jobId": job_id }))
        }
        ("POST", "/preload") => {
            let request = parse_agent_job_request(&request.body)?;
            let (_, mut options) = agent_options_and_paths_from_request(request)?;
            options.generate_depth = true;
            options.keep_model_loaded = true;
            let job_id = enqueue_persistent_job(
                app.clone(),
                persistent_workers,
                active_jobs,
                Vec::new(),
                options,
                true,
            )?;
            Ok(serde_json::json!({ "ok": true, "jobId": job_id }))
        }
        ("POST", "/generate") => {
            let request = parse_agent_job_request(&request.body)?;
            let (paths, options) = agent_options_and_paths_from_request(request)?;
            let paths = resolve_inputs(paths.unwrap_or_default(), options.recursive);
            let job_id =
                start_depth_job_core(app.clone(), persistent_workers, active_jobs, paths, options)?;
            Ok(serde_json::json!({ "ok": true, "jobId": job_id }))
        }
        ("POST", "/cancel") => {
            let pid = active_jobs.request_cancel()?;
            kill_process_tree(pid)?;
            Ok(serde_json::json!({ "ok": true }))
        }
        ("POST", "/stop-worker") => {
            stop_persistent_worker_core(persistent_workers, active_jobs)?;
            Ok(serde_json::json!({ "ok": true }))
        }
        _ => Err(format!(
            "No agent endpoint for {} {}",
            request.method, request.path
        )),
    }
}

fn handle_agent_stream(
    mut stream: TcpStream,
    app: &AppHandle,
    persistent_workers: &PersistentWorkerState,
    active_jobs: &ActiveJobState,
    agent_state: &AgentServerState,
) {
    let result = read_http_request(&mut stream).and_then(|request| {
        if request.method == "OPTIONS" {
            return write_empty_response(&mut stream, "204 No Content");
        }
        match handle_agent_route(request, app, persistent_workers, active_jobs, agent_state) {
            Ok(payload) => write_json_response(&mut stream, "200 OK", payload),
            Err(error) => write_json_response(
                &mut stream,
                "400 Bad Request",
                serde_json::json!({ "ok": false, "error": error }),
            ),
        }
    });
    if let Err(error) = result {
        let _ = write_json_response(
            &mut stream,
            "400 Bad Request",
            serde_json::json!({ "ok": false, "error": error }),
        );
    }
}

fn run_agent_server(
    listener: TcpListener,
    app: AppHandle,
    persistent_workers: PersistentWorkerState,
    active_jobs: ActiveJobState,
    agent_state: AgentServerState,
    stop: Arc<AtomicBool>,
) {
    let _ = listener.set_nonblocking(true);
    while !stop.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => handle_agent_stream(
                stream,
                &app,
                &persistent_workers,
                &active_jobs,
                &agent_state,
            ),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(80));
            }
            Err(_) => {
                thread::sleep(Duration::from_millis(150));
            }
        }
    }
}

#[tauri::command]
fn get_agent_server_status(
    agent_state: State<'_, AgentServerState>,
    active_jobs: State<'_, ActiveJobState>,
) -> Result<AgentServerStatus, String> {
    agent_status_from(agent_state.inner(), active_jobs.inner())
}

fn set_agent_server_enabled_inner(
    app: AppHandle,
    agent_state: &AgentServerState,
    persistent_workers: &PersistentWorkerState,
    active_jobs: &ActiveJobState,
    enabled: bool,
    port: Option<u16>,
) -> Result<AgentServerStatus, String> {
    let port = validate_agent_api_port(port)?;
    {
        let mut control = agent_state
            .inner
            .lock()
            .map_err(|_| "Unable to lock agent server state.".to_string())?;

        if control.enabled && (!enabled || control.port() != port) {
            if let Some(stop) = control.stop.take() {
                stop.store(true, Ordering::SeqCst);
            }
            control.enabled = false;
        }
        control.port = port;

        if enabled && !control.enabled {
            let listener = TcpListener::bind(("127.0.0.1", port))
                .map_err(|error| format!("Unable to start Agent API: {error}"))?;
            let stop = Arc::new(AtomicBool::new(false));
            thread::spawn({
                let app = app.clone();
                let persistent_workers = persistent_workers.clone();
                let active_jobs = active_jobs.clone();
                let agent_state = agent_state.clone();
                let stop = stop.clone();
                move || {
                    run_agent_server(
                        listener,
                        app,
                        persistent_workers,
                        active_jobs,
                        agent_state,
                        stop,
                    )
                }
            });
            control.enabled = true;
            control.stop = Some(stop);
        }
    }

    agent_status_from(agent_state, active_jobs)
}

#[tauri::command]
fn set_agent_server_enabled(
    app: AppHandle,
    agent_state: State<'_, AgentServerState>,
    persistent_workers: State<'_, PersistentWorkerState>,
    active_jobs: State<'_, ActiveJobState>,
    enabled: bool,
    port: Option<u16>,
) -> Result<AgentServerStatus, String> {
    set_agent_server_enabled_inner(
        app,
        agent_state.inner(),
        persistent_workers.inner(),
        active_jobs.inner(),
        enabled,
        port,
    )
}

fn env_truthy(name: &str) -> bool {
    env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn agent_api_headless_requested() -> bool {
    env::args().any(|arg| arg == "--headless" || arg == "--serve-agent-api")
        || env_truthy("NEKO_AGENT_HEADLESS")
        || env_truthy("DEPTHMAP_AGENT_HEADLESS")
}

fn agent_api_cli_port() -> Option<u16> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--agent-api-port" || arg == "--agent-port" {
            return args.next().and_then(|value| value.parse::<u16>().ok());
        }
        if let Some(value) = arg
            .strip_prefix("--agent-api-port=")
            .or_else(|| arg.strip_prefix("--agent-port="))
        {
            return value.parse::<u16>().ok();
        }
    }
    env::var("DEPTHMAP_AGENT_API_PORT")
        .or_else(|_| env::var("NEKO_AGENT_API_PORT"))
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
}

fn hide_main_window(app: &tauri::App) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.hide();
    }
}

fn clamp_window_dimension(value: u32, min: u32, monitor_max: u32) -> u32 {
    let max = monitor_max.max(1);
    value.max(min.min(max)).min(max)
}

fn initial_window_size(monitor_width: u32, monitor_height: u32) -> tauri::PhysicalSize<u32> {
    let width_ratio = if monitor_width < monitor_height {
        PORTRAIT_WINDOW_WIDTH_RATIO
    } else {
        LANDSCAPE_WINDOW_WIDTH_RATIO
    };
    let aspect = DEFAULT_WINDOW_HEIGHT as f64 / DEFAULT_WINDOW_WIDTH as f64;
    let mut width = (monitor_width as f64 * width_ratio).round();
    let mut height = (width * aspect).round();
    let max_height = (monitor_height as f64 * 0.9).round().max(1.0);

    if height > max_height {
        height = max_height;
        width = (height / aspect).round();
    }

    tauri::PhysicalSize::new(width.max(1.0) as u32, height.max(1.0) as u32)
}

fn apply_initial_window_size(app: &tauri::App) {
    let Some(window) = app.get_webview_window("main") else {
        return;
    };
    let monitor_size = window
        .current_monitor()
        .ok()
        .flatten()
        .or_else(|| window.primary_monitor().ok().flatten())
        .map(|monitor| *monitor.size());
    let monitor_width = monitor_size
        .as_ref()
        .map(|size| size.width)
        .unwrap_or(DEFAULT_WINDOW_WIDTH);
    let monitor_height = monitor_size
        .as_ref()
        .map(|size| size.height)
        .unwrap_or(DEFAULT_WINDOW_HEIGHT);
    let size = initial_window_size(monitor_width, monitor_height);
    let size = tauri::PhysicalSize::new(
        clamp_window_dimension(size.width, MIN_WINDOW_WIDTH, monitor_width),
        clamp_window_dimension(size.height, MIN_WINDOW_HEIGHT, monitor_height),
    );

    if let Err(error) = window.set_size(tauri::Size::Physical(size)) {
        eprintln!("Failed to initialize window size: {error}");
    }
    let _ = window.center();
}

pub fn run() {
    tauri::Builder::default()
        .manage(ActiveJobState::default())
        .manage(PersistentWorkerState::default())
        .manage(AgentServerState::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let headless = agent_api_headless_requested();
            if !headless {
                apply_initial_window_size(app);
            }
            if headless {
                hide_main_window(app);
                let handle = app.handle().clone();
                let agent_state = app.state::<AgentServerState>().inner().clone();
                let persistent_workers = app.state::<PersistentWorkerState>().inner().clone();
                let active_jobs = app.state::<ActiveJobState>().inner().clone();
                if let Err(error) = set_agent_server_enabled_inner(
                    handle,
                    &agent_state,
                    &persistent_workers,
                    &active_jobs,
                    true,
                    agent_api_cli_port(),
                ) {
                    eprintln!("Unable to start Agent API: {error}");
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            cancel_active_job,
            check_runtime,
            get_agent_server_status,
            get_model_status,
            get_vram_status,
            download_model,
            open_model_folder,
            open_package_report,
            open_containing_folder,
            preload_model,
            preview_image_data_url,
            resolve_inputs,
            set_agent_server_enabled,
            start_depth_job,
            stop_persistent_worker
        ])
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}
