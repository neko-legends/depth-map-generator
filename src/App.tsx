import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { getCurrentWindow, PhysicalSize } from "@tauri-apps/api/window";
import {
  CheckCircle2,
  Cpu,
  Download,
  FolderOpen,
  Image,
  Loader2,
  Play,
  RotateCcw,
  Settings2,
  Sparkles,
  TriangleAlert,
  XCircle
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

type Device = "auto" | "cuda" | "mps" | "cpu";
type OutputFormat = "png8" | "png16" | "npy";
type ModelRepo = "depth-anything/DA3-LARGE-1.1" | "depth-anything/DA3-SMALL";
type ResizeMode = "none" | "width" | "height" | "long";
type QueueStatus = "pending" | "running" | "done" | "error";

type RuntimeStatus = {
  pythonFound: boolean;
  pythonCommand: string | null;
  appDataDir: string;
  workerPath: string;
};

type GpuVram = {
  name: string;
  memoryUsedMb: number;
  memoryTotalMb: number;
};

type VramStatus = {
  available: boolean;
  gpus: GpuVram[];
  message: string;
};

type DepthOptions = {
  device: Device;
  modelRepo: ModelRepo;
  outputFormat: OutputFormat;
  generateDepth: boolean;
  convertWebp: boolean;
  webpQuality: number;
  invert: boolean;
  recursive: boolean;
  overwrite: boolean;
  suffix: string;
  resizeMode: ResizeMode;
  resizeValue: number;
};

type QueueItem = {
  path: string;
  outputPath?: string;
  status: QueueStatus;
  message?: string;
};

type WorkerEvent = {
  type:
    | "setup"
    | "model"
    | "image_start"
    | "image_done"
    | "image_error"
    | "done"
    | "error";
  message?: string;
  path?: string;
  output?: string;
};

const defaultOptions: DepthOptions = {
  device: "auto",
  modelRepo: "depth-anything/DA3-LARGE-1.1",
  outputFormat: "png8",
  generateDepth: true,
  convertWebp: false,
  webpQuality: 92,
  invert: true,
  recursive: true,
  overwrite: false,
  suffix: "_depth",
  resizeMode: "none",
  resizeValue: 1024
};

const fileName = (path: string) => path.split(/[\\/]/).pop() ?? path;
const isTauriRuntime = () => "__TAURI_INTERNALS__" in window || "__TAURI__" in window;

function statusIcon(status: QueueStatus) {
  if (status === "running") return <Loader2 className="spin" size={16} />;
  if (status === "done") return <CheckCircle2 size={16} />;
  if (status === "error") return <XCircle size={16} />;
  return <Image size={16} />;
}

function formatVram(vram: VramStatus | null) {
  if (!vram || !vram.available || vram.gpus.length === 0) {
    return { value: "--", label: "VRAM unavailable" };
  }

  const used = vram.gpus.reduce((sum, gpu) => sum + gpu.memoryUsedMb, 0);
  const total = vram.gpus.reduce((sum, gpu) => sum + gpu.memoryTotalMb, 0);
  const gpuLabel = vram.gpus.length === 1 ? vram.gpus[0].name : `${vram.gpus.length} GPUs`;
  return {
    value: `${(used / 1024).toFixed(1)}/${(total / 1024).toFixed(1)} GB`,
    label: gpuLabel
  };
}

function modelName(modelRepo: ModelRepo) {
  return modelRepo === "depth-anything/DA3-SMALL" ? "DA3-SMALL" : "DA3-LARGE-1.1";
}

function App() {
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [vram, setVram] = useState<VramStatus | null>(null);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [options, setOptions] = useState<DepthOptions>(defaultOptions);
  const [busy, setBusy] = useState(false);
  const [headline, setHeadline] = useState("Drop images or folders");
  const [log, setLog] = useState<string[]>([]);
  const [dragging, setDragging] = useState(false);

  const completeCount = useMemo(
    () => queue.filter((item) => item.status === "done").length,
    [queue]
  );
  const errorCount = useMemo(
    () => queue.filter((item) => item.status === "error").length,
    [queue]
  );
  const vramDisplay = useMemo(() => formatVram(vram), [vram]);

  const pushLog = useCallback((line: string) => {
    setLog((current) => [line, ...current].slice(0, 80));
  }, []);

  useEffect(() => {
    const updateScale = () => {
      const scale = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
      document.documentElement.style.setProperty("--ui-scale", String(scale));
    };
    updateScale();
    window.addEventListener("resize", updateScale);
    return () => window.removeEventListener("resize", updateScale);
  }, []);

  useEffect(() => {
    if (!isTauriRuntime()) {
      setVram({ available: false, gpus: [], message: "Preview mode" });
      return;
    }

    const refreshVram = async () => {
      try {
        setVram(await invoke<VramStatus>("get_vram_status"));
      } catch (error) {
        setVram({ available: false, gpus: [], message: String(error) });
      }
    };

    void refreshVram();
    const interval = window.setInterval(refreshVram, 2500);
    return () => window.clearInterval(interval);
  }, []);

  const resolvePaths = useCallback(
    async (paths: string[]) => {
      if (paths.length === 0) return;
      let resolved: string[];
      try {
        resolved = await invoke<string[]>("resolve_inputs", {
          paths,
          recursive: options.recursive
        });
      } catch (error) {
        pushLog(`Could not read dropped/selected paths: ${String(error)}`);
        return;
      }
      setQueue((current) => {
        const known = new Set(current.map((item) => item.path));
        const additions = resolved
          .filter((path) => !known.has(path))
          .map((path) => ({ path, status: "pending" as const }));
        return [...current, ...additions];
      });
      setHeadline(`${resolved.length} image${resolved.length === 1 ? "" : "s"} ready`);
      pushLog(`Queued ${resolved.length} image${resolved.length === 1 ? "" : "s"}.`);
    },
    [options.recursive, pushLog]
  );

  useEffect(() => {
    if (!isTauriRuntime()) {
      setRuntime({
        pythonFound: true,
        pythonCommand: "Preview mode",
        appDataDir: "Browser preview",
        workerPath: "Tauri runtime required"
      });
      pushLog("Browser preview mode. Launch with Tauri to use local files and DA3.");
      return;
    }

    invoke<RuntimeStatus>("check_runtime")
      .then(setRuntime)
      .catch((error) => pushLog(String(error)));

    const unlistenPromise = listen<WorkerEvent>("depth-worker-event", (event) => {
      const payload = event.payload;
      if (payload.message) {
        setHeadline(payload.message);
        pushLog(payload.message);
      }

      if (payload.type === "image_start" && payload.path) {
        setQueue((current) =>
          current.map((item) =>
            item.path === payload.path
              ? { ...item, status: "running", message: "Processing" }
              : item
          )
        );
      }

      if (payload.type === "image_done" && payload.path) {
        setQueue((current) =>
          current.map((item) =>
            item.path === payload.path
              ? {
                  ...item,
                  status: "done",
                  outputPath: payload.output,
                  message: "Saved"
                }
              : item
          )
        );
      }

      if (payload.type === "image_error" && payload.path) {
        setQueue((current) =>
          current.map((item) =>
            item.path === payload.path
              ? {
                  ...item,
                  status: "error",
                  message: payload.message ?? "Failed"
                }
              : item
          )
        );
      }

      if (payload.type === "done" || payload.type === "error") {
        setBusy(false);
      }
    });

    const dragPromise = getCurrentWindow().onDragDropEvent((event) => {
      if (event.payload.type === "over") {
        setDragging(true);
      }
      if (event.payload.type === "drop") {
        setDragging(false);
        pushLog(`Dropped ${event.payload.paths.length} item${event.payload.paths.length === 1 ? "" : "s"}.`);
        void resolvePaths(event.payload.paths);
      }
      if (event.payload.type === "leave") {
        setDragging(false);
      }
    });
    let aspectTimer: number | undefined;
    const enforceAspectRatio = async () => {
      const windowHandle = getCurrentWindow();
      const size = await windowHandle.outerSize();
      let width = Math.max(640, size.width);
      let height = Math.max(360, size.height);
      const heightFromWidth = Math.round(width * 9 / 16);
      const widthFromHeight = Math.round(height * 16 / 9);

      if (Math.abs(heightFromWidth - height) <= Math.abs(widthFromHeight - width)) {
        height = heightFromWidth;
      } else {
        width = widthFromHeight;
      }

      if (width < 640 || height < 360) {
        width = 640;
        height = 360;
      }

      if (Math.abs(width - size.width) > 1 || Math.abs(height - size.height) > 1) {
        await windowHandle.setSize(new PhysicalSize(width, height));
      }
    };
    const aspectPromise = getCurrentWindow().onResized(() => {
      if (aspectTimer !== undefined) {
        window.clearTimeout(aspectTimer);
      }
      aspectTimer = window.setTimeout(() => {
        void enforceAspectRatio();
      }, 160);
    });

    return () => {
      if (aspectTimer !== undefined) {
        window.clearTimeout(aspectTimer);
      }
      void unlistenPromise.then((unlisten) => unlisten());
      void dragPromise.then((unlisten) => unlisten());
      void aspectPromise.then((unlisten) => unlisten());
    };
  }, [pushLog, resolvePaths]);

  const chooseInputs = async () => {
    if (!isTauriRuntime()) {
      pushLog("File picking requires the Tauri desktop shell.");
      return;
    }
    let selected: string | string[] | null;
    try {
      selected = await open({
        multiple: true,
        directory: false,
        filters: [
          {
            name: "Images",
            extensions: ["png", "jpg", "jpeg", "jpe", "jfif", "jpf", "jp2", "j2k", "webp", "bmp", "tif", "tiff"]
          }
        ]
      });
    } catch (error) {
      pushLog(`Image picker failed: ${String(error)}`);
      return;
    }
    if (Array.isArray(selected)) {
      await resolvePaths(selected);
    } else if (typeof selected === "string") {
      await resolvePaths([selected]);
    } else {
      pushLog("Image picker canceled.");
    }
  };

  const chooseFolder = async () => {
    if (!isTauriRuntime()) {
      pushLog("Folder picking requires the Tauri desktop shell.");
      return;
    }
    let selected: string | string[] | null;
    try {
      selected = await open({ multiple: true, directory: true });
    } catch (error) {
      pushLog(`Folder picker failed: ${String(error)}`);
      return;
    }
    if (Array.isArray(selected)) {
      await resolvePaths(selected);
    } else if (typeof selected === "string") {
      await resolvePaths([selected]);
    } else {
      pushLog("Folder picker canceled.");
    }
  };

  const start = async () => {
    if (!isTauriRuntime()) {
      pushLog("Generation requires the Tauri desktop shell.");
      return;
    }
    const pending = queue
      .filter((item) => item.status === "pending" || item.status === "error")
      .map((item) => item.path);
    if (pending.length === 0 || busy) return;
    if (!options.generateDepth && !options.convertWebp) {
      pushLog("Enable depth generation, WebP conversion, or both.");
      return;
    }
    setBusy(true);
    setQueue((current) =>
      current.map((item) =>
        pending.includes(item.path) ? { ...item, status: "pending", message: undefined } : item
      )
    );
    try {
      await invoke("start_depth_job", { paths: pending, options });
    } catch (error) {
      setBusy(false);
      pushLog(String(error));
      setHeadline("Unable to start worker");
    }
  };

  const reset = () => {
    setQueue([]);
    setLog([]);
    setHeadline("Drop images or folders");
  };

  return (
    <div className="scale-stage">
      <main className="app-shell">
        <aside className="sidebar">
          <div className="brand">
            <div className="brand-mark">
              <Sparkles size={22} />
            </div>
            <div>
              <h1>DepthMap AI</h1>
              <p>DA3-LARGE-1.1 local depth maps</p>
            </div>
          </div>

          <section className="panel">
          <div className="panel-title">
            <Settings2 size={16} />
            Settings
          </div>

          <label className="field">
            <span>Device</span>
            <select
              value={options.device}
              onChange={(event) =>
                setOptions((current) => ({
                  ...current,
                  device: event.target.value as Device
                }))
              }
            >
              <option value="auto">Auto</option>
              <option value="cuda">CUDA</option>
              <option value="mps">Apple MPS</option>
              <option value="cpu">CPU</option>
            </select>
          </label>

          <label className="field">
            <span>Depth format</span>
            <select
              value={options.outputFormat}
              disabled={!options.generateDepth}
              onChange={(event) =>
                setOptions((current) => ({
                  ...current,
                  outputFormat: event.target.value as OutputFormat
                }))
              }
            >
              <option value="png8">PNG 8-bit</option>
              <option value="png16">PNG 16-bit</option>
              <option value="npy">NumPy .npy</option>
            </select>
          </label>

          <label className="toggle">
            <input
              type="checkbox"
              checked={options.generateDepth}
              onChange={(event) =>
                setOptions((current) => ({ ...current, generateDepth: event.target.checked }))
              }
            />
            <span>Generate depth map</span>
          </label>

          <label className="field">
            <span>Suffix</span>
            <input
              value={options.suffix}
              disabled={!options.generateDepth}
              onChange={(event) =>
                setOptions((current) => ({ ...current, suffix: event.target.value }))
              }
            />
          </label>

          <div className="split-fields">
            <label className="field">
              <span>Resize</span>
              <select
                value={options.resizeMode}
                disabled={!options.generateDepth}
                onChange={(event) =>
                  setOptions((current) => ({
                    ...current,
                    resizeMode: event.target.value as ResizeMode
                  }))
                }
              >
                <option value="none">Original</option>
                <option value="width">Peg width</option>
                <option value="height">Peg height</option>
                <option value="long">Peg long side</option>
              </select>
            </label>

            <label className="field">
              <span>Pixels</span>
              <input
                type="number"
                min="64"
                max="8192"
                step="64"
                disabled={!options.generateDepth || options.resizeMode === "none"}
                aria-disabled={!options.generateDepth || options.resizeMode === "none"}
                value={options.resizeValue}
                onChange={(event) =>
                  setOptions((current) => ({
                    ...current,
                    resizeValue: Number(event.target.value) || current.resizeValue
                  }))
                }
              />
            </label>
          </div>

          <label className="toggle">
            <input
              type="checkbox"
              checked={options.invert}
              disabled={!options.generateDepth}
              onChange={(event) =>
                setOptions((current) => ({ ...current, invert: event.target.checked }))
              }
            />
            <span>Invert depth map</span>
          </label>

          <label className="toggle">
            <input
              type="checkbox"
              checked={options.convertWebp}
              onChange={(event) =>
                setOptions((current) => ({ ...current, convertWebp: event.target.checked }))
              }
            />
            <span>Convert source to WebP</span>
          </label>

          <label className="field compact-field">
            <span>WebP quality</span>
            <input
              type="number"
              min="1"
              max="100"
              disabled={!options.convertWebp}
              value={options.webpQuality}
              onChange={(event) =>
                setOptions((current) => ({
                  ...current,
                  webpQuality: Number(event.target.value) || current.webpQuality
                }))
              }
            />
          </label>

          <label className="toggle">
            <input
              type="checkbox"
              checked={options.recursive}
              onChange={(event) =>
                setOptions((current) => ({ ...current, recursive: event.target.checked }))
              }
            />
            <span>Scan folders recursively</span>
          </label>

          <label className="toggle">
            <input
              type="checkbox"
              checked={options.overwrite}
              onChange={(event) =>
                setOptions((current) => ({ ...current, overwrite: event.target.checked }))
              }
            />
            <span>Overwrite existing outputs</span>
          </label>
          </section>

          <section className="runtime">
          <div className="panel-title">
            <Cpu size={16} />
            Runtime
          </div>
          {runtime?.pythonFound ? (
            <p className="ok">Python: {runtime.pythonCommand}</p>
          ) : (
            <p className="warn">Python 3.10+ not found on PATH.</p>
          )}
          <label className="field runtime-model">
            <span>Depth model</span>
            <select
              value={options.modelRepo}
              disabled={!options.generateDepth}
              onChange={(event) =>
                setOptions((current) => ({
                  ...current,
                  modelRepo: event.target.value as ModelRepo
                }))
              }
            >
              <option value="depth-anything/DA3-LARGE-1.1">DA3-LARGE-1.1</option>
              <option value="depth-anything/DA3-SMALL">DA3-SMALL</option>
            </select>
          </label>
          <p className="muted">
            {options.generateDepth
              ? `${modelName(options.modelRepo)} downloads on first run.`
              : "Model downloads only when depth generation is enabled."}
          </p>
          </section>
        </aside>

        <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Local batch generator</p>
            <h2>{headline}</h2>
          </div>
          <div className="actions">
            <button className="secondary" onClick={chooseInputs}>
              <Image size={17} />
              Images
            </button>
            <button className="secondary" onClick={chooseFolder}>
              <FolderOpen size={17} />
              Folder
            </button>
            <button className="secondary icon" onClick={reset} title="Clear queue">
              <RotateCcw size={17} />
            </button>
            <button
              className="primary"
              disabled={busy || queue.length === 0 || (!options.generateDepth && !options.convertWebp)}
              onClick={start}
            >
              {busy ? <Loader2 className="spin" size={17} /> : <Play size={17} />}
              Generate
            </button>
          </div>
        </header>

        <div className="stats">
          <div>
            <span>{queue.length}</span>
            queued
          </div>
          <div>
            <span>{completeCount}</span>
            saved
          </div>
          <div>
            <span>{errorCount}</span>
            failed
          </div>
          <div>
            <span className="vram-value">{vramDisplay.value}</span>
            {vramDisplay.label}
          </div>
        </div>

        <section className={`drop-zone ${dragging ? "dragging" : ""}`}>
          <Download size={42} />
          <div>
            <h3>Drop images or folders here</h3>
            <p>Depth maps and WebP files are written beside each source image.</p>
          </div>
        </section>

        <section className="queue">
          <div className="queue-header">
            <span>Input</span>
            <span>Status</span>
            <span>Outputs</span>
          </div>
          {queue.length === 0 ? (
            <div className="empty">
              <TriangleAlert size={18} />
              No images queued.
            </div>
          ) : (
            queue.map((item) => (
              <div className={`queue-row ${item.status}`} key={item.path}>
                <div>
                  <strong>{fileName(item.path)}</strong>
                  <small>{item.path}</small>
                </div>
                <div className="status">
                  {statusIcon(item.status)}
                  {item.message ?? item.status}
                </div>
                <small>{item.outputPath ?? "-"}</small>
              </div>
            ))
          )}
        </section>

        <section className="log">
          {log.length === 0 ? <p>Worker messages will appear here.</p> : null}
          {log.map((line, index) => (
            <p key={`${line}-${index}`}>{line}</p>
          ))}
        </section>
        </section>
      </main>
    </div>
  );
}

export default App;
