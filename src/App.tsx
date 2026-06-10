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
  Package,
  Play,
  RotateCcw,
  Settings2,
  Square,
  TriangleAlert,
  X,
  XCircle
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Device = "auto" | "cuda" | "mps" | "cpu" | `cuda:${number}`;
type OutputFormat = "png8" | "png16" | "npy";
type ModelRepo =
  | "sy000/MDA/mda_mog_sky_l2"
  | "depth-anything/DA3-LARGE-1.1"
  | "depth-anything/DA3-SMALL";
type ResizeMode = "none" | "width" | "height" | "long";
type ProcessResMethod =
  | "upper_bound_resize"
  | "lower_bound_resize"
  | "upper_bound_crop"
  | "lower_bound_crop";
type QueueStatus = "pending" | "running" | "done" | "error" | "canceled";

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

type ModelStatus = {
  repo: ModelRepo;
  name: string;
  installed: boolean;
  modelPath: string;
  sizeBytes: number;
};

type AgentServerStatus = {
  enabled: boolean;
  port: number;
  url: string;
  openapiUrl: string;
  busy: boolean;
  activeJobId: string | null;
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
  autoGenerateOnChange: boolean;
  outputDir: string;
  suffix: string;
  resizeMode: ResizeMode;
  resizeValue: number;
  processRes: number;
  processResMethod: ProcessResMethod;
  normalizeLow: number;
  normalizeHigh: number;
  depthGamma: number;
  keepModelLoaded: boolean;
};

type QueueItem = {
  path: string;
  outputHistory?: string[];
  status: QueueStatus;
  message?: string;
};

type WorkerEvent = {
  type:
    | "setup"
    | "model"
    | "model_ready"
    | "download_progress"
    | "image_start"
    | "image_done"
    | "image_error"
    | "done"
    | "error"
    | "canceled";
  message?: string;
  path?: string;
  output?: string;
  modelRepo?: ModelRepo;
  device?: string;
  bytesDownloaded?: number;
  bytesTotal?: number;
  percent?: number | null;
};

type DownloadProgress = {
  modelRepo: ModelRepo;
  message: string;
  bytesDownloaded: number;
  bytesTotal: number;
  percent: number | null;
};

type PreviewImages = {
  input?: string;
  output?: string;
  error?: string;
};

type ImageModal = {
  path: string;
  title: string;
  dataUrl?: string;
  error?: string;
};

const defaultOptions: DepthOptions = {
  device: "auto",
  modelRepo: "sy000/MDA/mda_mog_sky_l2",
  outputFormat: "png8",
  generateDepth: true,
  convertWebp: false,
  webpQuality: 92,
  invert: true,
  recursive: true,
  overwrite: false,
  autoGenerateOnChange: false,
  outputDir: "",
  suffix: "_depth",
  resizeMode: "none",
  resizeValue: 1024,
  processRes: 504,
  processResMethod: "upper_bound_resize",
  normalizeLow: 1,
  normalizeHigh: 99,
  depthGamma: 1,
  keepModelLoaded: false
};

const SETTINGS_STORAGE_KEY = "depthmapai.settings.v1";
const AGENT_STORAGE_KEY = "depthmapai.agentControlEnabled.v1";
const AGENT_PORT_STORAGE_KEY = "depthmapai.agentApiPort.v1";
const DEFAULT_AGENT_API_PORT = 17334;
const MDA_DEFAULT_MIGRATION_KEY = "depthmapai.mdaDefaultMigration.v1";
const KEEP_LOADED_DEFAULT_MIGRATION_KEY = "depthmapai.keepLoadedDefaultMigration.v1";
const MODEL_PROMPT_SEEN_KEY = "depthmapai.modelPromptSeen.v1";
const APP_VERSION = "v26.6.5";
const DEVICE_OPTIONS: Device[] = ["auto", "cuda", "mps", "cpu"];
const OUTPUT_FORMAT_OPTIONS: OutputFormat[] = ["png8", "png16", "npy"];
const MODEL_REPO_OPTIONS: ModelRepo[] = [
  "sy000/MDA/mda_mog_sky_l2",
  "depth-anything/DA3-LARGE-1.1",
  "depth-anything/DA3-SMALL"
];
const RESIZE_MODE_OPTIONS: ResizeMode[] = ["none", "width", "height", "long"];
const PROCESS_RES_METHOD_OPTIONS: ProcessResMethod[] = [
  "upper_bound_resize",
  "lower_bound_resize",
  "upper_bound_crop",
  "lower_bound_crop"
];
const MODEL_DETAILS: Record<ModelRepo, { name: string; size: string; vram: string; note: string }> = {
  "sy000/MDA/mda_mog_sky_l2": {
    name: "MDA-DA3-MoG-Sky",
    size: "5.42 GB",
    vram: "Recommended: 12 GB+ VRAM",
    note: "Sharper edge depth and fewer flying points."
  },
  "depth-anything/DA3-LARGE-1.1": {
    name: "DA3-LARGE-1.1",
    size: "1.64 GB",
    vram: "Recommended: 8 GB+ VRAM",
    note: "Strong fallback with a smaller download."
  },
  "depth-anything/DA3-SMALL": {
    name: "DA3-SMALL",
    size: "137 MB",
    vram: "Recommended: 4 GB+ VRAM",
    note: "Lower VRAM, faster download."
  }
};
const LOADING_NOISE = ["!@!#$%%^^&()", "#$%%^^&()!@!", "%%^^&()!@!#$", "^^&()!@!#$%%", "&()!@!#$%%^^"];

const fileName = (path: string) => path.split(/[\\/]/).pop() ?? path;
const isTauriRuntime = () => "__TAURI_INTERNALS__" in window || "__TAURI__" in window;

function normalizeAgentPort(value: string) {
  const port = Number(value);
  return Number.isInteger(port) && port >= 1 && port <= 65535 ? port : null;
}

function coerceChoice<T extends string>(value: unknown, choices: T[], fallback: T) {
  return typeof value === "string" && choices.includes(value as T) ? (value as T) : fallback;
}

function coerceDevice(value: unknown, fallback: Device) {
  if (typeof value !== "string") return fallback;
  if (DEVICE_OPTIONS.includes(value as Device) || /^cuda:\d+$/.test(value)) return value as Device;
  return fallback;
}

function coerceBoolean(value: unknown, fallback: boolean) {
  return typeof value === "boolean" ? value : fallback;
}

function coerceNumber(value: unknown, fallback: number, min: number, max: number) {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(min, Math.min(max, value))
    : fallback;
}

function loadSavedOptions(): DepthOptions {
  if (typeof window === "undefined") return defaultOptions;

  try {
    const saved = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!saved) return defaultOptions;
    const parsedValue = JSON.parse(saved);
    if (!parsedValue || typeof parsedValue !== "object") return defaultOptions;
    const parsed = parsedValue as Partial<DepthOptions>;
    const loaded = {
      device: coerceDevice(parsed.device, defaultOptions.device),
      modelRepo: coerceChoice(parsed.modelRepo, MODEL_REPO_OPTIONS, defaultOptions.modelRepo),
      outputFormat: coerceChoice(parsed.outputFormat, OUTPUT_FORMAT_OPTIONS, defaultOptions.outputFormat),
      generateDepth: coerceBoolean(parsed.generateDepth, defaultOptions.generateDepth),
      convertWebp: coerceBoolean(parsed.convertWebp, defaultOptions.convertWebp),
      webpQuality: coerceNumber(parsed.webpQuality, defaultOptions.webpQuality, 1, 100),
      invert: coerceBoolean(parsed.invert, defaultOptions.invert),
      recursive: coerceBoolean(parsed.recursive, defaultOptions.recursive),
      overwrite: coerceBoolean(parsed.overwrite, defaultOptions.overwrite),
      autoGenerateOnChange: coerceBoolean(
        parsed.autoGenerateOnChange,
        defaultOptions.autoGenerateOnChange
      ),
      outputDir: typeof parsed.outputDir === "string" ? parsed.outputDir : defaultOptions.outputDir,
      suffix: typeof parsed.suffix === "string" ? parsed.suffix : defaultOptions.suffix,
      resizeMode: coerceChoice(parsed.resizeMode, RESIZE_MODE_OPTIONS, defaultOptions.resizeMode),
      resizeValue: coerceNumber(parsed.resizeValue, defaultOptions.resizeValue, 64, 8192),
      processRes: coerceNumber(parsed.processRes, defaultOptions.processRes, 336, 1008),
      processResMethod: coerceChoice(
        parsed.processResMethod,
        PROCESS_RES_METHOD_OPTIONS,
        defaultOptions.processResMethod
      ),
      normalizeLow: coerceNumber(parsed.normalizeLow, defaultOptions.normalizeLow, 0, 99),
      normalizeHigh: coerceNumber(parsed.normalizeHigh, defaultOptions.normalizeHigh, 1, 100),
      depthGamma: coerceNumber(parsed.depthGamma, defaultOptions.depthGamma, 0.2, 4),
      keepModelLoaded: coerceBoolean(parsed.keepModelLoaded, defaultOptions.keepModelLoaded)
    };
    if (loaded.normalizeLow >= loaded.normalizeHigh) {
      loaded.normalizeLow = defaultOptions.normalizeLow;
      loaded.normalizeHigh = defaultOptions.normalizeHigh;
    }
    if (
      parsed.modelRepo === "depth-anything/DA3-LARGE-1.1" &&
      window.localStorage.getItem(MDA_DEFAULT_MIGRATION_KEY) !== "done"
    ) {
      loaded.modelRepo = defaultOptions.modelRepo;
      window.localStorage.setItem(MDA_DEFAULT_MIGRATION_KEY, "done");
    }
    if (window.localStorage.getItem(KEEP_LOADED_DEFAULT_MIGRATION_KEY) !== "done") {
      loaded.keepModelLoaded = defaultOptions.keepModelLoaded;
      window.localStorage.setItem(KEEP_LOADED_DEFAULT_MIGRATION_KEY, "done");
    }
    return loaded;
  } catch {
    return defaultOptions;
  }
}

function statusIcon(status: QueueStatus) {
  if (status === "running") return <Loader2 className="spin" size={16} />;
  if (status === "done") return <CheckCircle2 size={16} />;
  if (status === "error" || status === "canceled") return <XCircle size={16} />;
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

function gpuDeviceOptions(vram: VramStatus | null): Array<{ value: Device; label: string }> {
  const options: Array<{ value: Device; label: string }> = [{ value: "auto", label: "Auto" }];
  if (vram?.available && vram.gpus.length > 0) {
    vram.gpus.forEach((gpu, index) => {
      options.push({ value: `cuda:${index}` as Device, label: `GPU ${index} - ${gpu.name}` });
    });
  } else {
    options.push({ value: "cuda", label: "CUDA" });
  }
  options.push({ value: "cpu", label: "CPU" });
  return options;
}

function modelName(modelRepo: ModelRepo) {
  return MODEL_DETAILS[modelRepo]?.name ?? modelRepo.split("/").pop() ?? modelRepo;
}

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "--";
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  return `${Math.round(bytes / (1024 * 1024))} MB`;
}

function parseOutputPaths(output?: string) {
  return output
    ?.split(";")
    .map((path) => path.trim())
    .filter(Boolean) ?? [];
}

function isPreviewableImage(path: string) {
  return /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(path);
}

function previewOutputPath(paths?: string[]) {
  return paths?.find(isPreviewableImage);
}

function compactPath(path: string) {
  if (!path) return "Same as source";
  const parts = path.split(/[\\/]/).filter(Boolean);
  if (parts.length <= 2) return path;
  return `...${path.includes("\\") ? "\\" : "/"}${parts.slice(-2).join(path.includes("\\") ? "\\" : "/")}`;
}

function App() {
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [vram, setVram] = useState<VramStatus | null>(null);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const queueRef = useRef<QueueItem[]>([]);
  const [options, setOptions] = useState<DepthOptions>(loadSavedOptions);
  const [agentControlEnabled, setAgentControlEnabled] = useState(
    () => window.localStorage.getItem(AGENT_STORAGE_KEY) === "1"
  );
  const [agentPort, setAgentPort] = useState(
    () => window.localStorage.getItem(AGENT_PORT_STORAGE_KEY) ?? String(DEFAULT_AGENT_API_PORT)
  );
  const [agentStatus, setAgentStatus] = useState<AgentServerStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [canceling, setCanceling] = useState(false);
  const [headline, setHeadline] = useState("Drop images or folders");
  const [log, setLog] = useState<string[]>([]);
  const [dragging, setDragging] = useState(false);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [previewImages, setPreviewImages] = useState<PreviewImages>({});
  const [imageModal, setImageModal] = useState<ImageModal | null>(null);
  const [outputPreviews, setOutputPreviews] = useState<Record<string, string>>({});
  const [modelStatuses, setModelStatuses] = useState<Partial<Record<ModelRepo, ModelStatus>>>({});
  const [showModelPrompt, setShowModelPrompt] = useState(false);
  const [downloadingModel, setDownloadingModel] = useState<ModelRepo | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null);
  const [preloadingModel, setPreloadingModel] = useState<ModelRepo | null>(null);
  const [loadedModelKey, setLoadedModelKey] = useState<string | null>(null);
  const [loadingNoise, setLoadingNoise] = useState(LOADING_NOISE[0]);
  const preloadKeyRef = useRef<string | null>(null);
  const activePreloadKeyRef = useRef<string | null>(null);
  const failedPreloadKeyRef = useRef<string | null>(null);
  const preloadTimerRef = useRef<number | null>(null);
  const autoGenerateTimerRef = useRef<number | null>(null);
  const previousAutoGenerateKeyRef = useRef<string | null>(null);
  const autoInstallModelKeyRef = useRef<string | null>(null);

  const completeCount = useMemo(
    () => queue.filter((item) => item.status === "done").length,
    [queue]
  );
  const errorCount = useMemo(
    () => queue.filter((item) => item.status === "error").length,
    [queue]
  );
  const vramDisplay = useMemo(() => formatVram(vram), [vram]);
  const gpuOptions = useMemo(() => gpuDeviceOptions(vram), [vram]);
  const selectedModelStatus = modelStatuses[options.modelRepo];
  const activeDownload = downloadProgress;
  const activeModelLoad = preloadingModel && !activeDownload ? preloadingModel : null;
  const currentModelReady = loadedModelKey?.startsWith(`${options.modelRepo}:`);
  const selectedItem = useMemo(
    () => queue.find((item) => item.path === selectedPath) ?? queue[0],
    [queue, selectedPath]
  );
  const previewOutput = previewOutputPath(selectedItem?.outputHistory);
  const autoGenerateParamsKey = useMemo(
    () =>
      JSON.stringify({
        device: options.device,
        modelRepo: options.modelRepo,
        outputFormat: options.outputFormat,
        generateDepth: options.generateDepth,
        convertWebp: options.convertWebp,
        webpQuality: options.webpQuality,
        invert: options.invert,
        outputDir: options.outputDir,
        suffix: options.suffix,
        resizeMode: options.resizeMode,
        resizeValue: options.resizeValue,
        processRes: options.processRes,
        processResMethod: options.processResMethod,
        normalizeLow: options.normalizeLow,
        normalizeHigh: options.normalizeHigh,
        depthGamma: options.depthGamma
      }),
    [
      options.convertWebp,
      options.depthGamma,
      options.device,
      options.generateDepth,
      options.invert,
      options.modelRepo,
      options.normalizeHigh,
      options.normalizeLow,
      options.outputDir,
      options.outputFormat,
      options.processRes,
      options.processResMethod,
      options.resizeMode,
      options.resizeValue,
      options.suffix,
      options.webpQuality
    ]
  );

  const pushLog = useCallback((line: string) => {
    setLog((current) => [line, ...current].slice(0, 80));
  }, []);

  useEffect(() => {
    if (!activeModelLoad) {
      setLoadingNoise(LOADING_NOISE[0]);
      return;
    }
    let index = 0;
    const interval = window.setInterval(() => {
      index = (index + 1) % LOADING_NOISE.length;
      setLoadingNoise(LOADING_NOISE[index]);
    }, 180);
    return () => window.clearInterval(interval);
  }, [activeModelLoad]);

  useEffect(() => {
    queueRef.current = queue;
  }, [queue]);

  useEffect(() => {
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(options));
  }, [options]);

  useEffect(() => {
    if (!isTauriRuntime()) return;
    const imagePaths = Array.from(
      new Set(queue.flatMap((item) => item.outputHistory ?? []).filter(isPreviewableImage))
    );
    const missing = imagePaths.filter((path) => outputPreviews[path] === undefined);
    if (missing.length === 0) return;

    let canceled = false;
    const loadOutputPreviews = async () => {
      const loaded: Record<string, string> = {};
      await Promise.all(
        missing.map(async (path) => {
          try {
            loaded[path] = await invoke<string>("preview_image_data_url", { path });
          } catch {
            loaded[path] = "";
          }
        })
      );
      if (!canceled && Object.keys(loaded).length > 0) {
        setOutputPreviews((current) => ({ ...current, ...loaded }));
      }
    };

    void loadOutputPreviews();
    return () => {
      canceled = true;
    };
  }, [outputPreviews, queue]);

  useEffect(() => {
    if (!selectedItem || !isTauriRuntime()) {
      setPreviewImages({});
      return;
    }

    let canceled = false;
    setPreviewImages({});

    const loadPreview = async () => {
      try {
        const input = await invoke<string>("preview_image_data_url", { path: selectedItem.path });
        const output = previewOutput
          ? await invoke<string>("preview_image_data_url", { path: previewOutput })
          : undefined;
        if (!canceled) {
          setPreviewImages({ input, output });
        }
      } catch (error) {
        if (!canceled) {
          setPreviewImages({ error: String(error) });
        }
      }
    };

    void loadPreview();
    return () => {
      canceled = true;
    };
  }, [previewOutput, selectedItem]);

  useEffect(() => {
    if (!imageModal || imageModal.dataUrl || imageModal.error || !isTauriRuntime()) return;

    let canceled = false;
    const loadModalPreview = async () => {
      try {
        const dataUrl = await invoke<string>("preview_image_data_url", { path: imageModal.path });
        if (!canceled) {
          setImageModal((current) =>
            current?.path === imageModal.path ? { ...current, dataUrl } : current
          );
        }
      } catch (error) {
        if (!canceled) {
          setImageModal((current) =>
            current?.path === imageModal.path ? { ...current, error: String(error) } : current
          );
        }
      }
    };

    void loadModalPreview();
    return () => {
      canceled = true;
    };
  }, [imageModal]);

  useEffect(() => {
    if (!imageModal) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setImageModal(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [imageModal]);

  const refreshModelStatus = useCallback(async () => {
    if (!isTauriRuntime()) return [] as ModelStatus[];
    const statuses = await invoke<ModelStatus[]>("get_model_status");
    setModelStatuses(Object.fromEntries(statuses.map((status) => [status.repo, status])));
    return statuses;
  }, []);

  const refreshAgentStatus = useCallback(async () => {
    if (!isTauriRuntime()) return;
    const status = await invoke<AgentServerStatus>("get_agent_server_status");
    setAgentStatus(status);
    if (status.enabled || !window.localStorage.getItem(AGENT_PORT_STORAGE_KEY)) {
      setAgentPort(String(status.port || DEFAULT_AGENT_API_PORT));
    }
  }, []);

  const toggleAgentControl = useCallback(
    async (enabled: boolean) => {
      const port = normalizeAgentPort(agentPort);
      if (enabled && port === null) {
        pushLog("Choose an Agent API port between 1 and 65535.");
        return;
      }
      setAgentControlEnabled(enabled);
      window.localStorage.setItem(AGENT_STORAGE_KEY, enabled ? "1" : "0");
      if (port !== null) {
        window.localStorage.setItem(AGENT_PORT_STORAGE_KEY, String(port));
      }
      if (!isTauriRuntime()) return;
      try {
        const status = await invoke<AgentServerStatus>("set_agent_server_enabled", {
          enabled,
          port: port ?? agentStatus?.port ?? DEFAULT_AGENT_API_PORT
        });
        setAgentStatus(status);
        setAgentPort(String(status.port));
        pushLog(status.message);
      } catch (error) {
        setAgentControlEnabled(false);
        window.localStorage.setItem(AGENT_STORAGE_KEY, "0");
        pushLog(String(error));
      }
    },
    [agentPort, agentStatus?.port, pushLog]
  );

  const applyAgentPort = useCallback(async () => {
    const port = normalizeAgentPort(agentPort);
    if (port === null) {
      pushLog("Choose an Agent API port between 1 and 65535.");
      return;
    }
    window.localStorage.setItem(AGENT_PORT_STORAGE_KEY, String(port));
    if (!isTauriRuntime() || !agentControlEnabled) return;
    try {
      const status = await invoke<AgentServerStatus>("set_agent_server_enabled", {
        enabled: true,
        port
      });
      setAgentStatus(status);
      setAgentPort(String(status.port));
      pushLog(`Agent API moved to ${status.url}`);
    } catch (error) {
      pushLog(String(error));
    }
  }, [agentControlEnabled, pushLog]);

  const markModelPromptSeen = useCallback(() => {
    window.localStorage.setItem(MODEL_PROMPT_SEEN_KEY, "1");
    setShowModelPrompt(false);
  }, []);

  const downloadSelectedModel = useCallback(
    async (modelRepo: ModelRepo, fromPrompt = false) => {
      if (!isTauriRuntime()) {
        pushLog("Model downloads require the Tauri desktop shell.");
        return;
      }
      if (busy || downloadingModel) {
        pushLog("A job is already running.");
        return;
      }
      if (fromPrompt) {
        markModelPromptSeen();
      }
      setOptions((current) => ({ ...current, modelRepo, generateDepth: true }));
      setDownloadingModel(modelRepo);
      setDownloadProgress({
        modelRepo,
        message: `Preparing ${modelName(modelRepo)}`,
        bytesDownloaded: 0,
        bytesTotal: modelStatuses[modelRepo]?.sizeBytes ?? 0,
        percent: 0
      });
      setHeadline(`Preparing ${modelName(modelRepo)} download`);
      try {
        await invoke("download_model", { modelRepo });
      } catch (error) {
        setDownloadingModel(null);
        setDownloadProgress(null);
        pushLog(`Model download failed to start: ${String(error)}`);
      }
    },
    [busy, downloadingModel, markModelPromptSeen, modelStatuses, pushLog]
  );

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

  useEffect(() => {
    if (!isTauriRuntime()) return;
    void refreshAgentStatus().catch((error) => pushLog(`Agent API check failed: ${String(error)}`));
  }, [pushLog, refreshAgentStatus]);

  useEffect(() => {
    if (!isTauriRuntime() || !agentControlEnabled) return;
    const port = normalizeAgentPort(agentPort) ?? DEFAULT_AGENT_API_PORT;
    void invoke<AgentServerStatus>("set_agent_server_enabled", { enabled: true, port })
      .then((status) => {
        setAgentStatus(status);
        setAgentPort(String(status.port));
      })
      .catch((error) => {
        setAgentControlEnabled(false);
        window.localStorage.setItem(AGENT_STORAGE_KEY, "0");
        pushLog(`Agent API failed to start: ${String(error)}`);
      });
  }, [agentControlEnabled, agentPort, pushLog]);

  useEffect(() => {
    if (!isTauriRuntime()) return;
    void refreshModelStatus()
      .then((statuses) => {
        const selectedStatus = statuses.find((status) => status.repo === options.modelRepo);
        if (
          options.generateDepth &&
          selectedStatus &&
          !selectedStatus.installed &&
          !busy &&
          !downloadingModel &&
          autoInstallModelKeyRef.current !== options.modelRepo
        ) {
          autoInstallModelKeyRef.current = options.modelRepo;
          window.localStorage.setItem(MODEL_PROMPT_SEEN_KEY, "1");
          pushLog(`Installing ${modelName(options.modelRepo)} for local generation.`);
          setDownloadingModel(options.modelRepo);
          setDownloadProgress({
            modelRepo: options.modelRepo,
            message: `Preparing ${modelName(options.modelRepo)}`,
            bytesDownloaded: 0,
            bytesTotal: selectedStatus.sizeBytes,
            percent: 0
          });
          setHeadline(`Preparing ${modelName(options.modelRepo)} download`);
          void invoke("download_model", { modelRepo: options.modelRepo }).catch((error) => {
            setDownloadingModel(null);
            setDownloadProgress(null);
            pushLog(`Model download failed to start: ${String(error)}`);
          });
          return;
        }

        const promptSeen = window.localStorage.getItem(MODEL_PROMPT_SEEN_KEY) === "1";
        const anyInstalled = statuses.some((status) => status.installed);
        if (!promptSeen && !anyInstalled) {
          setShowModelPrompt(true);
        }
      })
      .catch((error) => pushLog(`Could not read model status: ${String(error)}`));
  }, [
    busy,
    downloadingModel,
    options.generateDepth,
    options.modelRepo,
    pushLog,
    refreshModelStatus
  ]);

  useEffect(() => {
    if (!isTauriRuntime()) return;

    if (!options.keepModelLoaded || !options.generateDepth) {
      if (preloadTimerRef.current !== null) {
        window.clearTimeout(preloadTimerRef.current);
        preloadTimerRef.current = null;
      }
      preloadKeyRef.current = null;
      activePreloadKeyRef.current = null;
      failedPreloadKeyRef.current = null;
      setPreloadingModel(null);
      setLoadedModelKey(null);
      void invoke("stop_persistent_worker").catch((error) =>
        pushLog(`Could not stop cached model worker: ${String(error)}`)
      );
      return;
    }

    if (preloadTimerRef.current !== null) {
      window.clearTimeout(preloadTimerRef.current);
      preloadTimerRef.current = null;
    }

    if (!selectedModelStatus?.installed || busy || downloadingModel || preloadingModel) {
      return;
    }

    const preloadKey = `${options.modelRepo}:${options.device}`;
    if (preloadKeyRef.current === preloadKey || failedPreloadKeyRef.current === preloadKey) {
      return;
    }

    preloadKeyRef.current = preloadKey;
    const preloadOptions: DepthOptions = {
      ...defaultOptions,
      device: options.device,
      modelRepo: options.modelRepo,
      generateDepth: true,
      keepModelLoaded: true
    };
    preloadTimerRef.current = window.setTimeout(() => {
      preloadTimerRef.current = null;
      setPreloadingModel(preloadOptions.modelRepo);
      setLoadedModelKey(null);
      activePreloadKeyRef.current = preloadKey;
      setHeadline(`Loading ${modelName(preloadOptions.modelRepo)} into memory`);
      pushLog(`Loading ${modelName(preloadOptions.modelRepo)} into memory for faster batches.`);
      void invoke("preload_model", { options: preloadOptions }).catch((error) => {
        activePreloadKeyRef.current = null;
        failedPreloadKeyRef.current = preloadKey;
        setPreloadingModel(null);
        setLoadedModelKey(null);
        pushLog(`Model preload failed: ${String(error)}`);
      });
    }, 1200);

    return () => {
      if (preloadTimerRef.current !== null) {
        window.clearTimeout(preloadTimerRef.current);
        preloadTimerRef.current = null;
      }
    };
  }, [
    busy,
    downloadingModel,
    options.device,
    options.generateDepth,
    options.keepModelLoaded,
    options.modelRepo,
    preloadingModel,
    pushLog,
    selectedModelStatus?.installed
  ]);

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
      if (resolved.length > 0) {
        setSelectedPath((current) => current ?? resolved[0]);
      }
      const knownBeforeAdd = new Set(queueRef.current.map((item) => item.path));
      const addedCount = resolved.filter((path) => !knownBeforeAdd.has(path)).length;
      setHeadline(`${addedCount} new image${addedCount === 1 ? "" : "s"} ready`);
      pushLog(
        addedCount === resolved.length
          ? `Queued ${addedCount} image${addedCount === 1 ? "" : "s"}.`
          : `Queued ${addedCount} new image${addedCount === 1 ? "" : "s"}; ${resolved.length - addedCount} duplicate${resolved.length - addedCount === 1 ? "" : "s"} skipped.`
      );
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
      pushLog("Browser preview mode. Launch with Tauri to use local files and depth models.");
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

      if (payload.type === "model_ready" && payload.modelRepo) {
        setLoadedModelKey(`${payload.modelRepo}:${payload.device ?? "auto"}`);
        activePreloadKeyRef.current = null;
        failedPreloadKeyRef.current = null;
        setPreloadingModel(null);
      }

      if (payload.type === "download_progress" && payload.modelRepo) {
        const percent = typeof payload.percent === "number" ? payload.percent : null;
        setDownloadProgress({
          modelRepo: payload.modelRepo,
          message: payload.message ?? `Downloading ${modelName(payload.modelRepo)}`,
          bytesDownloaded: payload.bytesDownloaded ?? 0,
          bytesTotal: payload.bytesTotal ?? 0,
          percent
        });
        setHeadline(
          percent === null
            ? `Downloading ${modelName(payload.modelRepo)}`
            : `Downloading ${modelName(payload.modelRepo)} ${percent.toFixed(1)}%`
        );
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
        const outputPaths = parseOutputPaths(payload.output);
        if (outputPaths.length > 0) {
          setOutputPreviews((current) => {
            const next = { ...current };
            outputPaths.forEach((path) => {
              delete next[path];
            });
            return next;
          });
        }
        setQueue((current) =>
          current.map((item) =>
            item.path === payload.path
              ? (() => {
                  const previous = item.outputHistory ?? [];
                  const merged = [
                    ...outputPaths,
                    ...previous.filter((path) => !outputPaths.includes(path))
                  ];
                  return {
                    ...item,
                    status: "done",
                    outputHistory: merged,
                    message: outputPaths.length > 0 ? `${merged.length} output${merged.length === 1 ? "" : "s"}` : "Saved"
                  };
                })()
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

      if (payload.type === "canceled") {
        setQueue((current) =>
          current.map((item) =>
            item.status === "pending" || item.status === "running"
              ? { ...item, status: "canceled", message: "Canceled" }
              : item
          )
        );
      }

      if (payload.type === "done" || payload.type === "error" || payload.type === "canceled") {
        setBusy(false);
        setCanceling(false);
        setDownloadingModel(null);
        setDownloadProgress(null);
        setPreloadingModel(null);
        if (payload.type === "error" || payload.type === "canceled") {
          setLoadedModelKey(null);
          if (activePreloadKeyRef.current) {
            failedPreloadKeyRef.current = activePreloadKeyRef.current;
            activePreloadKeyRef.current = null;
          } else {
            preloadKeyRef.current = null;
          }
        }
        void refreshModelStatus().catch((error) =>
          pushLog(`Could not refresh model status: ${String(error)}`)
        );
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
    let aspectAdjusting = false;
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
        aspectAdjusting = true;
        try {
          await windowHandle.setSize(new PhysicalSize(width, height));
        } finally {
          window.setTimeout(() => {
            aspectAdjusting = false;
          }, 250);
        }
      }
    };
    const aspectPromise = getCurrentWindow().onResized(() => {
      if (aspectAdjusting) {
        return;
      }
      if (aspectTimer !== undefined) {
        window.clearTimeout(aspectTimer);
      }
      aspectTimer = window.setTimeout(() => {
        void enforceAspectRatio();
      }, 850);
    });

    return () => {
      if (aspectTimer !== undefined) {
        window.clearTimeout(aspectTimer);
      }
      void unlistenPromise.then((unlisten) => unlisten());
      void dragPromise.then((unlisten) => unlisten());
      void aspectPromise.then((unlisten) => unlisten());
    };
  }, [pushLog, refreshModelStatus, resolvePaths]);

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

  const chooseOutputFolder = async () => {
    if (!isTauriRuntime()) {
      pushLog("Folder picking requires the Tauri desktop shell.");
      return;
    }
    try {
      const selected = await open({ multiple: false, directory: true });
      if (typeof selected === "string") {
        setOptions((current) => ({ ...current, outputDir: selected }));
        pushLog(`Output folder set: ${selected}`);
      } else {
        pushLog("Output folder picker canceled.");
      }
    } catch (error) {
      pushLog(`Output folder picker failed: ${String(error)}`);
    }
  };

  const resetOutputFolder = () => {
    setOptions((current) => ({ ...current, outputDir: "" }));
    pushLog("Output folder reset to source folders.");
  };

  const openContainingFolder = async (path: string) => {
    if (!isTauriRuntime()) {
      pushLog("Opening folders requires the Tauri desktop shell.");
      return;
    }
    try {
      await invoke("open_containing_folder", { path });
    } catch (error) {
      pushLog(`Could not open folder: ${String(error)}`);
    }
  };

  const openModelFolder = async () => {
    if (!isTauriRuntime()) {
      pushLog("Opening the model folder requires the Tauri desktop shell.");
      return;
    }
    try {
      await invoke("open_model_folder");
      pushLog("Opened model install folder.");
    } catch (error) {
      pushLog(`Could not open model folder: ${String(error)}`);
    }
  };

  const openPackageReport = async () => {
    if (!isTauriRuntime()) {
      pushLog("Opening the package report requires the Tauri desktop shell.");
      return;
    }
    try {
      await invoke("open_package_report");
      pushLog("Opened package report.");
    } catch (error) {
      pushLog(`Could not open package report: ${String(error)}`);
    }
  };

  const openInputPreview = (path: string) => {
    if (!isPreviewableImage(path)) {
      pushLog("This input format cannot be previewed in the app.");
      return;
    }
    setSelectedPath(path);
    setImageModal({ path, title: fileName(path) });
  };

  const resetSettings = () => {
    setOptions(defaultOptions);
    pushLog("Settings reset to defaults.");
  };

  const resetDepthTuning = () => {
    setOptions((current) => ({
      ...current,
      processRes: defaultOptions.processRes,
      processResMethod: defaultOptions.processResMethod,
      normalizeLow: defaultOptions.normalizeLow,
      normalizeHigh: defaultOptions.normalizeHigh,
      depthGamma: defaultOptions.depthGamma
    }));
    pushLog(
      `Depth tuning reset: ${defaultOptions.processRes}px, long side, black ${defaultOptions.normalizeLow}, white ${defaultOptions.normalizeHigh}, gamma ${defaultOptions.depthGamma}.`
    );
  };

  const start = useCallback(async (trigger: "manual" | "auto" = "manual") => {
    if (!isTauriRuntime()) {
      pushLog("Generation requires the Tauri desktop shell.");
      return;
    }
    if (busy || downloadingModel || preloadingModel) return;
    const pendingPaths = queue
      .filter(
        (item) => item.status === "pending" || item.status === "error" || item.status === "canceled"
      )
      .map((item) => item.path);
    const rerunningCompletedQueue = pendingPaths.length === 0 && queue.length > 0;
    const runnablePaths = rerunningCompletedQueue ? queue.map((item) => item.path) : pendingPaths;
    if (runnablePaths.length === 0) return;
    if (!options.generateDepth && !options.convertWebp) {
      pushLog("Enable depth generation, WebP conversion, or both.");
      return;
    }
    if (trigger === "auto") {
      pushLog("Parameters changed. Regenerating queued images.");
    }
    if (rerunningCompletedQueue) {
      pushLog(
        options.overwrite
          ? "Rerunning queued images and replacing existing outputs."
          : "Rerunning queued images. Existing outputs will get numbered filenames."
      );
    }
    if (options.generateDepth && !modelStatuses[options.modelRepo]?.installed) {
      setDownloadProgress({
        modelRepo: options.modelRepo,
        message: `Preparing ${modelName(options.modelRepo)}`,
        bytesDownloaded: 0,
        bytesTotal: modelStatuses[options.modelRepo]?.sizeBytes ?? 0,
        percent: 0
      });
    }
    setBusy(true);
    setQueue((current) =>
      current.map((item) =>
        runnablePaths.includes(item.path)
          ? { ...item, status: "pending", message: undefined }
          : item
      )
    );
    try {
      await invoke("start_depth_job", { paths: runnablePaths, options });
    } catch (error) {
      setBusy(false);
      setCanceling(false);
      setDownloadProgress(null);
      pushLog(String(error));
      setHeadline("Unable to start worker");
    }
  }, [busy, downloadingModel, modelStatuses, options, preloadingModel, pushLog, queue]);

  useEffect(() => {
    if (previousAutoGenerateKeyRef.current === null) {
      previousAutoGenerateKeyRef.current = autoGenerateParamsKey;
      return;
    }

    const changed = previousAutoGenerateKeyRef.current !== autoGenerateParamsKey;
    previousAutoGenerateKeyRef.current = autoGenerateParamsKey;
    if (!changed || !options.autoGenerateOnChange) return;

    if (autoGenerateTimerRef.current !== null) {
      window.clearTimeout(autoGenerateTimerRef.current);
      autoGenerateTimerRef.current = null;
    }

    if (
      queue.length === 0 ||
      busy ||
      downloadingModel ||
      preloadingModel ||
      (!options.generateDepth && !options.convertWebp)
    ) {
      return;
    }

    autoGenerateTimerRef.current = window.setTimeout(() => {
      autoGenerateTimerRef.current = null;
      void start("auto");
    }, 650);

    return () => {
      if (autoGenerateTimerRef.current !== null) {
        window.clearTimeout(autoGenerateTimerRef.current);
        autoGenerateTimerRef.current = null;
      }
    };
  }, [
    autoGenerateParamsKey,
    busy,
    downloadingModel,
    options.autoGenerateOnChange,
    options.convertWebp,
    options.generateDepth,
    preloadingModel,
    queue.length,
    start
  ]);

  const cancelActiveJob = async () => {
    if (!isTauriRuntime() || (!busy && !downloadingModel && !preloadingModel) || canceling) return;
    setCanceling(true);
    setHeadline("Canceling job");
    pushLog("Cancel requested.");
    try {
      await invoke("cancel_active_job");
    } catch (error) {
      setCanceling(false);
      pushLog(`Cancel failed: ${String(error)}`);
    }
  };

  const reset = () => {
    if (autoGenerateTimerRef.current !== null) {
      window.clearTimeout(autoGenerateTimerRef.current);
      autoGenerateTimerRef.current = null;
    }
    setQueue([]);
    setOutputPreviews({});
    setLog([]);
    setSelectedPath(null);
    setHeadline("Drop images or folders");
  };

  return (
    <div className="scale-stage">
      <main className="app-shell">
        <aside className="sidebar">
          <div className="brand">
            <div>
              <h1>DepthMap AI</h1>
              <p>MDA local depth maps</p>
            </div>
          </div>

          <section className="panel">
          <div className="panel-title">
            <Settings2 size={16} />
            Settings
            <button className="secondary mini-button" onClick={resetSettings} title="Reset all settings">
              <RotateCcw size={14} />
              Reset
            </button>
          </div>

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

          <div className="field">
            <span>Output folder</span>
            <div className="output-folder-row">
              <input value={compactPath(options.outputDir)} title={options.outputDir || "Same as source"} readOnly />
              <button className="secondary mini-button" onClick={chooseOutputFolder} title="Choose output folder">
                <FolderOpen size={14} />
              </button>
              <button className="secondary mini-button" onClick={resetOutputFolder} title="Use source folders">
                Same
              </button>
            </div>
          </div>

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

          <div className="tuning-header">
            <span>Depth tuning</span>
            <button className="secondary mini-button" onClick={resetDepthTuning} title="Reset depth tuning">
              <RotateCcw size={14} />
              Reset
            </button>
          </div>

          <div className="split-fields">
            <label className="field">
              <span>Depth resolution</span>
              <input
                type="range"
                min="336"
                max="1008"
                step="14"
                disabled={!options.generateDepth}
                value={options.processRes}
                onChange={(event) =>
                  setOptions((current) => ({
                    ...current,
                    processRes: Number(event.target.value) || current.processRes
                  }))
                }
              />
            </label>

            <label className="field">
              <span>{options.processRes}px</span>
              <select
                value={options.processResMethod}
                disabled={!options.generateDepth}
                onChange={(event) =>
                  setOptions((current) => ({
                    ...current,
                    processResMethod: event.target.value as ProcessResMethod
                  }))
                }
              >
                <option value="upper_bound_resize">Long</option>
                <option value="lower_bound_resize">Short</option>
                <option value="upper_bound_crop">Crop L</option>
                <option value="lower_bound_crop">Crop S</option>
              </select>
            </label>
          </div>

          <div className="range-field">
            <div className="range-labels">
              <span>Depth window</span>
              <strong>
                {options.normalizeLow.toFixed(1)} - {options.normalizeHigh.toFixed(1)}
              </strong>
            </div>
            <div
              className="dual-range"
              style={
                {
                  "--range-low": `${options.normalizeLow}%`,
                  "--range-high": `${options.normalizeHigh}%`
                } as React.CSSProperties
              }
            >
              <input
                type="range"
                min="0"
                max="99"
                step="0.5"
                disabled={!options.generateDepth}
                value={options.normalizeLow}
                onChange={(event) =>
                  setOptions((current) => ({
                    ...current,
                    normalizeLow: Math.min(Number(event.target.value), current.normalizeHigh - 0.5)
                  }))
                }
              />
              <input
                type="range"
                min="1"
                max="100"
                step="0.5"
                disabled={!options.generateDepth}
                value={options.normalizeHigh}
                onChange={(event) =>
                  setOptions((current) => ({
                    ...current,
                    normalizeHigh: Math.max(Number(event.target.value), current.normalizeLow + 0.5)
                  }))
                }
              />
            </div>
          </div>

          <div className="single-field">
            <label className="field">
              <span>Gamma</span>
              <input
                type="range"
                min="0.2"
                max="4"
                step="0.1"
                disabled={!options.generateDepth}
                value={options.depthGamma}
                onChange={(event) =>
                  setOptions((current) => ({
                    ...current,
                    depthGamma: Number(event.target.value) || current.depthGamma
                  }))
                }
              />
              <strong>{options.depthGamma.toFixed(1)}</strong>
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

          {options.convertWebp ? (
            <label className="field compact-field">
              <span>WebP quality</span>
              <input
                type="number"
                min="1"
                max="100"
                value={options.webpQuality}
                onChange={(event) =>
                  setOptions((current) => ({
                    ...current,
                    webpQuality: Number(event.target.value) || current.webpQuality
                  }))
                }
              />
            </label>
          ) : null}

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

          <label className="toggle">
            <input
              type="checkbox"
              checked={options.autoGenerateOnChange}
              onChange={(event) =>
                setOptions((current) => ({ ...current, autoGenerateOnChange: event.target.checked }))
              }
            />
            <span>Auto generate on changes</span>
          </label>

          <label className="toggle">
            <input
              type="checkbox"
              checked={agentControlEnabled}
              onChange={(event) => void toggleAgentControl(event.target.checked)}
            />
            <span>Agent API</span>
          </label>
          <label className="field agent-port-field">
            <span>API port</span>
            <input
              type="number"
              min="1"
              max="65535"
              value={agentPort}
              onChange={(event) => setAgentPort(event.target.value)}
              onBlur={() => void applyAgentPort()}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.currentTarget.blur();
                }
              }}
            />
          </label>
          {agentStatus?.enabled ? (
            <div className="agent-status">
              <span>{agentStatus.url}</span>
              <small>OpenAPI: {agentStatus.openapiUrl}</small>
            </div>
          ) : null}
          </section>

          <div className="app-version">{APP_VERSION}</div>
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
              disabled={
                busy ||
                Boolean(downloadingModel) ||
                Boolean(preloadingModel) ||
                queue.length === 0 ||
                (!options.generateDepth && !options.convertWebp)
              }
              onClick={() => void start()}
            >
              {busy ? <Loader2 className="spin" size={17} /> : <Play size={17} />}
              Generate
            </button>
            <button
              className="secondary"
              disabled={(!busy && !downloadingModel && !preloadingModel) || canceling}
              onClick={cancelActiveJob}
            >
              {canceling ? <Loader2 className="spin" size={17} /> : <Square size={17} />}
              Cancel
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
          <div className="gpu-stat">
            <div className="gpu-stat-head">
              <span className="vram-value">{vramDisplay.value}</span>
              <select
                value={options.device}
                onChange={(event) =>
                  setOptions((current) => ({
                    ...current,
                    device: event.target.value as Device
                  }))
                }
                title="GPU for model loading and generation"
              >
                {gpuOptions.map((option) => (
                  <option value={option.value} key={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <small>{vramDisplay.label}</small>
          </div>
        </div>

        {activeDownload ? (
          <section className="download-status">
            <div>
              <strong>{activeDownload.message}</strong>
              <small>
                {formatBytes(activeDownload.bytesDownloaded)} / {formatBytes(activeDownload.bytesTotal)}
              </small>
            </div>
            <div className="progress-shell" aria-label="Model download progress">
              <div
                className="progress-fill"
                style={{ width: `${Math.max(0, Math.min(100, activeDownload.percent ?? 0))}%` }}
              />
            </div>
            <span>{activeDownload.percent === null ? "--" : `${activeDownload.percent.toFixed(1)}%`}</span>
          </section>
        ) : null}

        {activeModelLoad ? (
          <section className="model-load-status">
            <div>
              <strong>Loading {modelName(activeModelLoad)} into memory</strong>
              <small>Keeping the model warm for faster repeat batches</small>
            </div>
            <code aria-label="Model loading indicator">{loadingNoise}</code>
          </section>
        ) : null}

        <div className="media-grid">
          <section className={`drop-zone ${dragging ? "dragging" : ""}`}>
            <Download size={42} />
            <div>
              <h3>Drop images or folders here</h3>
              <p>
                {options.outputDir
                  ? "Depth maps and WebP files are written to the selected output folder."
                  : "Depth maps and WebP files are written beside each source image."}
              </p>
            </div>
          </section>

          <section className="preview-panel">
            <div className="preview-head">
              <strong>Preview</strong>
              <small>{selectedItem ? fileName(selectedItem.path) : "No image selected"}</small>
            </div>
            {selectedItem ? (
              <div className="preview-images">
                <div>
                  <span>Input</span>
                  {previewImages.input ? (
                    <img src={previewImages.input} alt="" />
                  ) : (
                    <div className="preview-empty">Loading</div>
                  )}
                </div>
                <div>
                  <span>Output</span>
                  {previewImages.output ? (
                    <img src={previewImages.output} alt="" />
                  ) : (
                    <div className="preview-empty">Pending</div>
                  )}
                </div>
              </div>
            ) : (
              <div className="preview-empty">Select or queue an image.</div>
            )}
            {previewImages.error ? <small className="preview-error">{previewImages.error}</small> : null}
          </section>
        </div>

        <section className="runtime workspace-runtime">
          <div className="runtime-heading">
            <div className="panel-title">
              <Cpu size={16} />
              Runtime
            </div>
            {runtime?.pythonFound ? (
              <p className="ok">Python: {runtime.pythonCommand}</p>
            ) : (
              <p className="warn">Python 3.10, 3.11, or 3.12 not found on PATH.</p>
            )}
          </div>
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
              <option value="sy000/MDA/mda_mog_sky_l2">MDA-DA3-MoG-Sky</option>
              <option value="depth-anything/DA3-LARGE-1.1">DA3-LARGE-1.1</option>
              <option value="depth-anything/DA3-SMALL">DA3-SMALL</option>
            </select>
          </label>
          <label className="toggle keep-loaded-toggle">
            <input
              type="checkbox"
              checked={options.keepModelLoaded}
              disabled={!options.generateDepth}
              onChange={(event) =>
                setOptions((current) => ({ ...current, keepModelLoaded: event.target.checked }))
              }
            />
            <span>Keep model loaded</span>
          </label>
          <div className="runtime-summary">
            <p className="muted">
              {!options.generateDepth
                ? "Model downloads only when depth generation is enabled."
                : currentModelReady
                  ? `${modelName(options.modelRepo)} is loaded in memory.`
                  : preloadingModel === options.modelRepo
                    ? `${modelName(options.modelRepo)} is loading into memory.`
                    : selectedModelStatus?.installed
                  ? `${modelName(options.modelRepo)} is installed.`
                  : `${modelName(options.modelRepo)} downloads when needed.`}
            </p>
            <div className="runtime-actions">
              <button className="secondary folder-button" onClick={openModelFolder}>
                <FolderOpen size={16} />
                Models
              </button>
              <button className="secondary folder-button" onClick={openPackageReport}>
                <Package size={16} />
                Packages
              </button>
            </div>
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
              <div
                className={`queue-row ${item.status} ${selectedItem?.path === item.path ? "selected" : ""}`}
                key={item.path}
                onClick={() => setSelectedPath(item.path)}
                role="button"
                tabIndex={0}
              >
                <div className="queue-input">
                  <button
                    className="input-preview-button"
                    onClick={(event) => {
                      event.stopPropagation();
                      openInputPreview(item.path);
                    }}
                    title="Open larger input preview"
                  >
                    <strong>{fileName(item.path)}</strong>
                    <small>{item.path}</small>
                  </button>
                </div>
                <div className="status">
                  {statusIcon(item.status)}
                  {item.message ?? item.status}
                </div>
                <div className="output-history">
                  {(item.outputHistory?.length ?? 0) > 0 ? (
                    <>
                      <div className="output-thumbs">
                        {item.outputHistory?.map((path) => (
                          <button
                            className={`output-thumb ${isPreviewableImage(path) ? "" : "file-output"}`}
                            key={path}
                            onClick={(event) => {
                              event.stopPropagation();
                              setSelectedPath(item.path);
                            }}
                            title={path}
                          >
                            {isPreviewableImage(path) && outputPreviews[path] ? (
                              <img src={outputPreviews[path]} alt="" />
                            ) : (
                              <span>{fileName(path).split(".").pop()?.toUpperCase() ?? "FILE"}</span>
                            )}
                          </button>
                        ))}
                      </div>
                      <button
                        className="secondary icon output-folder"
                        onClick={(event) => {
                          event.stopPropagation();
                          void openContainingFolder(item.outputHistory?.[0] ?? item.path);
                        }}
                        title="Open output folder"
                      >
                        <FolderOpen size={15} />
                      </button>
                    </>
                  ) : (
                    <span className="output-empty">No outputs yet</span>
                  )}
                </div>
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
      {showModelPrompt ? (
        <div className="modal-backdrop" role="presentation">
          <section className="model-modal" role="dialog" aria-modal="true" aria-labelledby="model-dialog-title">
            <div className="modal-heading">
              <div>
                <p className="eyebrow">First launch</p>
                <h3 id="model-dialog-title">Select a model to download</h3>
              </div>
              <button className="secondary" onClick={markModelPromptSeen}>
                Later
              </button>
            </div>
            <div className="model-choices">
              {(Object.entries(MODEL_DETAILS) as Array<[ModelRepo, (typeof MODEL_DETAILS)[ModelRepo]]>).map(
                ([repo, details]) => (
                  <button
                    className="model-choice"
                    key={repo}
                    disabled={Boolean(downloadingModel)}
                    onClick={() => void downloadSelectedModel(repo, true)}
                  >
                    <span>
                      <strong>{details.name}</strong>
                      <small>{details.note}</small>
                    </span>
                    <span>
                      <strong>{details.size}</strong>
                      <small>{details.vram}</small>
                    </span>
                  </button>
                )
              )}
            </div>
          </section>
        </div>
      ) : null}
      {imageModal ? (
        <div
          className="modal-backdrop image-modal-backdrop"
          role="presentation"
          onClick={() => setImageModal(null)}
        >
          <section
            className="image-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="input-preview-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-heading image-modal-heading">
              <div>
                <p className="eyebrow">Input preview</p>
                <h3 id="input-preview-title">{imageModal.title}</h3>
              </div>
              <button className="secondary icon" onClick={() => setImageModal(null)} title="Close preview">
                <X size={18} />
              </button>
            </div>
            <div className="image-modal-frame">
              {imageModal.dataUrl ? (
                <img src={imageModal.dataUrl} alt="" />
              ) : imageModal.error ? (
                <div className="preview-empty">{imageModal.error}</div>
              ) : (
                <div className="preview-empty">Loading</div>
              )}
            </div>
            <small className="image-modal-path">{imageModal.path}</small>
          </section>
        </div>
      ) : null}
    </div>
  );
}

export default App;
