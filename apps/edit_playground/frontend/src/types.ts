export type Destination = "prompt" | "edit_source";
export type AudioLibraryKind = "presets" | "uploads" | "revisions";

export interface AudioLibraryItem {
  id: string;
  kind: AudioLibraryKind;
  title: string;
  audio_url: string;
  transcript: string;
  duration_seconds: number;
  multiple_speakers: boolean;
  created_at: string | null;
}

export interface AudioLibraryPage {
  items: AudioLibraryItem[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  selected_id?: string;
}

export interface AudioLibraryApplyRequest {
  kind: AudioLibraryKind;
  item_id: string;
  destination: Destination;
  action: "bind" | "insert" | "replace";
  expected_version?: string;
  index?: number;
  replace_segment_id?: string;
}

export interface AudioSegment {
  id: string;
  source_id: string;
  audio_url: string;
  transcript: string;
  duration_seconds: number;
  revision_id: string | null;
  multiple_speakers: boolean;
  allow_xvector: boolean;
}

export type NoiseLibraryKind = "presets" | "uploads";

export interface NoiseLibraryItem {
  id: string;
  kind: NoiseLibraryKind;
  title: string;
  audio_url: string;
  duration_seconds: number;
  created_at: string | null;
}

export interface NoiseLibraryPage {
  items: NoiseLibraryItem[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  selected_id?: string;
}

export interface ModelOption {
  id: string;
  label: string;
  path: string;
  status: "pending" | "loading" | "warming" | "ready" | "error";
  error: string | null;
  runtime_loaded: boolean;
}

export interface AudioReference {
  source_id: string;
  audio_url: string;
  text: string;
  revision_id: string | null;
  duration_seconds: number;
  composition_version: string;
  use_xvector: boolean;
  noise?: {
    item_id: string;
    kind: NoiseLibraryKind;
    snr_db: number;
  } | null;
  segments: AudioSegment[];
  origin?: { kind: "preset" | "revision" | "custom"; name: string | null };
}

export interface Revision {
  id: string;
  audio_url: string;
  text: string;
  parent_id: string | null;
  created_at: string;
  metadata: {
    kind?: "tts" | "edit";
    duration_seconds?: number;
    sample_rate?: number;
    [key: string]: unknown;
  };
}

export interface SessionSnapshot {
  session_id: string;
  selected_model_id?: string;
  models?: ModelOption[];
  prompt: AudioReference | null;
  edit_source: AudioReference | null;
  revisions: Revision[];
  latest: Revision | null;
  presets: Array<{ name: string; text: string }>;
  prompt_presets?: Array<{ name: string; text: string }>;
  edit_source_presets?: Array<{ name: string; text: string }>;
  defaults: { target_text: string; edit_operations: EditOperation[]; enhance: boolean };
  ode_methods: string[];
}

export interface HealthStatus {
  frontend: "building" | "ready" | "error";
  model: "warming" | "ready" | "error";
  frontend_error?: string | null;
  model_error?: string | null;
  optimize: boolean;
  precision: string;
  runtime_loaded: boolean;
  max_generate_length: number;
  warmup: "running" | "complete" | "skipped" | "error";
  ready: boolean;
  default_model_id: string;
  runtime_loaded_count: number;
  models: ModelOption[];
  uploads_enabled: boolean;
  recognition_available?: boolean;
  demo_url?: string;
  paper_url?: string;
}

export interface RecognitionResult {
  text: string;
  language: string;
}

export type OperationKind =
  | "replace"
  | "insert"
  | "delete"
  | "emotion"
  | "pitch"
  | "rate"
  | "pause";

export interface EditOperation {
  id: string;
  segment_id: string;
  kind: OperationKind;
  start: number;
  end: number;
  params: Record<string, string | number>;
}

export interface GenerationSettings {
  ode_method: string;
  num_steps: number;
  guidance_scale: number;
  speaker_scale: number;
  use_xvector: boolean;
  seed: number;
}

export type EditXVectorMode = boolean | "auto";

export interface EditGenerationSettings extends Omit<GenerationSettings, "use_xvector"> {
  use_xvector: EditXVectorMode;
}

export interface GenerationEvent {
  phase: "idle" | "queued" | "preparing" | "inference" | "saving" | "complete" | "error";
  progress: number | null;
  message?: string;
  snapshot?: SessionSnapshot;
}

export interface CompiledEdit {
  target_text: string;
  instruction: string;
}
