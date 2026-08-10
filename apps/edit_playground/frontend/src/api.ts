import type {
  AudioLibraryApplyRequest,
  AudioLibraryKind,
  AudioLibraryPage,
  CompiledEdit,
  Destination,
  EditOperation,
  EditGenerationSettings,
  GenerationEvent,
  GenerationSettings,
  HealthStatus,
  NoiseLibraryKind,
  NoiseLibraryPage,
  RecognitionResult,
  SessionSnapshot,
} from "./types";
import { getSharedGradioClient } from "./gradioTransport";

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail || response.statusText);
  }
  return response.json() as Promise<T>;
}

export const MAX_UPLOAD_BYTES = 8_192 * 1_024;
export const MAX_AUDIO_SECONDS = 30;

function assertUploadSize(file: File) {
  if (file.size > MAX_UPLOAD_BYTES) {
    throw new Error("Audio file must not exceed 8192 KiB.");
  }
}

function errorDetail(error: unknown) {
  return error instanceof Error && error.message ? error.message : String(error);
}

function transportError(action: "Generation" | "Recognition", stage: "connection" | "queue", error: unknown) {
  const detail = errorDetail(error);
  if (stage === "connection") {
    return new Error(`${action} could not connect to the local inference queue. ${detail}`);
  }
  return new Error(`${action} could not enter or read from the local inference queue. ${detail}`);
}

export const studioApi = {
  health() {
    return fetch("/api/healthz").then(json<HealthStatus>);
  },
  snapshot(sessionId: string) {
    return fetch(`/api/session/${encodeURIComponent(sessionId)}`).then(json<SessionSnapshot>);
  },
  selectModel(sessionId: string, modelId: string) {
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/model`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    }).then(json<SessionSnapshot>);
  },

  noiseLibrary(sessionId: string, kind: NoiseLibraryKind, page = 1) {
    const query = new URLSearchParams({ page: String(page), page_size: "10" });
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/noise-library/${kind}?${query}`).then(json<NoiseLibraryPage>);
  },

  createNoiseUpload(sessionId: string, file: File, historyConsent = false) {
    assertUploadSize(file);
    const form = new FormData();
    form.append("audio", file);
    form.append("history_consent", String(historyConsent));
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/noise-library/uploads`, {
      method: "POST",
      body: form,
    }).then(json<NoiseLibraryPage>);
  },

  setSourceNoise(sessionId: string, kind: NoiseLibraryKind, itemId: string, snrDb: number, expectedVersion: string) {
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/edit-source/noise`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind, item_id: itemId, snr_db: snrDb, expected_version: expectedVersion }),
    }).then(json<SessionSnapshot>);
  },

  clearSourceNoise(sessionId: string, expectedVersion: string) {
    const query = new URLSearchParams({ expected_version: expectedVersion });
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/edit-source/noise?${query}`, {
      method: "DELETE",
    }).then(json<SessionSnapshot>);
  },

  audioLibrary(sessionId: string, kind: AudioLibraryKind, page = 1) {
    const query = new URLSearchParams({ page: String(page), page_size: "10" });
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/audio-library/${kind}?${query}`).then(json<AudioLibraryPage>);
  },

  createLibraryUpload(sessionId: string, file: File, transcript: string, multipleSpeakers: boolean, historyConsent = false) {
    assertUploadSize(file);
    const form = new FormData();
    form.append("audio", file);
    form.append("transcript", transcript);
    form.append("multiple_speakers", String(multipleSpeakers));
    form.append("history_consent", String(historyConsent));
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/audio-library/uploads`, {
      method: "POST",
      body: form,
    }).then(json<AudioLibraryPage>);
  },

  applyAudioLibrary(sessionId: string, request: AudioLibraryApplyRequest) {
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/audio-library/apply`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    }).then(json<SessionSnapshot>);
  },

  upload(sessionId: string, destination: Destination, file: File, transcript: string, multipleSpeakers: boolean, historyConsent = false) {
    assertUploadSize(file);
    const form = new FormData();
    form.append("audio", file);
    form.append("transcript", transcript);
    form.append("multiple_speakers", String(multipleSpeakers));
    form.append("history_consent", String(historyConsent));
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/upload/${destination}`, {
      method: "POST",
      body: form,
    }).then(json<SessionSnapshot>);
  },

  insertSegment(sessionId: string, sourceId: string, index: number, expectedVersion: string) {
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/edit-source/segments`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_id: sourceId, index, expected_version: expectedVersion }),
    }).then(json<SessionSnapshot>);
  },

  uploadSegment(sessionId: string, file: File, transcript: string, multipleSpeakers: boolean, historyConsent: boolean, expectedVersion: string, options: { index?: number; replaceSegmentId?: string }) {
    assertUploadSize(file);
    const form = new FormData();
    form.append("audio", file);
    form.append("transcript", transcript);
    form.append("multiple_speakers", String(multipleSpeakers));
    form.append("history_consent", String(historyConsent));
    form.append("expected_version", expectedVersion);
    if (options.index !== undefined) form.append("index", String(options.index));
    if (options.replaceSegmentId) form.append("replace_segment_id", options.replaceSegmentId);
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/edit-source/segments/upload`, { method: "POST", body: form }).then(json<SessionSnapshot>);
  },

  mutateSegment(sessionId: string, segmentId: string, payload: Record<string, unknown>) {
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/edit-source/segments/${encodeURIComponent(segmentId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(json<SessionSnapshot>);
  },

  deleteSegment(sessionId: string, segmentId: string, expectedVersion: string) {
    const query = new URLSearchParams({ expected_version: expectedVersion });
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/edit-source/segments/${encodeURIComponent(segmentId)}?${query}`, { method: "DELETE" }).then(json<SessionSnapshot>);
  },

  updateSegmentTranscript(sessionId: string, segmentId: string, transcript: string, expectedVersion: string) {
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/edit-source/segments/${encodeURIComponent(segmentId)}/transcript`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript, expected_version: expectedVersion }),
    }).then(json<SessionSnapshot>);
  },

  route(sessionId: string, source: string, destination: Destination) {
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/route`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, destination }),
    }).then(json<SessionSnapshot>);
  },

  preset(sessionId: string, name: string, destination: Destination) {
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/preset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, destination }),
    }).then(json<SessionSnapshot>);
  },

  clearReference(sessionId: string, destination: Destination) {
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/reference/${destination}`, {
      method: "DELETE",
    }).then(json<SessionSnapshot>);
  },

  updateText(sessionId: string, destination: Destination, text: string) {
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/text/${destination}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }).then(json<SessionSnapshot>);
  },

  compile(sessionId: string, expectedVersion: string, operations: EditOperation[], enhance = false) {
    return fetch(`/api/session/${encodeURIComponent(sessionId)}/compile-edit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expected_version: expectedVersion, operations, enhance }),
    }).then(json<CompiledEdit>);
  },

  async generate(
    endpoint: "synthesize_tts" | "synthesize_edit",
    args: unknown[],
    onEvent: (event: GenerationEvent) => void,
  ): Promise<void> {
    let client;
    try {
      client = await getSharedGradioClient();
    } catch (error) {
      throw transportError("Generation", "connection", error);
    }
    let job;
    try {
      job = client.submit(`/${endpoint}`, args, undefined, null, true);
    } catch (error) {
      throw transportError("Generation", "queue", error);
    }
    let failed = false;
    try {
      for await (const result of job) {
        if (result.type === "status") {
          if (result.stage === "pending") {
            const position = typeof result.position === "number" ? `Queue ${result.position + 1}` : undefined;
            onEvent({ phase: "queued", progress: 0, message: position });
          } else if (result.stage === "error") {
            const message = typeof result.message === "string" ? result.message : "Generation inference failed.";
            onEvent({ phase: "error", progress: null, message });
            failed = true;
          }
          continue;
        }
        if (result.type !== "data") continue;
        const value = Array.isArray(result.data) ? result.data[0] : result.data;
        if (typeof value === "string") {
          const event = JSON.parse(value) as GenerationEvent;
          if (event.phase === "error") failed = true;
          onEvent(event);
        }
      }
    } catch (error) {
      throw transportError("Generation", "queue", error);
    }
    if (failed) return;
  },

  async recognize(sessionId: string, file: File): Promise<RecognitionResult> {
    assertUploadSize(file);
    const form = new FormData();
    form.append("audio", file);
    const prepared = await fetch(
      `/api/session/${encodeURIComponent(sessionId)}/recognition/prepare`,
      { method: "POST", body: form },
    ).then(json<{ job_id: string }>);
    let primaryError: unknown;
    try {
      let client;
      try {
        client = await getSharedGradioClient();
      } catch (error) {
        throw transportError("Recognition", "connection", error);
      }
      let job;
      try {
        job = client.submit(
          "/recognize",
          [prepared.job_id],
          undefined,
          null,
          true,
        );
      } catch (error) {
        throw transportError("Recognition", "queue", error);
      }
      let result: RecognitionResult | null = null;
      try {
        for await (const event of job) {
          if (event.type === "status" && event.stage === "error") {
            throw new Error(
              `Recognition inference failed. ${typeof event.message === "string" ? event.message : "The ASR service returned an error."}`,
            );
          }
          if (event.type !== "data") continue;
          const value = Array.isArray(event.data) ? event.data[0] : event.data;
          if (typeof value === "string") {
            result = JSON.parse(value) as RecognitionResult;
          }
        }
      } catch (error) {
        if (error instanceof Error && error.message.startsWith("Recognition inference failed.")) throw error;
        throw transportError("Recognition", "queue", error);
      }
      if (!result?.text.trim()) throw new Error("Recognition inference failed. No speech was recognized.");
      return result;
    } catch (error) {
      primaryError = error;
      throw error;
    } finally {
      try {
        const response = await fetch(
          `/api/session/${encodeURIComponent(sessionId)}/recognition/${encodeURIComponent(prepared.job_id)}`,
          { method: "DELETE" },
        );
        if (!response.ok) {
          throw new Error(`HTTP ${response.status} ${response.statusText}`.trim());
        }
      } catch (error) {
        if (primaryError !== undefined) {
          console.warn("Recognition cleanup failed after an earlier error.", error);
        } else {
          throw new Error(`Recognition completed, but temporary upload cleanup failed. ${errorDetail(error)}`);
        }
      }
    }
  },

  tts(
    sessionId: string,
    text: string,
    settings: GenerationSettings & { use_xvector: boolean; model_id: string },
    onEvent: (event: GenerationEvent) => void,
  ) {
    return this.generate("synthesize_tts", [sessionId, text, JSON.stringify(settings)], onEvent);
  },

  edit(
    sessionId: string,
    expectedVersion: string,
    operations: EditOperation[],
    settings: EditGenerationSettings & { model_id: string; enhance: boolean },
    onEvent: (event: GenerationEvent) => void,
  ) {
    return this.generate(
      "synthesize_edit",
      [sessionId, JSON.stringify(operations), expectedVersion, JSON.stringify(settings)],
      onEvent,
    );
  },
};
