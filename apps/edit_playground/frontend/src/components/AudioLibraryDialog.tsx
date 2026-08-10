import { useEffect, useRef, useState } from "react";
import { Check, ChevronLeft, ChevronRight, Upload, Users, X } from "lucide-react";
import { studioApi } from "../api";
import type {
  AudioLibraryApplyRequest,
  AudioLibraryItem,
  AudioLibraryKind,
  AudioLibraryPage,
  SessionSnapshot,
} from "../types";
import { AudioPlayer, AudioVisualizer, type VisualizationMode, VisualizationToggle } from "./AudioCard";
import { AudioUploadDialog } from "./AudioUploadDialog";

const TABS: Array<{ id: AudioLibraryKind; label: string }> = [
  { id: "presets", label: "Presets" },
  { id: "uploads", label: "Uploads" },
  { id: "revisions", label: "Revisions" },
];

export interface AudioLibraryIntent extends Omit<AudioLibraryApplyRequest, "kind" | "item_id"> {
  title: string;
  initialFile?: File;
  currentDuration?: number;
  currentSegmentCount?: number;
  replacedDuration?: number;
}

function itemTime(value: string | null) {
  return value ? new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
}

function duration(value: number) {
  const seconds = Number.isFinite(value) ? Math.max(0, value) : 0;
  return `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
}

export function AudioLibraryDialog({ sessionId, intent, onCancel, onApplied, uploadsEnabled = true, recognitionAvailable = false }: {
  sessionId: string;
  intent: AudioLibraryIntent;
  onCancel: () => void;
  onApplied: (snapshot: SessionSnapshot, item: AudioLibraryItem) => void;
  uploadsEnabled?: boolean;
  recognitionAvailable?: boolean;
}) {
  const [tab, setTab] = useState<AudioLibraryKind>("presets");
  const [pageNumber, setPageNumber] = useState(1);
  const [page, setPage] = useState<AudioLibraryPage | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(uploadsEnabled ? intent.initialFile || null : null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [previewMode, setPreviewMode] = useState<VisualizationMode>("waveform");
  const [previewAvailable, setPreviewAvailable] = useState(true);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    studioApi.audioLibrary(sessionId, tab, pageNumber).then((value) => {
      if (active) setPage(value);
    }).catch((reason: unknown) => {
      if (active) setError(reason instanceof Error ? reason.message : String(reason));
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [pageNumber, sessionId, tab]);

  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !pendingFile && !submitting) onCancel();
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onCancel, pendingFile, submitting]);

  const selected = page?.items.find((item) => item.id === selectedId) || null;
  useEffect(() => setPreviewAvailable(true), [selected?.id]);
  const disabledReason = (item: AudioLibraryItem) => {
    if (intent.destination !== "edit_source") return "";
    if (intent.action === "bind") return item.duration_seconds > 30 ? "Source Audio exceeds 30 seconds." : "";
    if (intent.action === "insert") {
      if ((intent.currentSegmentCount || 0) >= 3) return "Source Audio already has 3 segments.";
      return (intent.currentDuration || 0) + item.duration_seconds > 30 ? "Combined Source Audio exceeds 30 seconds." : "";
    }
    const total = (intent.currentDuration || 0) - (intent.replacedDuration || 0) + item.duration_seconds;
    return total > 30 ? "Combined Source Audio exceeds 30 seconds." : "";
  };

  const changeTab = (next: AudioLibraryKind) => {
    setTab(next);
    setPageNumber(1);
    setSelectedId(null);
  };
  const changePage = (next: number) => {
    setPageNumber(next);
    setSelectedId(null);
  };
  const saveUpload = async (transcript: string, multipleSpeakers: boolean, historyConsent: boolean) => {
    if (!pendingFile) return;
    setError("");
    try {
      const next = await studioApi.createLibraryUpload(sessionId, pendingFile, transcript, multipleSpeakers, historyConsent);
      setTab("uploads");
      setPageNumber(1);
      setPage(next);
      setSelectedId(next.selected_id || next.items[0]?.id || null);
      setPendingFile(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  };
  const apply = async () => {
    if (!selected || disabledReason(selected)) return;
    setSubmitting(true);
    setError("");
    try {
      const snapshot = await studioApi.applyAudioLibrary(sessionId, {
        kind: selected.kind,
        item_id: selected.id,
        destination: intent.destination,
        action: intent.action,
        expected_version: intent.expected_version,
        index: intent.index,
        replace_segment_id: intent.replace_segment_id,
      });
      onApplied(snapshot, selected);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && !pendingFile && onCancel()}>
      <section className="audio-library" role="dialog" aria-modal="true" aria-label={intent.title}>
        <header className="audio-library__header"><h2>{intent.title}</h2><button className="icon-button" aria-label="Close audio library" onClick={onCancel}><X size={17} /></button></header>
        <div className="audio-library__tabs" role="tablist">{TABS.filter((item) => uploadsEnabled || item.id !== "uploads").map((item) => <button key={item.id} role="tab" aria-selected={tab === item.id} className={tab === item.id ? "active" : ""} onClick={() => changeTab(item.id)}>{item.label}</button>)}</div>
        {uploadsEnabled && tab === "uploads" && <div className="audio-library__upload"><input ref={fileInput} hidden type="file" accept="audio/*" onChange={(event) => { const file = event.currentTarget.files?.[0]; event.currentTarget.value = ""; if (file) setPendingFile(file); }} /><button className="secondary-button" onClick={() => fileInput.current?.click()}><Upload size={15} /> Upload Audio</button></div>}
        <div className="audio-library__body">
          <div className="audio-library__list" role="radiogroup" aria-label={`${TABS.find((item) => item.id === tab)?.label} audio`}>
            {loading && <div className="library-empty">Loading…</div>}
            {!loading && !page?.items.length && <div className="library-empty">No audio available</div>}
            {!loading && page?.items.map((item) => {
              const reason = disabledReason(item);
              return <label key={item.id} className={`library-item ${selectedId === item.id ? "library-item--selected" : ""} ${reason ? "library-item--disabled" : ""}`} title={reason || item.transcript}>
                <input type="radio" name="audio-library-item" checked={selectedId === item.id} disabled={Boolean(reason)} onChange={() => setSelectedId(item.id)} />
                <AudioPlayer src={item.audio_url} compact />
                <span className="library-item__copy"><strong>{item.title}</strong><span>{item.transcript}</span></span>
                <span className="library-item__meta">{itemTime(item.created_at) || duration(item.duration_seconds)}</span>
              </label>;
            })}
          </div>
          <div className="audio-library__preview">
            {selected ? <><div className="library-preview__header"><span>Audio Preview</span><VisualizationToggle mode={previewMode} disabled={!previewAvailable} onChange={setPreviewMode} /></div><AudioVisualizer src={selected.audio_url} mode={previewMode} onUnavailable={() => { setPreviewAvailable(false); setPreviewMode("waveform"); }} /><div className="library-preview__meta"><span>{duration(selected.duration_seconds)}</span>{selected.multiple_speakers && <span><Users size={13} /> Multiple speakers</span>}</div><label><span>Transcript</span><textarea readOnly value={selected.transcript} /></label></> : <div className="library-empty">Select one audio to preview</div>}
          </div>
        </div>
        <footer className="audio-library__footer"><div className="library-pagination"><button className="icon-button" aria-label="Previous page" disabled={!page || page.page <= 1} onClick={() => changePage(pageNumber - 1)}><ChevronLeft size={16} /></button><span>{page?.page || 1} / {page?.total_pages || 1}</span><button className="icon-button" aria-label="Next page" disabled={!page || page.page >= page.total_pages} onClick={() => changePage(pageNumber + 1)}><ChevronRight size={16} /></button></div>{error && <span className="field-error">{error}</span>}<button className="secondary-button" disabled={submitting} onClick={onCancel}>Cancel</button><button className="primary-button" disabled={!selected || Boolean(selected && disabledReason(selected)) || submitting} onClick={() => void apply()}><Check size={15} /> Use Selected</button></footer>
      </section>
      {uploadsEnabled && pendingFile && <AudioUploadDialog file={pendingFile} title="Upload Audio" error={error} recognitionAvailable={recognitionAvailable} onRecognize={(file) => studioApi.recognize(sessionId, file)} onCancel={() => { setPendingFile(null); setError(""); }} onConfirm={saveUpload} />}
    </div>
  );
}
