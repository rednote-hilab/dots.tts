import { useEffect, useMemo, useRef, useState } from "react";
import { Check, ChevronLeft, ChevronRight, Trash2, Upload, Wind, X } from "lucide-react";
import { studioApi } from "../api";
import type { AudioReference, NoiseLibraryKind, NoiseLibraryPage, SessionSnapshot } from "../types";
import { AudioPlayer, AudioVisualizer, VisualizationToggle, type VisualizationMode } from "./AudioCard";

const TABS: Array<{ id: NoiseLibraryKind; label: string }> = [
  { id: "presets", label: "Presets" },
  { id: "uploads", label: "Uploads" },
];

function duration(seconds: number) {
  if (!Number.isFinite(seconds)) return "0:00";
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
}

export function noiseStrengthFromSnr(snrDb: number) {
  return 20 - snrDb;
}

export function snrFromNoiseStrength(strength: number) {
  return 20 - strength;
}

export function NoiseLibraryDialog({ sessionId, reference, expectedVersion, onCancel, onApplied, uploadsEnabled = true }: {
  sessionId: string;
  reference: AudioReference;
  expectedVersion: string;
  onCancel: () => void;
  onApplied: (snapshot: SessionSnapshot) => void;
  uploadsEnabled?: boolean;
}) {
  const [tab, setTab] = useState<NoiseLibraryKind>(uploadsEnabled ? reference.noise?.kind || "presets" : "presets");
  const [pageNumber, setPageNumber] = useState(1);
  const [page, setPage] = useState<NoiseLibraryPage | null>(null);
  const [selectedId, setSelectedId] = useState(reference.noise?.item_id || "");
  const [snrDb, setSnrDb] = useState(reference.noise?.snr_db ?? 15);
  const [mode, setMode] = useState<VisualizationMode>("waveform");
  const [visualizationAvailable, setVisualizationAvailable] = useState(true);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [historyConsent, setHistoryConsent] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const load = async (kind: NoiseLibraryKind, nextPage: number) => {
    setLoading(true);
    setError("");
    try {
      const value = await studioApi.noiseLibrary(sessionId, kind, nextPage);
      setPage(value);
      if (value.selected_id) setSelectedId(value.selected_id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void load(tab, pageNumber); }, [sessionId, tab, pageNumber]);
  const selected = useMemo(
    () => page?.items.find((item) => item.id === selectedId) || null,
    [page, selectedId],
  );

  const changeTab = (kind: NoiseLibraryKind) => {
    setTab(kind);
    setPageNumber(1);
    setSelectedId(kind === reference.noise?.kind ? reference.noise.item_id : "");
  };

  const upload = async (file: File) => {
    setSubmitting(true);
    setError("");
    try {
      const value = await studioApi.createNoiseUpload(sessionId, file, historyConsent);
      setTab("uploads");
      setPageNumber(1);
      setPage(value);
      setSelectedId(value.selected_id || "");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSubmitting(false);
    }
  };

  const apply = async () => {
    if (!selected) return;
    setSubmitting(true);
    setError("");
    try {
      onApplied(await studioApi.setSourceNoise(
        sessionId,
        selected.kind,
        selected.id,
        snrDb,
        expectedVersion,
      ));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSubmitting(false);
    }
  };

  const remove = async () => {
    setSubmitting(true);
    setError("");
    try {
      onApplied(await studioApi.clearSourceNoise(sessionId, expectedVersion));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSubmitting(false);
    }
  };

  return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onCancel()}>
    <section className="audio-library noise-library" role="dialog" aria-modal="true" aria-label="Background Noise for Full Source Audio">
      <header className="audio-library__header"><div><Wind size={18} /><h2>Background Noise</h2></div><button className="icon-button" aria-label="Close Noise Library" onClick={onCancel}><X size={17} /></button></header>
      <div className="audio-library__tabs" role="tablist">{TABS.filter((item) => uploadsEnabled || item.id !== "uploads").map((item) => <button key={item.id} role="tab" aria-selected={tab === item.id} className={tab === item.id ? "active" : ""} onClick={() => changeTab(item.id)}>{item.label}</button>)}</div>
      {uploadsEnabled && tab === "uploads" && <div className="audio-library__upload"><input ref={fileInput} hidden type="file" accept="audio/*" onChange={(event) => { const file = event.currentTarget.files?.[0]; event.currentTarget.value = ""; if (file) void upload(file); }} /><button className="secondary-button" disabled={submitting} onClick={() => fileInput.current?.click()}><Upload size={15} /> Upload Noise</button><label className="multiple-speakers"><input type="checkbox" checked={historyConsent} onChange={(event) => setHistoryConsent(event.target.checked)} /> Allow private history retention / 允许私下保存历史</label></div>}
      <div className="audio-library__body">
        <div className="audio-library__list" role="radiogroup" aria-label={`${tab} noise`}>
          {loading && <div className="library-empty">Loading…</div>}
          {!loading && !page?.items.length && <div className="library-empty">No noise available</div>}
          {!loading && page?.items.map((item) => <label key={item.id} className={`library-item ${selectedId === item.id ? "library-item--selected" : ""}`}>
            <input type="radio" name="noise-library-item" checked={selectedId === item.id} onChange={() => { setSelectedId(item.id); setVisualizationAvailable(true); setMode("waveform"); }} />
            <AudioPlayer src={item.audio_url} compact />
            <span className="library-item__copy"><strong>{item.title}</strong><span>Background noise</span></span>
            <span className="library-item__meta">{duration(item.duration_seconds)}</span>
          </label>)}
        </div>
        <div className="audio-library__preview">
          {selected ? <><div className="library-preview__header"><span>Noise Preview</span><VisualizationToggle mode={mode} disabled={!visualizationAvailable} onChange={setMode} /></div><AudioVisualizer src={selected.audio_url} mode={mode} onUnavailable={() => { setVisualizationAvailable(false); setMode("waveform"); }} /><div className="noise-level"><div><span>Noise Strength</span><output>{snrDb} dB SNR</output></div><input aria-label="Noise Strength" aria-valuetext={`${snrDb} dB SNR`} type="range" min={0} max={20} step={1} value={noiseStrengthFromSnr(snrDb)} onChange={(event) => setSnrDb(snrFromNoiseStrength(Number(event.target.value)))} /><div className="noise-level__legend"><span>Weak</span><span>Strong</span></div></div></> : <div className="library-empty">Select one noise to preview</div>}
        </div>
      </div>
      <footer className="audio-library__footer"><div className="library-pagination"><button className="icon-button" aria-label="Previous page" disabled={!page || page.page <= 1} onClick={() => setPageNumber((value) => value - 1)}><ChevronLeft size={16} /></button><span>{page?.page || 1} / {page?.total_pages || 1}</span><button className="icon-button" aria-label="Next page" disabled={!page || page.page >= page.total_pages} onClick={() => setPageNumber((value) => value + 1)}><ChevronRight size={16} /></button></div>{error && <span className="field-error">{error}</span>}{reference.noise && <button className="secondary-button noise-remove" disabled={submitting} onClick={() => void remove()}><Trash2 size={14} /> Remove Noise</button>}<button className="secondary-button" disabled={submitting} onClick={onCancel}>Cancel</button><button className="primary-button" disabled={!selected || submitting} onClick={() => void apply()}><Check size={15} /> Apply Noise</button></footer>
    </section>
  </div>;
}
