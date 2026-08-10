import { useEffect, useMemo, useState } from "react";
import { Check, LoaderCircle, ScanText, X } from "lucide-react";
import { MAX_AUDIO_SECONDS, MAX_UPLOAD_BYTES } from "../api";
import type { RecognitionResult } from "../types";
import { AudioVisualizer, type VisualizationMode, VisualizationToggle } from "./AudioCard";

export function AudioUploadDialog({ file, title, error = "", onCancel, onConfirm, recognitionAvailable = false, onRecognize }: {
  file: File;
  title: string;
  error?: string;
  onCancel: () => void;
  onConfirm: (transcript: string, multipleSpeakers: boolean, historyConsent: boolean) => Promise<void>;
  recognitionAvailable?: boolean;
  onRecognize?: (file: File) => Promise<RecognitionResult>;
}) {
  const [transcript, setTranscript] = useState("");
  const [multipleSpeakers, setMultipleSpeakers] = useState(false);
  const [historyConsent, setHistoryConsent] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [recognizing, setRecognizing] = useState(false);
  const [localError, setLocalError] = useState("");
  const [mode, setMode] = useState<VisualizationMode>("waveform");
  const [visualizationAvailable, setVisualizationAvailable] = useState(true);
  const url = useMemo(() => URL.createObjectURL(file), [file]);
  useEffect(() => () => URL.revokeObjectURL(url), [url]);
  useEffect(() => {
    setLocalError(
      file.size > MAX_UPLOAD_BYTES ? "Audio file must not exceed 8192 KiB." : "",
    );
    const audio = new Audio(url);
    const validateDuration = () => {
      if (Number.isFinite(audio.duration) && audio.duration > MAX_AUDIO_SECONDS) {
        setLocalError("Audio must not exceed 30 seconds.");
      }
    };
    audio.addEventListener("loadedmetadata", validateDuration);
    return () => audio.removeEventListener("loadedmetadata", validateDuration);
  }, [file.size, url]);
  useEffect(() => {
    const close = (event: KeyboardEvent) => { if (event.key === "Escape" && !submitting) { event.stopImmediatePropagation(); onCancel(); } };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onCancel, submitting]);
  const confirm = async () => {
    setSubmitting(true);
    try { await onConfirm(transcript.trim(), multipleSpeakers, historyConsent); } finally { setSubmitting(false); }
  };
  const recognize = async () => {
    if (!onRecognize || localError) return;
    setRecognizing(true);
    setLocalError("");
    try {
      const result = await onRecognize(file);
      setTranscript(result.text);
    } catch (reason) {
      setLocalError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setRecognizing(false);
    }
  };
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onCancel()}>
      <section className="upload-dialog" role="dialog" aria-modal="true" aria-label={title}>
        <header><h2>{title}</h2><VisualizationToggle mode={mode} disabled={!visualizationAvailable} onChange={setMode} /><button className="icon-button modal-close" disabled={submitting} aria-label="Cancel upload" onClick={onCancel}><X size={17} /></button></header>
        <AudioVisualizer src={url} mode={mode} onUnavailable={() => { setVisualizationAvailable(false); setMode("waveform"); }} />
        <label className="transcript-field"><span className="upload-transcript-heading"><span>Transcript</span><button type="button" className="secondary-button recognition-button" aria-label="Recognition" disabled={!recognitionAvailable || recognizing || submitting || Boolean(localError)} title={recognitionAvailable ? "Recognize English or Mandarin speech" : "Recognition is unavailable"} onClick={() => void recognize()}>{recognizing ? <LoaderCircle className="spin" size={14} /> : <ScanText size={14} />} {recognizing ? "Recognizing…" : "Recognition"}</button></span><textarea aria-label="Transcript" autoFocus value={transcript} onChange={(event) => setTranscript(event.target.value)} placeholder="Enter the transcript paired with this audio." /></label>
        <p className="upload-constraints">Audio ≤ 30 seconds · File ≤ 8192 KiB</p>
        <label className="multiple-speakers"><input type="checkbox" checked={multipleSpeakers} onChange={(event) => setMultipleSpeakers(event.target.checked)} /> This audio contains multiple speakers</label>
        <label className="multiple-speakers"><input type="checkbox" checked={historyConsent} onChange={(event) => setHistoryConsent(event.target.checked)} /> Allow this audio and related generations to be retained privately for research feedback. / 允许私下保存此音频及相关推理结果用于研究反馈。</label>
        {(localError || error) && <div className="field-error upload-dialog__error">{localError || error}</div>}
        <footer><button className="secondary-button" disabled={submitting || recognizing} onClick={onCancel}>Cancel</button><button className="primary-button" disabled={submitting || recognizing || Boolean(localError) || !transcript.trim()} onClick={() => void confirm()}><Check size={15} /> Use Audio</button></footer>
      </section>
    </div>
  );
}
