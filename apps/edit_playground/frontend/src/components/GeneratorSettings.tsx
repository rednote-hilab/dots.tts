import { ChevronDown } from "lucide-react";
import type { EditGenerationSettings, GenerationSettings } from "../types";

const DEFAULT_SPEAKER_GUIDANCE_SCALE = 1.5;

type Settings = GenerationSettings | EditGenerationSettings;

interface Props<T extends Settings> {
  settings: T;
  odeMethods: string[];
  onChange: (settings: T) => void;
  speakerGuidanceAvailable: boolean;
  speakerGuidanceMode?: "boolean" | "auto";
  autoSpeakerGuidanceEnabled?: boolean;
}

export function GeneratorSettingsPanel<T extends Settings>({ settings, odeMethods, onChange, speakerGuidanceAvailable, speakerGuidanceMode = "boolean", autoSpeakerGuidanceEnabled = true }: Props<T>) {
  const requestedMode = settings.use_xvector;
  const speakerGuidanceEnabled = speakerGuidanceAvailable && (requestedMode === "auto" ? autoSpeakerGuidanceEnabled : requestedMode);
  const update = (patch: Partial<T>) => onChange({ ...settings, ...patch });
  return <div className="generation-config"><div className="settings-grid">
    <label>ODE Method<select value={settings.ode_method} onChange={(event) => update({ ode_method: event.target.value } as Partial<T>)}>{odeMethods.map((method) => <option key={method} value={method}>{method === "rk4" ? "RK4" : method[0].toUpperCase() + method.slice(1)}</option>)}</select></label>
    <RangeSetting label="Steps" min={1} max={32} step={1} value={settings.num_steps} onChange={(num_steps) => update({ num_steps } as Partial<T>)} />
    <RangeSetting label="Guidance" min={0} max={3} step={0.1} value={settings.guidance_scale} onChange={(guidance_scale) => update({ guidance_scale } as Partial<T>)} />
    <div className="speaker-guidance-setting">{speakerGuidanceMode === "auto" ? <><div className="speaker-guidance-heading"><span>Speaker Guidance</span><output>{!speakerGuidanceAvailable ? "Forced Off" : requestedMode === "auto" ? `Auto · ${speakerGuidanceEnabled ? "On" : "Off"}` : requestedMode ? "On" : "Off"}</output></div><div className="speaker-guidance-modes" role="group" aria-label="Speaker Guidance mode">{(["auto", true, false] as const).map((mode) => <button key={String(mode)} type="button" aria-pressed={requestedMode === mode} disabled={!speakerGuidanceAvailable} onClick={() => update({ use_xvector: mode, speaker_scale: mode === true ? DEFAULT_SPEAKER_GUIDANCE_SCALE : settings.speaker_scale } as Partial<T>)}>{mode === "auto" ? "Auto" : mode ? "On" : "Off"}</button>)}</div></> : <label className="speaker-guidance-toggle"><input type="checkbox" checked={speakerGuidanceEnabled} disabled={!speakerGuidanceAvailable} onChange={(event) => update({ use_xvector: event.target.checked, speaker_scale: event.target.checked ? DEFAULT_SPEAKER_GUIDANCE_SCALE : settings.speaker_scale } as Partial<T>)} /> Use Speaker Guidance</label>}<RangeSetting label="Scale" min={0} max={3} step={0.1} value={settings.speaker_scale} disabled={!speakerGuidanceEnabled} hideValue={!speakerGuidanceEnabled} hideThumb={!speakerGuidanceEnabled} onChange={(speaker_scale) => update({ speaker_scale } as Partial<T>)} /></div>
    <label>Seed<input type="number" value={settings.seed} onChange={(event) => update({ seed: Number(event.target.value) } as Partial<T>)} /></label>
  </div>{!speakerGuidanceAvailable && <div className="conditioning-off">Speaker guidance unavailable for this audio</div>}</div>;
}

function RangeSetting({ label, value, min, max, step, disabled = false, hideValue = false, hideThumb = false, onChange }: { label: string; value: number; min: number; max: number; step: number; disabled?: boolean; hideValue?: boolean; hideThumb?: boolean; onChange: (value: number) => void }) {
  return <label className={`range-setting ${hideThumb ? "range-setting--thumbless" : ""}`}><span>{label}{!hideValue && <output>{value}</output>}</span><input aria-label={label} type="range" value={value} min={min} max={max} step={step} disabled={disabled} onChange={(event) => onChange(Number(event.target.value))} /></label>;
}

export function SplitGenerationButton({ label, icon, leading, disabled, open, onOpenChange, onClick, children }: { label: string; icon: React.ReactNode; leading?: React.ReactNode; disabled: boolean; open: boolean; onOpenChange: (open: boolean) => void; onClick: () => void; children: React.ReactNode }) {
  return <div className={`split-generation ${open ? "split-generation--open" : ""}`}><div className="split-generation__bar">{leading}<div className="split-buttons"><button className="primary-button" disabled={disabled} onClick={onClick}>{icon}{label}</button><button className="primary-button split-arrow" aria-label={`${label} generation config`} aria-expanded={open} onClick={() => onOpenChange(!open)}><ChevronDown size={15} /></button></div></div><div className="generation-config-collapse" aria-hidden={!open}><div className="generation-config-collapse__inner" inert={!open}>{children}</div></div></div>;
}
