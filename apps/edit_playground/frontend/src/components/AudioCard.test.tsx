import { act, fireEvent, render, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { AudioCard, safeCreateWaveform } from "./AudioCard";

const createWave = vi.hoisted(() => vi.fn());
const createSpectrogram = vi.hoisted(() => vi.fn());
vi.mock("wavesurfer.js", () => ({ default: { create: createWave } }));
vi.mock("wavesurfer.js/plugins/spectrogram", () => ({ default: { create: createSpectrogram } }));

let waveEvents: Record<string, (...values: any[]) => void>;
let pluginEvents: Record<string, (...values: any[]) => void>;
let wave: Record<string, ReturnType<typeof vi.fn>>;
let plugin: Record<string, ReturnType<typeof vi.fn>>;

beforeEach(() => {
  waveEvents = {};
  pluginEvents = {};
  wave = {
    on: vi.fn((name: string, callback: (...values: any[]) => void) => { waveEvents[name] = callback; return vi.fn(); }),
    destroy: vi.fn(), pause: vi.fn(), playPause: vi.fn(), seekTo: vi.fn(), registerPlugin: vi.fn(),
  };
  plugin = {
    on: vi.fn((name: string, callback: (...values: any[]) => void) => { pluginEvents[name] = callback; return vi.fn(); }),
  };
  createWave.mockReset();
  createWave.mockReturnValue(wave);
  createSpectrogram.mockReset();
  createSpectrogram.mockReturnValue(plugin);
  vi.stubGlobal("Worker", class Worker {});
});

it("falls back to the native audio player when waveform setup throws", async () => {
  expect(safeCreateWaveform(() => { throw new Error("waveform unavailable"); })).toBeNull();
  createWave.mockReturnValue(null);
  const { container } = render(
    <AudioCard
      title="Source Audio"
      data={{ sourceId: "source", audioUrl: "/source.wav", text: "hello" }}
      destination="edit_source"
      onRoute={() => {}}
    />,
  );
  await waitFor(() => expect(container.querySelector("audio")).toBeInTheDocument());
  expect(container.querySelector(".waveform-canvas")).not.toBeInTheDocument();
});

it("opens the audio library from the empty audio action", () => {
  const upload = vi.fn();
  const { getByRole } = render(
    <AudioCard title="Prompt Audio" data={null} destination="prompt" onUpload={upload} onRoute={() => {}} />,
  );
  fireEvent.click(getByRole("button", { name: "Drop or choose audio" }));
  expect(upload).toHaveBeenCalledWith();
});

it("decodes at 48 kHz and lazily creates a full-band linear spectrogram", async () => {
  const view = render(<AudioCard title="Source Audio" data={{ sourceId: "source", audioUrl: "/source.wav", text: "hello" }} destination="edit_source" onRoute={() => {}} />);
  expect(createWave).toHaveBeenCalledWith(expect.objectContaining({ sampleRate: 48_000, height: 54 }));
  expect(createSpectrogram).not.toHaveBeenCalled();
  expect(view.getByRole("button", { name: "Show spectrogram" })).toBeInTheDocument();
  act(() => waveEvents.ready(4));
  fireEvent.click(view.getByRole("button", { name: "Show spectrogram" }));
  await waitFor(() => expect(createSpectrogram).toHaveBeenCalledWith(expect.objectContaining({
    sampleRate: 48_000,
    height: 168,
    frequencyMin: 0,
    frequencyMax: 24_000,
    fftSamples: 2048,
    scale: "linear",
    windowFunc: "hann",
    useWebWorker: true,
    fallbackToMainThread: false,
  })));
  expect(wave.registerPlugin).toHaveBeenCalledWith(plugin);
  expect(view.container.querySelector(".audio-visualizer")).toHaveClass("audio-visualizer--spectrogram");
  fireEvent.click(view.getByRole("button", { name: "Show waveform" }));
  expect(createSpectrogram).toHaveBeenCalledOnce();
  expect(createWave).toHaveBeenCalledOnce();
});

it("seeks through the spectrogram and keeps its playhead synchronized", async () => {
  const view = render(<AudioCard title="Prompt Audio" data={{ sourceId: "prompt", audioUrl: "/prompt.wav", text: "hello" }} destination="prompt" onRoute={() => {}} />);
  act(() => waveEvents.ready(8));
  fireEvent.click(view.getByRole("button", { name: "Show spectrogram" }));
  await waitFor(() => expect(pluginEvents.click).toBeTypeOf("function"));
  act(() => pluginEvents.click(0.25));
  expect(wave.seekTo).toHaveBeenCalledWith(0.25);
  act(() => waveEvents.timeupdate(2));
  expect(view.container.querySelector<HTMLElement>(".spectrogram-playhead")?.style.left).toBe("25%");
});

it("falls back to waveform and disables the toggle after a spectrogram failure", async () => {
  createSpectrogram.mockImplementation(() => { throw new Error("worker failed"); });
  const view = render(<AudioCard title="Source Audio" data={{ sourceId: "source", audioUrl: "/source.wav", text: "hello" }} destination="edit_source" onRoute={() => {}} />);
  act(() => waveEvents.ready(4));
  fireEvent.click(view.getByRole("button", { name: "Show spectrogram" }));
  await waitFor(() => expect(view.getByRole("button", { name: "Spectrogram unavailable" })).toBeDisabled());
  expect(view.container.querySelector(".audio-visualizer")).toHaveClass("audio-visualizer--waveform");
});

it("keeps compact players dense and gives full players independent modes", () => {
  const view = render(<><AudioCard title="R1" compact data={{ sourceId: "one", audioUrl: "/one.wav", text: "one" }} onRoute={() => {}} /><AudioCard title="R2" data={{ sourceId: "two", audioUrl: "/two.wav", text: "two" }} onRoute={() => {}} /></>);
  expect(view.getAllByRole("button", { name: "Show spectrogram" })).toHaveLength(1);
  expect(view.container.querySelector(".audio-card--compact .audio-visualizer")).not.toBeInTheDocument();
});
