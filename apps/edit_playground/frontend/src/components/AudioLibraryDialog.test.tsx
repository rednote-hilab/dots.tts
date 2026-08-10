import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { studioApi } from "../api";
import type { AudioLibraryPage, SessionSnapshot } from "../types";
import { AudioLibraryDialog } from "./AudioLibraryDialog";

const createWave = vi.hoisted(() => vi.fn());
vi.mock("wavesurfer.js", () => ({ default: { create: createWave } }));
vi.mock("wavesurfer.js/plugins/spectrogram", () => ({ default: { create: vi.fn() } }));

const presetPage: AudioLibraryPage = {
  items: [{
    id: "female_en",
    kind: "presets",
    title: "female_en",
    audio_url: "/female.wav",
    transcript: "Preset transcript",
    duration_seconds: 1.5,
    multiple_speakers: false,
    created_at: null,
  }],
  page: 1,
  page_size: 10,
  total: 11,
  total_pages: 2,
};

const emptyPage = (_kind: "uploads" | "revisions"): AudioLibraryPage => ({
  items: [], page: 1, page_size: 10, total: 0, total_pages: 1,
});

beforeEach(() => {
  vi.restoreAllMocks();
  Object.defineProperty(URL, "createObjectURL", { configurable: true, value: vi.fn(() => "blob:audio") });
  Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });
  createWave.mockReset();
  createWave.mockReturnValue({
    on: vi.fn(() => () => {}),
    destroy: vi.fn(),
    pause: vi.fn(),
    playPause: vi.fn(),
  });
  vi.spyOn(studioApi, "audioLibrary").mockImplementation(async (_session, kind, page = 1) => {
    if (kind === "presets") return { ...presetPage, page, items: page === 1 ? presetPage.items : [] };
    return emptyPage(kind);
  });
});

describe("AudioLibraryDialog", () => {
  it("defaults to Presets and applies only the confirmed single selection", async () => {
    const appliedSnapshot = { session_id: "browser" } as SessionSnapshot;
    const apply = vi.spyOn(studioApi, "applyAudioLibrary").mockResolvedValue(appliedSnapshot);
    const onApplied = vi.fn();
    render(<AudioLibraryDialog sessionId="browser" intent={{ title: "Choose Edit Source", destination: "edit_source", action: "bind" }} onCancel={() => {}} onApplied={onApplied} />);

    const dialog = screen.getByRole("dialog", { name: "Choose Edit Source" });
    expect(within(dialog).getByRole("tab", { name: "Presets" })).toHaveAttribute("aria-selected", "true");
    const confirm = within(dialog).getByRole("button", { name: /Use Selected/ });
    expect(confirm).toBeDisabled();
    const radio = await within(dialog).findByRole("radio");
    fireEvent.click(radio);
    expect(apply).not.toHaveBeenCalled();
    expect(within(dialog).getByDisplayValue("Preset transcript")).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Show spectrogram" })).toBeInTheDocument();
    expect(within(dialog).getAllByRole("button", { name: "Play" })).toHaveLength(2);
    fireEvent.click(confirm);

    await waitFor(() => expect(apply).toHaveBeenCalledWith("browser", {
      kind: "presets",
      item_id: "female_en",
      destination: "edit_source",
      action: "bind",
      expected_version: undefined,
      index: undefined,
      replace_segment_id: undefined,
    }));
    expect(onApplied).toHaveBeenCalledWith(appliedSnapshot, presetPage.items[0]);
  });

  it("clears selection when switching tabs or pages", async () => {
    vi.spyOn(studioApi, "applyAudioLibrary").mockResolvedValue({ session_id: "browser" } as SessionSnapshot);
    render(<AudioLibraryDialog sessionId="browser" intent={{ title: "Add Audio Segment", destination: "edit_source", action: "insert", index: 1, expected_version: "v1", currentDuration: 1, currentSegmentCount: 1 }} onCancel={() => {}} onApplied={() => {}} />);
    const dialog = screen.getByRole("dialog", { name: "Add Audio Segment" });
    fireEvent.click(await within(dialog).findByRole("radio"));
    expect(within(dialog).getByRole("button", { name: /Use Selected/ })).toBeEnabled();
    fireEvent.click(within(dialog).getByRole("tab", { name: "Uploads" }));
    await waitFor(() => expect(studioApi.audioLibrary).toHaveBeenCalledWith("browser", "uploads", 1));
    expect(within(dialog).getByRole("button", { name: /Use Selected/ })).toBeDisabled();
    fireEvent.click(within(dialog).getByRole("tab", { name: "Presets" }));
    await within(dialog).findByRole("radio");
    fireEvent.click(within(dialog).getByLabelText("Next page"));
    await waitFor(() => expect(studioApi.audioLibrary).toHaveBeenCalledWith("browser", "presets", 2));
    expect(within(dialog).getByRole("button", { name: /Use Selected/ })).toBeDisabled();
  });

  it("saves a nested upload into the catalog without applying it", async () => {
    const uploaded = {
      id: "upload-one",
      kind: "uploads" as const,
      title: "clip.wav",
      audio_url: "/clip.wav",
      transcript: "Uploaded transcript",
      duration_seconds: 2,
      multiple_speakers: true,
      created_at: "2026-07-21T00:00:00+00:00",
    };
    vi.spyOn(studioApi, "createLibraryUpload").mockResolvedValue({ items: [uploaded], page: 1, page_size: 10, total: 1, total_pages: 1, selected_id: uploaded.id });
    const apply = vi.spyOn(studioApi, "applyAudioLibrary").mockResolvedValue({ session_id: "browser" } as SessionSnapshot);
    const { container } = render(<AudioLibraryDialog sessionId="browser" intent={{ title: "Choose Voice Prompt", destination: "prompt", action: "bind" }} onCancel={() => {}} onApplied={() => {}} />);
    const parent = screen.getByRole("dialog", { name: "Choose Voice Prompt" });
    fireEvent.click(within(parent).getByRole("tab", { name: "Uploads" }));
    await waitFor(() => expect(studioApi.audioLibrary).toHaveBeenCalledWith("browser", "uploads", 1));
    const input = container.querySelector<HTMLInputElement>('input[type="file"]')!;
    fireEvent.change(input, { target: { files: [new File(["audio"], "clip.wav", { type: "audio/wav" })] } });
    const uploadDialog = screen.getByRole("dialog", { name: "Upload Audio" });
    fireEvent.change(within(uploadDialog).getByPlaceholderText("Enter the transcript paired with this audio."), { target: { value: "Uploaded transcript" } });
    fireEvent.click(within(uploadDialog).getByRole("checkbox", { name: "This audio contains multiple speakers" }));
    fireEvent.click(within(uploadDialog).getByRole("checkbox", { name: /Allow this audio/ }));
    fireEvent.click(within(uploadDialog).getByRole("button", { name: /Use Audio/ }));

    await waitFor(() => expect(studioApi.createLibraryUpload).toHaveBeenCalledWith("browser", expect.any(File), "Uploaded transcript", true, true));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Upload Audio" })).not.toBeInTheDocument());
    expect(apply).not.toHaveBeenCalled();
    expect(within(parent).getByRole("radio")).toBeChecked();
    expect(within(parent).getByRole("button", { name: /Use Selected/ })).toBeEnabled();
  });

  it("omits upload and recognition UI when uploads are disabled", async () => {
    const { container } = render(<AudioLibraryDialog sessionId="browser" intent={{ title: "Choose Edit Source", destination: "edit_source", action: "bind", initialFile: new File(["audio"], "ignored.wav") }} uploadsEnabled={false} recognitionAvailable onCancel={() => {}} onApplied={() => {}} />);
    const dialog = screen.getByRole("dialog", { name: "Choose Edit Source" });
    await within(dialog).findByRole("radio");
    expect(within(dialog).queryByRole("tab", { name: "Uploads" })).not.toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: "Upload Audio" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Recognition" })).not.toBeInTheDocument();
    expect(container.querySelector('input[type="file"]')).not.toBeInTheDocument();
    expect(studioApi.audioLibrary).not.toHaveBeenCalledWith("browser", "uploads", expect.anything());
  });
});
