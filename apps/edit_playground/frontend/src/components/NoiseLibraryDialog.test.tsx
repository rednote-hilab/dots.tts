import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { studioApi } from "../api";
import type { AudioReference, AudioSegment, NoiseLibraryPage, SessionSnapshot } from "../types";
import {
  NoiseLibraryDialog,
  noiseStrengthFromSnr,
  snrFromNoiseStrength,
} from "./NoiseLibraryDialog";

const createWave = vi.hoisted(() => vi.fn());
vi.mock("wavesurfer.js", () => ({ default: { create: createWave } }));
vi.mock("wavesurfer.js/plugins/spectrogram", () => ({
  default: { create: vi.fn() },
}));

const segment: AudioSegment = {
  id: "segment-one",
  source_id: "segment-one",
  audio_url: "/source.wav",
  transcript: "Source transcript",
  duration_seconds: 2,
  revision_id: null,
  multiple_speakers: false,
  allow_xvector: true,
};

const reference: AudioReference = {
  source_id: "edit_source",
  audio_url: "/source.wav",
  text: segment.transcript,
  revision_id: null,
  duration_seconds: 2,
  composition_version: "version-one",
  use_xvector: true,
  segments: [segment],
};

const presets: NoiseLibraryPage = {
  items: [{
    id: "ambient",
    kind: "presets",
    title: "Ambient",
    audio_url: "/ambient.wav",
    duration_seconds: 10,
    created_at: null,
  }],
  page: 1,
  page_size: 10,
  total: 1,
  total_pages: 1,
};

beforeEach(() => {
  vi.restoreAllMocks();
  createWave.mockReset();
  createWave.mockReturnValue({
    on: vi.fn(() => () => {}),
    destroy: vi.fn(),
    pause: vi.fn(),
    playPause: vi.fn(),
  });
  vi.spyOn(studioApi, "noiseLibrary").mockImplementation(
    async (_session, kind) => kind === "presets"
      ? presets
      : { ...presets, items: [], total: 0 },
  );
});

describe("NoiseLibraryDialog", () => {
  it("maps the visual slider from weak on the left to strong on the right", () => {
    expect(noiseStrengthFromSnr(20)).toBe(0);
    expect(noiseStrengthFromSnr(0)).toBe(20);
    expect(snrFromNoiseStrength(0)).toBe(20);
    expect(snrFromNoiseStrength(20)).toBe(0);
  });

  it("selects a noise and applies the chosen SNR atomically", async () => {
    const snapshot = { session_id: "browser" } as SessionSnapshot;
    const setNoise = vi.spyOn(studioApi, "setSourceNoise").mockResolvedValue(snapshot);
    const onApplied = vi.fn();
    render(
      <NoiseLibraryDialog
        sessionId="browser"
        reference={reference}
        expectedVersion="version-one"
        onCancel={() => {}}
        onApplied={onApplied}
      />,
    );
    const dialog = screen.getByRole("dialog", { name: "Background Noise for Full Source Audio" });
    expect(within(dialog).getByRole("tab", { name: "Presets" })).toHaveAttribute("aria-selected", "true");
    fireEvent.click(await within(dialog).findByRole("radio"));
    fireEvent.change(within(dialog).getByRole("slider", { name: "Noise Strength" }), {
      target: { value: "16" },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: /Apply Noise/ }));

    await waitFor(() => expect(setNoise).toHaveBeenCalledWith(
      "browser",
      "presets",
      "ambient",
      4,
      "version-one",
    ));
    expect(onApplied).toHaveBeenCalledWith(snapshot);
  });

  it("removes an existing overlay without affecting the source selection", async () => {
    const snapshot = { session_id: "browser" } as SessionSnapshot;
    const clearNoise = vi.spyOn(studioApi, "clearSourceNoise").mockResolvedValue(snapshot);
    render(
      <NoiseLibraryDialog
        sessionId="browser"
        reference={{
          ...reference,
          noise: { item_id: "ambient", kind: "presets", snr_db: 12 },
        }}
        expectedVersion="version-two"
        onCancel={() => {}}
        onApplied={() => {}}
      />,
    );
    const dialog = screen.getByRole("dialog", { name: "Background Noise for Full Source Audio" });
    fireEvent.click(within(dialog).getByRole("button", { name: /Remove Noise/ }));
    await waitFor(() => expect(clearNoise).toHaveBeenCalledWith(
      "browser",
      "version-two",
    ));
  });

  it("keeps preset noise while hiding upload UI when uploads are disabled", async () => {
    const { container } = render(
      <NoiseLibraryDialog
        sessionId="browser"
        reference={reference}
        expectedVersion="version-one"
        uploadsEnabled={false}
        onCancel={() => {}}
        onApplied={() => {}}
      />,
    );
    const dialog = screen.getByRole("dialog", { name: "Background Noise for Full Source Audio" });
    expect(await within(dialog).findByRole("radio")).toBeInTheDocument();
    expect(within(dialog).queryByRole("tab", { name: "Uploads" })).not.toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: /Apply Noise/ })).toBeInTheDocument();
    expect(container.querySelector('input[type="file"]')).not.toBeInTheDocument();
    expect(studioApi.noiseLibrary).not.toHaveBeenCalledWith("browser", "uploads", expect.anything());
  });
});
