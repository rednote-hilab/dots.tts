import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AudioUploadDialog } from "./AudioUploadDialog";

const createWave = vi.hoisted(() => vi.fn());
vi.mock("wavesurfer.js", () => ({ default: { create: createWave } }));
vi.mock("wavesurfer.js/plugins/spectrogram", () => ({ default: { create: vi.fn() } }));

describe("AudioUploadDialog", () => {
  beforeEach(() => {
    URL.createObjectURL = vi.fn(() => "blob:test");
    URL.revokeObjectURL = vi.fn();
    createWave.mockReset();
    createWave.mockReturnValue({ on: vi.fn(() => () => {}), destroy: vi.fn(), pause: vi.fn(), playPause: vi.fn() });
  });
  it("requires a transcript and sends the multiple-speaker choice atomically", async () => {
    const confirm = vi.fn().mockResolvedValue(undefined);
    render(<AudioUploadDialog file={new File(["audio"], "clip.wav", { type: "audio/wav" })} title="Add Audio Segment" onCancel={() => {}} onConfirm={confirm} />);
    const submit = screen.getByRole("button", { name: "Use Audio" });
    expect(submit).toBeDisabled();
    fireEvent.change(screen.getByRole("textbox", { name: "Transcript" }), { target: { value: "  paired words  " } });
    fireEvent.click(screen.getByRole("checkbox", { name: "This audio contains multiple speakers" }));
    fireEvent.click(screen.getByRole("checkbox", { name: /Allow this audio/ }));
    fireEvent.click(submit);
    await waitFor(() => expect(confirm).toHaveBeenCalledWith("paired words", true, true));
  });

  it("cancels without confirming", () => {
    const cancel = vi.fn();
    const confirm = vi.fn().mockResolvedValue(undefined);
    render(<AudioUploadDialog file={new File(["audio"], "clip.wav", { type: "audio/wav" })} title="Upload Voice Prompt" onCancel={cancel} onConfirm={confirm} />);
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(cancel).toHaveBeenCalledOnce();
    expect(confirm).not.toHaveBeenCalled();
  });

  it("uses the full visualization player in the upload dialog", () => {
    render(<AudioUploadDialog file={new File(["audio"], "clip.wav", { type: "audio/wav" })} title="Upload Audio" onCancel={() => {}} onConfirm={vi.fn().mockResolvedValue(undefined)} />);
    expect(screen.getByRole("button", { name: "Show spectrogram" })).toBeInTheDocument();
    expect(document.querySelector(".audio-visualizer")).toBeInTheDocument();
    expect(document.querySelector(".upload-dialog > audio")).not.toBeInTheDocument();
  });

  it("recognizes a transcript without submitting the upload", async () => {
    const recognize = vi.fn().mockResolvedValue({ text: "识别后的文本", language: "Chinese" });
    const confirm = vi.fn().mockResolvedValue(undefined);
    const file = new File(["audio"], "clip.wav", { type: "audio/wav" });
    render(<AudioUploadDialog file={file} title="Upload Audio" recognitionAvailable onRecognize={recognize} onCancel={() => {}} onConfirm={confirm} />);
    fireEvent.click(screen.getByRole("button", { name: "Recognition" }));
    await waitFor(() => expect(recognize).toHaveBeenCalledWith(file));
    expect(screen.getByRole("textbox", { name: "Transcript" })).toHaveValue("识别后的文本");
    expect(confirm).not.toHaveBeenCalled();
  });

  it("rejects files larger than 8192 KiB before upload", () => {
    const file = new File([new Uint8Array(8192 * 1024 + 1)], "large.wav", { type: "audio/wav" });
    render(<AudioUploadDialog file={file} title="Upload Audio" recognitionAvailable onRecognize={vi.fn()} onCancel={() => {}} onConfirm={vi.fn().mockResolvedValue(undefined)} />);
    expect(screen.getByText("Audio file must not exceed 8192 KiB.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Recognition" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Use Audio" })).toBeDisabled();
  });
});
