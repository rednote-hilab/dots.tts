import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DRAG_TYPE } from "./AudioCard";
import { ReferenceDropZone } from "./ReferenceDropZone";

describe("ReferenceDropZone", () => {
  it("accepts a routed audio reference across the whole paired block", () => {
    const route = vi.fn();
    const { container } = render(
      <ReferenceDropZone destination="edit_source" onRoute={route} onUpload={() => {}}>
        <div>Source text area</div>
      </ReferenceDropZone>,
    );
    const target = container.firstElementChild!;
    const dataTransfer = {
      types: [DRAG_TYPE],
      files: [],
      dropEffect: "none",
      getData: (type: string) => type === DRAG_TYPE ? "revision-one" : "",
    };
    fireEvent.dragEnter(target, { dataTransfer });
    expect(screen.getByText("Drop as Edit Source")).toBeInTheDocument();
    fireEvent.drop(target, { dataTransfer });
    expect(route).toHaveBeenCalledWith("revision-one", "edit_source");
    expect(screen.queryByText("Drop as Edit Source")).not.toBeInTheDocument();
  });

  it("clears its overlay after cancelled drags and window blur", () => {
    const { container } = render(
      <ReferenceDropZone destination="prompt" onRoute={() => {}} onUpload={() => {}}>
        <div><span>Prompt transcript area</span></div>
      </ReferenceDropZone>,
    );
    const target = container.firstElementChild!;
    const dataTransfer = { types: [DRAG_TYPE], files: [], dropEffect: "none", getData: () => "" };
    fireEvent.dragEnter(target, { dataTransfer });
    expect(screen.getByText("Drop as Voice Prompt")).toBeInTheDocument();
    fireEvent(window, new Event("dragend"));
    expect(screen.queryByText("Drop as Voice Prompt")).not.toBeInTheDocument();
    fireEvent.dragEnter(target, { dataTransfer });
    fireEvent(window, new Event("blur"));
    expect(screen.queryByText("Drop as Voice Prompt")).not.toBeInTheDocument();
  });

  it("does not flicker or accumulate state while crossing nested children", () => {
    const { container } = render(
      <ReferenceDropZone destination="edit_source" onRoute={() => {}} onUpload={() => {}}>
        <div data-testid="audio"><span data-testid="transcript">Transcript</span></div>
      </ReferenceDropZone>,
    );
    const target = container.firstElementChild!;
    const audio = screen.getByTestId("audio");
    const transcript = screen.getByTestId("transcript");
    const dataTransfer = { types: [DRAG_TYPE], files: [], dropEffect: "none", getData: () => "" };
    fireEvent.dragEnter(target, { dataTransfer, relatedTarget: null });
    fireEvent.dragEnter(transcript, { dataTransfer, relatedTarget: audio });
    fireEvent.dragLeave(audio, { dataTransfer, relatedTarget: transcript });
    expect(screen.getByText("Drop as Edit Source")).toBeInTheDocument();
    fireEvent.dragLeave(target, { dataTransfer, relatedTarget: null });
    expect(screen.queryByText("Drop as Edit Source")).not.toBeInTheDocument();
  });

  it("rejects local file drops when uploads are disabled but preserves routed audio", () => {
    const route = vi.fn();
    const upload = vi.fn();
    const { container } = render(
      <ReferenceDropZone destination="edit_source" onRoute={route} onUpload={upload} uploadsEnabled={false}>
        <div>Protected source</div>
      </ReferenceDropZone>,
    );
    const target = container.firstElementChild!;
    const file = new File(["audio"], "source.wav", { type: "audio/wav" });
    fireEvent.dragEnter(target, {
      dataTransfer: { types: ["Files"], files: [file], getData: () => "" },
    });
    expect(screen.queryByText("Drop as Edit Source")).not.toBeInTheDocument();
    fireEvent.drop(target, {
      dataTransfer: { types: ["Files"], files: [file], getData: () => "" },
    });
    expect(upload).not.toHaveBeenCalled();

    const routed = {
      types: [DRAG_TYPE],
      files: [],
      dropEffect: "none",
      getData: (type: string) => type === DRAG_TYPE ? "revision-two" : "",
    };
    fireEvent.dragEnter(target, { dataTransfer: routed });
    expect(screen.getByText("Drop as Edit Source")).toBeInTheDocument();
    fireEvent.drop(target, { dataTransfer: routed });
    expect(route).toHaveBeenCalledWith("revision-two", "edit_source");
  });
});
