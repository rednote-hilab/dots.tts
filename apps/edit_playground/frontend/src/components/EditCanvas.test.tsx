import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { EditOperation } from "../types";
import { caretAnchorRect, caretOffsetFromPoint, constrainSelectionFocus, EditCanvas, EMOTION_TYPES, isUsableAnchorRect, operationConflicts, operationGlyph, PITCH_LEVELS, pitchLevelLabel, positionBlocked, rangeAnchorRect, RATE_VALUES, selectionConflicts, selectionOffsets, snapBoundary, textBoundaries, wordRangeAt } from "./EditCanvas";
import { OperationList } from "./OperationList";

const editorProps = {
  segmentId: "segment",
  operationColors: {} as Record<string, string>,
  editing: false,
  draft: "",
  hintVisible: false,
  onBeginEdit: () => {},
  onDraftChange: () => {},
  onSaveEdit: () => {},
  onCancelEdit: () => {},
  onDismissHint: () => {},
};

describe("EditCanvas model", () => {
  it("never exposes neutral pitch or rate slider values", () => {
    expect(PITCH_LEVELS).toEqual([
      { level: -3, semitones: -9 },
      { level: -2, semitones: -7 },
      { level: -1, semitones: -5 },
      { level: 1, semitones: 5 },
      { level: 2, semitones: 7 },
      { level: 3, semitones: 9 },
    ]);
    expect(RATE_VALUES).not.toContain(1);
    expect(RATE_VALUES[0]).toBe(0.55);
    expect(RATE_VALUES.at(-1)).toBe(1.95);
  });

  it("presents pitch levels without exposing semitone values", () => {
    expect(PITCH_LEVELS.map((item) => pitchLevelLabel(item.semitones))).toEqual([
      "Lower · level 3",
      "Lower · level 2",
      "Lower · level 1",
      "Raise · level 1",
      "Raise · level 2",
      "Raise · level 3",
    ]);
    const { container } = render(
      <OperationList
        operations={[{ segment_id: "segment", id: "pitch", kind: "pitch", start: 0, end: 5, params: { semitones: 7 } }]}
        colors={{}}
        segmentNumbers={{ segment: 1 }}
        onEdit={() => {}}
        onRemove={() => {}}
        onClear={() => {}}
      />,
    );
    expect(container).toHaveTextContent("Raise · level 2");
    expect(container).not.toHaveTextContent("semitone");
  });

  it("does not offer disgusted for newly created emotion operations", () => {
    expect(EMOTION_TYPES).not.toContain("disgusted");
  });

  it("rejects span overlap, nested spans and interior points", () => {
    const span: EditOperation = { segment_id: "segment", id: "one", kind: "emotion", start: 1, end: 4, params: { type: "happy" } };
    expect(operationConflicts({ segment_id: "segment", id: "two", kind: "pitch", start: 3, end: 5, params: { semitones: 2 } }, [span])).toBe(true);
    expect(operationConflicts({ segment_id: "segment", id: "point", kind: "pause", start: 2, end: 2, params: { act: "ins", level: 2 } }, [span])).toBe(true);
    expect(operationConflicts({ segment_id: "segment", id: "edge", kind: "pause", start: 4, end: 4, params: { act: "ins", level: 2 } }, [span])).toBe(false);
  });

  it("constrains forward and reverse selections at operation barriers", () => {
    const operations: EditOperation[] = [
      { segment_id: "segment", id: "span", kind: "emotion", start: 3, end: 6, params: { type: "happy" } },
      { segment_id: "segment", id: "point", kind: "pause", start: 8, end: 8, params: { act: "ins", level: 2 } },
    ];
    expect(constrainSelectionFocus(0, 5, operations)).toBe(3);
    expect(constrainSelectionFocus(10, 0, operations)).toBe(8);
    expect(constrainSelectionFocus(7, 10, operations)).toBe(8);
    expect(constrainSelectionFocus(7, 8, operations)).toBe(8);
    expect(constrainSelectionFocus(4, 10, operations)).toBeNull();
  });

  it("blocks operation interiors and duplicate points but releases deleted ranges", () => {
    const operations: EditOperation[] = [
      { segment_id: "segment", id: "span", kind: "delete", start: 2, end: 5, params: {} },
      { segment_id: "segment", id: "point", kind: "insert", start: 7, end: 7, params: { text: "x" } },
    ];
    expect(positionBlocked(3, operations)).toBe(true);
    expect(positionBlocked(2, operations)).toBe(false);
    expect(positionBlocked(7, operations)).toBe(true);
    expect(selectionConflicts(1, 6, operations)).toBe(true);
    expect(selectionConflicts(5, 7, operations)).toBe(false);
    expect(positionBlocked(3, [])).toBe(false);
    expect(selectionConflicts(1, 6, [])).toBe(false);
  });

  it("stops pointer selection at an existing operation without an overlap error", async () => {
    const operation: EditOperation = { segment_id: "segment", id: "span", kind: "delete", start: 3, end: 5, params: {} };
    const { container } = render(<EditCanvas {...editorProps} text="abcdefg" operations={[operation]} onOperationsChange={() => {}} />);
    const canvas = container.querySelector<HTMLElement>(".edit-canvas")!;
    container.querySelectorAll<HTMLElement>(".source-char").forEach((element, index) => {
      Object.defineProperty(element, "getBoundingClientRect", {
        configurable: true,
        value: () => new DOMRect(index * 10, 10, 10, 20),
      });
    });
    Object.defineProperty(Range.prototype, "getBoundingClientRect", {
      configurable: true,
      value: () => new DOMRect(0, 10, 30, 20),
    });
    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 1, clientY: 20 });
    fireEvent.pointerMove(canvas, { buttons: 1, pointerId: 1, clientX: 55, clientY: 20 });
    expect(container.querySelectorAll(".source-char--selected")).toHaveLength(3);
    fireEvent.pointerUp(canvas, { button: 0, pointerId: 1, clientX: 55, clientY: 20 });
    await waitFor(() => expect(within(container).getByText("Add operation")).toBeInTheDocument());
    expect(container).not.toHaveTextContent("overlap");
  });

  it("opens an existing operation directly from its canvas range", () => {
    const operation: EditOperation = { segment_id: "segment", id: "span", kind: "emotion", start: 1, end: 4, params: { type: "happy" } };
    const { container } = render(<EditCanvas {...editorProps} text="hello" operations={[operation]} onOperationsChange={() => {}} />);
    const active = container.querySelector<HTMLElement>('[data-source-index="2"]')!;
    fireEvent.pointerDown(active, { button: 0, pointerId: 1, clientX: 20, clientY: 20 });
    fireEvent.pointerUp(active, { button: 0, pointerId: 1, clientX: 20, clientY: 20 });
    expect(container.querySelector(".operation-popover")).toBeInTheDocument();
    expect(within(container).getByText("Edit operation")).toBeInTheDocument();
    expect(container.querySelector(".source-char--selected")).not.toBeInTheDocument();
  });

  it("uses whole-word boundaries for Latin text and single-character CJK boundaries", () => {
    expect(textBoundaries("hello, 世界!")).toEqual([0, 5, 6, 7, 8, 9, 10]);
    expect(snapBoundary("hello world", 2, -1)).toBe(0);
    expect(snapBoundary("hello world", 2, 1)).toBe(5);
    expect(wordRangeAt("hello 世界", 2)).toEqual({ start: 0, end: 5 });
    expect(wordRangeAt("hello 世界", 6)).toEqual({ start: 6, end: 7 });
  });

  it("double-clicks a complete word instead of a partial letter range", async () => {
    const { container } = render(<EditCanvas {...editorProps} text="hello world" operations={[]} onOperationsChange={() => {}} />);
    const canvas = container.querySelector<HTMLElement>(".edit-canvas")!;
    container.querySelectorAll<HTMLElement>(".source-char").forEach((element, index) => {
      Object.defineProperty(element, "getBoundingClientRect", { configurable: true, value: () => new DOMRect(index * 10, 10, 10, 20) });
    });
    Object.defineProperty(Range.prototype, "getBoundingClientRect", { configurable: true, value: () => new DOMRect(60, 10, 50, 20) });
    fireEvent.doubleClick(canvas, { clientX: 75, clientY: 20 });
    await waitFor(() => expect(container.querySelectorAll(".source-char--selected")).toHaveLength(5));
    expect(within(container).getByText("Add operation")).toBeInTheDocument();
  });

  it("maps emoji selections using Unicode code points", () => {
    const root = document.createElement("div");
    for (const char of Array.from("A😀B")) {
      const span = document.createElement("span");
      span.className = "source-char";
      span.textContent = char;
      root.appendChild(span);
    }
    document.body.appendChild(root);
    const node = root.children[1].firstChild!;
    const range = document.createRange();
    range.setStart(node, 0);
    range.setEnd(node, 2);
    Object.defineProperty(range, "getBoundingClientRect", { value: () => new DOMRect() });
    const selection = window.getSelection()!;
    selection.removeAllRanges();
    selection.addRange(range);
    expect(selectionOffsets(root)).toMatchObject({ start: 1, end: 2 });
    root.remove();
  });

  it("maps blank canvas clicks after the last line to the transcript end", () => {
    const root = document.createElement("div");
    for (const [index, char] of Array.from("A😀B").entries()) {
      const span = document.createElement("span");
      span.className = "source-char";
      span.textContent = char;
      Object.defineProperty(span, "getBoundingClientRect", {
        value: () => new DOMRect(index * 10, 10, 10, 20),
      });
      root.appendChild(span);
    }
    expect(caretOffsetFromPoint(root, 100, 20)).toBe(3);
    expect(caretOffsetFromPoint(root, -10, 20)).toBe(0);
    expect(caretOffsetFromPoint(root, 5, 100)).toBe(3);
  });

  it("builds stable start, middle, end and pointer anchors", () => {
    const root = document.createElement("div");
    Object.defineProperty(root, "getBoundingClientRect", { value: () => new DOMRect(100, 40, 300, 120) });
    for (const [index, char] of Array.from("A中😀").entries()) {
      const span = document.createElement("span");
      span.className = "source-char";
      span.textContent = char;
      Object.defineProperty(span, "getBoundingClientRect", { value: () => new DOMRect(110 + index * 12, 52, 12, 20) });
      root.appendChild(span);
    }
    expect(caretAnchorRect(root, 0)).toMatchObject({ x: 110, y: 52, height: 20 });
    expect(caretAnchorRect(root, 1)).toMatchObject({ x: 122, y: 52, height: 20 });
    expect(caretAnchorRect(root, 3)).toMatchObject({ x: 146, y: 52, height: 20 });
    expect(caretAnchorRect(root, 3, { x: 280, y: 100 })).toMatchObject({ x: 380, y: 140 });
  });

  it("rejects top-left and zero-size native rects and uses a range fallback", () => {
    const root = document.createElement("div");
    Object.defineProperty(root, "getBoundingClientRect", { value: () => new DOMRect(100, 50, 300, 120) });
    expect(isUsableAnchorRect(new DOMRect(0, 0, 20, 20), root.getBoundingClientRect(), true)).toBe(false);
    expect(isUsableAnchorRect(new DOMRect(120, 70, 0, 20), root.getBoundingClientRect(), true)).toBe(false);
    const fallback = new DOMRect(130, 80, 40, 20);
    const range = { getBoundingClientRect: () => new DOMRect(), getClientRects: () => [] } as unknown as Range;
    expect(rangeAnchorRect(range, root, fallback)).toBe(fallback);
  });

  it("renders a linked segment-local highlight", () => {
    const operation: EditOperation = { segment_id: "segment", id: "one", kind: "delete", start: 0, end: 1, params: {} };
    const { container } = render(<EditCanvas {...editorProps} operationColors={{ one: "#555" }} text="hello" operations={[operation]} onOperationsChange={() => {}} />);
    expect(container.querySelector(".source-char--delete")).toHaveTextContent("h");
    expect(container.querySelector(".operation-list")).not.toBeInTheDocument();
  });

  it("creates a code-point range with keyboard selection", async () => {
    const { container } = render(<EditCanvas {...editorProps} text="A😀BC" operations={[]} onOperationsChange={() => {}} />);
    const canvas = container.querySelector<HTMLElement>(".edit-canvas")!;
    Object.defineProperty(Range.prototype, "getBoundingClientRect", {
      configurable: true,
      value: () => new DOMRect(20, 20, 30, 18),
    });
    canvas.focus();
    fireEvent.keyDown(canvas, { key: "Home" });
    fireEvent.keyDown(canvas, { key: "ArrowRight", shiftKey: true });
    fireEvent.keyDown(canvas, { key: "ArrowRight", shiftKey: true });
    fireEvent.keyDown(canvas, { key: "ArrowRight", shiftKey: true });
    await waitFor(() => expect(within(container).getByText("Add operation")).toBeInTheDocument());
    expect(screen.queryByText("0–3")).not.toBeInTheDocument();
  });

  it("keeps the selected transcript highlighted while replacement input has focus", async () => {
    const { container } = render(<EditCanvas {...editorProps} text="A😀BC" operations={[]} onOperationsChange={() => {}} />);
    const canvas = container.querySelector<HTMLElement>(".edit-canvas")!;
    Object.defineProperty(Range.prototype, "getBoundingClientRect", {
      configurable: true,
      value: () => new DOMRect(20, 20, 30, 18),
    });
    canvas.focus();
    fireEvent.keyDown(canvas, { key: "Home" });
    fireEvent.keyDown(canvas, { key: "ArrowRight", shiftKey: true });
    fireEvent.keyDown(canvas, { key: "ArrowRight", shiftKey: true });
    await waitFor(() => expect(within(container).getByText("Add operation")).toBeInTheDocument());
    fireEvent.click(within(container).getByRole("button", { name: "replace" }));
    const input = await within(container).findByPlaceholderText("Replacement");
    fireEvent.focus(input);
    window.getSelection()?.removeAllRanges();
    expect(container.querySelectorAll(".source-char--selected")).toHaveLength(2);
  });

  it("renders readable text edits and semantic markers", () => {
    const operations: EditOperation[] = [
      { segment_id: "segment", id: "replace", kind: "replace", start: 0, end: 2, params: { target: "hi" } },
      { segment_id: "segment", id: "insert", kind: "insert", start: 2, end: 2, params: { text: " there" } },
      { segment_id: "segment", id: "pitch", kind: "pitch", start: 3, end: 5, params: { semitones: -4 } },
    ];
    const colors = { replace: "#a94e4a", insert: "#a94e4a", pitch: "#39769d" };
    const { container } = render(<EditCanvas {...editorProps} operationColors={colors} text="hello" operations={operations} onOperationsChange={() => {}} />);
    expect(container.querySelectorAll(".source-char--replace")).toHaveLength(2);
    expect(container.querySelector(".replacement-mark")).toHaveTextContent("hi");
    expect(container.querySelector(".point-mark--insert")).toHaveTextContent("there");
    expect(container.querySelector(".semantic-mark")).toHaveTextContent("🎵↓");
    expect(operationGlyph(operations[2])).toBe("🎵↓");
    expect(operationGlyph({ segment_id: "segment", id: "rate", kind: "rate", start: 0, end: 1, params: { factor: 1.8 } })).toBe("⏩");
    expect(operationGlyph({ segment_id: "segment", id: "rate-slow", kind: "rate", start: 0, end: 1, params: { factor: 0.55 } })).toBe("⏪");
  });

  it("shows a back control only after choosing an operation type", async () => {
    const { container } = render(<EditCanvas {...editorProps} text="hello world" operations={[]} onOperationsChange={() => {}} />);
    const canvas = container.querySelector<HTMLElement>(".edit-canvas")!;
    Object.defineProperty(Range.prototype, "getBoundingClientRect", { configurable: true, value: () => new DOMRect(20, 20, 30, 18) });
    canvas.focus();
    fireEvent.keyDown(canvas, { key: "Home" });
    fireEvent.keyDown(canvas, { key: "ArrowRight", shiftKey: true });
    await waitFor(() => expect(within(container).getByText("Add operation")).toBeInTheDocument());
    expect(within(container).queryByRole("button", { name: "Back to operation types" })).not.toBeInTheDocument();
    fireEvent.click(within(container).getByRole("button", { name: "replace" }));
    expect(within(container).getByRole("button", { name: "Back to operation types" })).toBeInTheDocument();
  });

  it("keeps the selected transcript highlighted while emotion description has focus", async () => {
    const { container } = render(<EditCanvas {...editorProps} text="emotion" operations={[]} onOperationsChange={() => {}} />);
    const canvas = container.querySelector<HTMLElement>(".edit-canvas")!;
    Object.defineProperty(Range.prototype, "getBoundingClientRect", {
      configurable: true,
      value: () => new DOMRect(20, 20, 30, 18),
    });
    canvas.focus();
    fireEvent.keyDown(canvas, { key: "Home" });
    fireEvent.keyDown(canvas, { key: "ArrowRight", shiftKey: true });
    fireEvent.keyDown(canvas, { key: "ArrowRight", shiftKey: true });
    await waitFor(() => expect(within(container).getByText("Add operation")).toBeInTheDocument());
    fireEvent.click(within(container).getByRole("button", { name: "emotion" }));
    const mode = container.querySelector<HTMLSelectElement>(".operation-form select")!;
    fireEvent.change(mode, { target: { value: "description" } });
    const input = await within(container).findByPlaceholderText("Emotion description");
    fireEvent.focus(input);
    window.getSelection()?.removeAllRanges();
    expect(container.querySelectorAll(".source-char--selected")).toHaveLength(7);
  });

  it("saves typed emotion with benchmark-compatible level 2", async () => {
    const change = vi.fn();
    const { container } = render(<EditCanvas {...editorProps} text="emotion" operations={[]} onOperationsChange={change} />);
    const canvas = container.querySelector<HTMLElement>(".edit-canvas")!;
    Object.defineProperty(Range.prototype, "getBoundingClientRect", {
      configurable: true,
      value: () => new DOMRect(20, 20, 30, 18),
    });
    canvas.focus();
    fireEvent.keyDown(canvas, { key: "Home" });
    fireEvent.keyDown(canvas, { key: "ArrowRight", shiftKey: true });
    await waitFor(() => expect(within(container).getByText("Add operation")).toBeInTheDocument());
    fireEvent.click(within(container).getByRole("button", { name: "emotion" }));
    fireEvent.click(within(container).getByRole("button", { name: "Add" }));
    expect(change).toHaveBeenCalledWith([
      expect.objectContaining({
        kind: "emotion",
        params: { type: "happy", level: 2 },
      }),
    ]);
  });

  it("rejects unsupported markup only when a text operation is submitted", async () => {
    const change = vi.fn();
    const { container } = render(<EditCanvas {...editorProps} text="source" operations={[]} onOperationsChange={change} />);
    const canvas = container.querySelector<HTMLElement>(".edit-canvas")!;
    Object.defineProperty(Range.prototype, "getBoundingClientRect", {
      configurable: true,
      value: () => new DOMRect(20, 20, 30, 18),
    });
    canvas.focus();
    fireEvent.keyDown(canvas, { key: "Home" });
    fireEvent.keyDown(canvas, { key: "ArrowRight", shiftKey: true });
    await waitFor(() => expect(within(container).getByText("Add operation")).toBeInTheDocument());
    fireEvent.click(within(container).getByRole("button", { name: "replace" }));
    fireEvent.change(within(container).getByPlaceholderText("Replacement"), { target: { value: "<ins>bad</ins>" } });
    fireEvent.click(within(container).getByRole("button", { name: "Add" }));
    expect(within(container).getByText("Replacement text contains unsupported markup.")).toBeInTheDocument();
    expect(change).not.toHaveBeenCalled();
  });

  it("draws a caret at Unicode boundaries and hides it for range selection", async () => {
    const { container } = render(<EditCanvas {...editorProps} text="A😀B" operations={[]} onOperationsChange={() => {}} />);
    const canvas = container.querySelector<HTMLElement>(".edit-canvas")!;
    Object.defineProperty(Range.prototype, "getBoundingClientRect", {
      configurable: true,
      value: () => new DOMRect(20, 20, 1, 18),
    });
    canvas.focus();
    fireEvent.keyDown(canvas, { key: "End" });
    await waitFor(() => expect(container.querySelector(".canvas-caret")).toBeInTheDocument());
    expect(container.querySelector(".canvas-caret")?.previousSibling).not.toBeNull();
    fireEvent.keyDown(canvas, { key: "ArrowLeft", shiftKey: true });
    await waitFor(() => expect(container.querySelector(".canvas-caret")).not.toBeInTheDocument());
  });

  it("replaces the canvas with an inline editor and hides operations", () => {
    const operation: EditOperation = { segment_id: "segment", id: "one", kind: "delete", start: 0, end: 1, params: {} };
    const { container } = render(<EditCanvas {...editorProps} editing draft="draft" text="source" operations={[operation]} onOperationsChange={() => {}} />);
    expect(screen.getByRole("textbox", { name: "Source Transcript" })).toHaveValue("draft");
    expect(container.querySelector(".edit-canvas")).not.toBeInTheDocument();
    expect(container.querySelector(".operation-list")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Save/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Cancel/ })).toBeInTheDocument();
    expect(screen.queryByText("Cancel")).not.toBeInTheDocument();
    expect(screen.queryByText("Save")).not.toBeInTheDocument();
  });

  it("places an icon-only transcript control beside the title and references it in the hint", () => {
    const { container } = render(<EditCanvas {...editorProps} hintVisible text="source" operations={[]} onOperationsChange={() => {}} />);
    const button = within(container).getByRole("button", { name: "Edit Source Transcript" });
    expect(button).toHaveAttribute("title", "Edit Source Transcript");
    expect(button.previousElementSibling).toHaveTextContent("Source Transcript");
    expect(button).not.toHaveTextContent("Edit transcript");
    expect(container.querySelector(".transcript-hint__pencil")).toBeInTheDocument();
  });

  it("does not open edit operations while the source audio is unavailable", async () => {
    const { container } = render(<EditCanvas {...editorProps} text="source" operations={[]} disabled transcriptDisabled onOperationsChange={() => {}} />);
    const canvas = container.querySelector<HTMLElement>(".edit-canvas")!;
    fireEvent.keyDown(canvas, { key: "Home" });
    fireEvent.keyDown(canvas, { key: "ArrowRight", shiftKey: true });
    fireEvent.contextMenu(canvas);
    expect(canvas).toHaveAttribute("aria-disabled", "true");
    expect(container.querySelector(".operation-popover")).not.toBeInTheDocument();
    expect(container.querySelector<HTMLButtonElement>('[aria-label="Edit Source Transcript"]')).toBeDisabled();
  });
});
