import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { autoUpdate, flip, offset, shift, useFloating } from "@floating-ui/react";
import { ArrowLeft, Check, Info, Pencil, Save, Trash2, X } from "lucide-react";
import { createId } from "../browser";
import type { EditOperation, OperationKind } from "../types";

export type OperationCategory = "text" | "emotion" | "pitch" | "rate" | "pause";

export const OPERATION_CATEGORY_STYLES: Record<OperationCategory, { color: string; background: string }> = {
  text: { color: "#a94e4a", background: "#f7dfdd" },
  emotion: { color: "#7552a4", background: "#eadff5" },
  pitch: { color: "#39769d", background: "#dcecf7" },
  rate: { color: "#3f7951", background: "#dff0e3" },
  pause: { color: "#85613d", background: "#eee3d5" },
};
const SPAN_KINDS: OperationKind[] = ["replace", "delete", "emotion", "pitch", "rate"];
const POINT_KINDS: OperationKind[] = ["insert", "pause"];
export const PITCH_LEVELS = [
  { level: -3, semitones: -9 },
  { level: -2, semitones: -7 },
  { level: -1, semitones: -5 },
  { level: 1, semitones: 5 },
  { level: 2, semitones: 7 },
  { level: 3, semitones: 9 },
];
export const EMOTION_TYPES = [
  "happy", "angry", "sad", "afraid", "melancholic", "surprised", "calm",
];

export function pitchLevelLabel(semitones: number): string {
  const nearest = PITCH_LEVELS.reduce((best, item) => (
    Math.abs(item.semitones - semitones) < Math.abs(best.semitones - semitones)
      ? item
      : best
  ));
  return `${nearest.semitones > 0 ? "Raise" : "Lower"} · level ${Math.abs(nearest.level)}`;
}
export const RATE_VALUES = [
  ...Array.from({ length: 45 }, (_, i) => Number((0.55 + i * 0.01).toFixed(2))),
  ...Array.from({ length: 95 }, (_, i) => Number((1.01 + i * 0.01).toFixed(2))),
];

interface SelectionState {
  start: number;
  end: number;
}

interface PointerAnchor {
  x: number;
  y: number;
}

interface PointerSelection {
  anchor: number;
  pointerId: number;
}

interface Props {
  segmentId: string;
  text: string;
  operations: EditOperation[];
  operationColors: Record<string, string>;
  onOperationsChange: (operations: EditOperation[]) => void;
  editing: boolean;
  draft: string;
  hintVisible: boolean;
  onBeginEdit: () => void;
  onDraftChange: (value: string) => void;
  onSaveEdit: () => void;
  onCancelEdit: () => void;
  onDismissHint: () => void;
  requestedEditId?: string | null;
  onRequestedEditHandled?: () => void;
  hoveredOperationId?: string | null;
  onHoverOperation?: (id: string | null) => void;
  disabled?: boolean;
  transcriptDisabled?: boolean;
  toolbarAction?: ReactNode;
}

export function TranscriptHint({ onDismiss }: { onDismiss: () => void }) {
  return <div className="transcript-hint transcript-hint--global"><Info size={14} /><span>Click or select a transcript position to add an edit. Use <Pencil className="transcript-hint__pencil" size={13} aria-label="Edit Source Transcript icon" /> to edit the transcript paired with each Source Audio segment.</span><button className="icon-button" aria-label="Dismiss Source Transcript hint" onClick={onDismiss}><X size={13} /></button></div>;
}

export function operationCategory(kind: OperationKind): OperationCategory {
  if (kind === "replace" || kind === "insert" || kind === "delete") return "text";
  return kind;
}

export function operationStyle(kind: OperationKind) {
  return OPERATION_CATEGORY_STYLES[operationCategory(kind)];
}

export function operationGlyph(operation: EditOperation): string | null {
  if (operation.kind === "emotion") {
    if (operation.params.desc) return "T";
    return ({
      happy: "😊", angry: "😠", sad: "😢", afraid: "😨", disgusted: "🤢",
      melancholic: "😔", surprised: "😮", calm: "😌",
    } as Record<string, string>)[String(operation.params.type)] || "😊";
  }
  if (operation.kind === "pitch") return Number(operation.params.semitones) > 0 ? "🎵↑" : "🎵↓";
  if (operation.kind === "rate") return Number(operation.params.factor) > 1 ? "⏩" : "⏪";
  if (operation.kind === "pause") return operation.params.act === "red" ? "⏸↓" : "⏸↑";
  return null;
}

const CJK_CHARACTER = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u;
const WORD_CHARACTER = /[\p{L}\p{N}\p{M}'’-]/u;

export function textBoundaries(value: string): number[] {
  const characters = Array.from(value);
  const boundaries = [0];
  for (let index = 1; index < characters.length; index += 1) {
    const left = characters[index - 1];
    const right = characters[index];
    if (CJK_CHARACTER.test(left) || CJK_CHARACTER.test(right) || !WORD_CHARACTER.test(left) || !WORD_CHARACTER.test(right)) {
      boundaries.push(index);
    }
  }
  boundaries.push(characters.length);
  return [...new Set(boundaries)];
}

export function snapBoundary(value: string, position: number, direction: -1 | 0 | 1): number {
  const boundaries = textBoundaries(value);
  if (boundaries.includes(position)) return position;
  const previous = boundaries.filter((item) => item < position).at(-1) ?? 0;
  const next = boundaries.find((item) => item > position) ?? Array.from(value).length;
  if (direction < 0) return previous;
  if (direction > 0) return next;
  return position - previous <= next - position ? previous : next;
}

export function wordRangeAt(value: string, position: number): SelectionState {
  const characters = Array.from(value);
  if (!characters.length) return { start: 0, end: 0 };
  const index = Math.min(Math.max(position, 0), characters.length - 1);
  if (CJK_CHARACTER.test(characters[index]) || !WORD_CHARACTER.test(characters[index])) {
    return { start: index, end: index + 1 };
  }
  const boundaries = textBoundaries(value);
  return {
    start: boundaries.filter((item) => item <= index).at(-1) ?? 0,
    end: boundaries.find((item) => item > index) ?? characters.length,
  };
}

export function selectionOffsets(root: HTMLElement): { start: number; end: number; rect: DOMRect; range: Range } | null {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0) return null;
  const range = selection.getRangeAt(0);
  if (!root.contains(range.startContainer) || !root.contains(range.endContainer)) return null;
  const sourceOffset = (container: Node, offset: number) => {
    const before = document.createRange();
    before.selectNodeContents(root);
    before.setEnd(container, offset);
    const fragment = before.cloneContents();
    return Array.from(fragment.querySelectorAll(".source-char"))
      .reduce((total, element) => total + Array.from(element.textContent || "").length, 0);
  };
  const start = sourceOffset(range.startContainer, range.startOffset);
  const end = sourceOffset(range.endContainer, range.endOffset);
  return { start: Math.min(start, end), end: Math.max(start, end), rect: range.getBoundingClientRect(), range };
}

export function caretOffsetFromPoint(root: HTMLElement, x: number, y: number): number {
  const characters = Array.from(root.querySelectorAll<HTMLElement>(".source-char"));
  if (!characters.length) return 0;
  const boxes = characters.map((element, index) => ({ index, rect: element.getBoundingClientRect() }));
  if (y < boxes[0].rect.top) return 0;
  if (y > boxes.at(-1)!.rect.bottom) return characters.length;
  const sameLine = boxes.filter(({ rect }) => y >= rect.top && y <= rect.bottom);
  const line = sameLine.length ? sameLine : boxes.filter(({ rect }) => {
    const nearest = boxes.reduce((best, item) => {
      const distance = Math.min(Math.abs(y - item.rect.top), Math.abs(y - item.rect.bottom));
      return distance < best.distance ? { distance, top: item.rect.top } : best;
    }, { distance: Number.POSITIVE_INFINITY, top: boxes[0].rect.top });
    return rect.top === nearest.top;
  });
  for (const { index, rect } of line) {
    if (x < rect.left + rect.width / 2) return index;
  }
  return line.at(-1)!.index + 1;
}

function finiteRect(rect: DOMRect) {
  return [rect.left, rect.top, rect.right, rect.bottom, rect.width, rect.height]
    .every(Number.isFinite);
}

export function isUsableAnchorRect(rect: DOMRect, rootRect: DOMRect, range = false) {
  if (!finiteRect(rect) || rect.height <= 0 || (range && rect.width <= 0)) return false;
  return !(rect.left === 0 && rect.top === 0 && (rootRect.left !== 0 || rootRect.top !== 0));
}

export function caretAnchorRect(root: HTMLElement, position: number, pointer?: PointerAnchor): DOMRect {
  const rootRect = root.getBoundingClientRect();
  const characters = Array.from(root.querySelectorAll<HTMLElement>(".source-char"));
  if (pointer) {
    const x = Math.min(rootRect.right, Math.max(rootRect.left, rootRect.left + pointer.x));
    const y = Math.min(rootRect.bottom, Math.max(rootRect.top, rootRect.top + pointer.y));
    return new DOMRect(x, y, 1, 1);
  }
  if (!characters.length) return new DOMRect(rootRect.left, rootRect.top, 1, Math.max(1, rootRect.height));
  const bounded = Math.max(0, Math.min(characters.length, position));
  const character = characters[Math.min(bounded, characters.length - 1)];
  const rect = character.getBoundingClientRect();
  const x = bounded === characters.length ? rect.right : rect.left;
  if (finiteRect(rect) && rect.height > 0) return new DOMRect(x, rect.top, 1, rect.height);
  return new DOMRect(rootRect.left, rootRect.top, 1, Math.max(1, rootRect.height));
}

export function rangeAnchorRect(range: Range, root: HTMLElement, fallback: DOMRect): DOMRect {
  const rootRect = root.getBoundingClientRect();
  const bounding = range.getBoundingClientRect();
  if (isUsableAnchorRect(bounding, rootRect, true)) return bounding;
  const clients = typeof range.getClientRects === "function" ? Array.from(range.getClientRects()) : [];
  const usable = clients.filter((rect) => isUsableAnchorRect(rect as DOMRect, rootRect, true));
  if (usable.length) {
    const left = Math.min(...usable.map((rect) => rect.left));
    const top = Math.min(...usable.map((rect) => rect.top));
    const right = Math.max(...usable.map((rect) => rect.right));
    const bottom = Math.max(...usable.map((rect) => rect.bottom));
    return new DOMRect(left, top, right - left, bottom - top);
  }
  return fallback;
}

export function operationConflicts(candidate: EditOperation, operations: EditOperation[]) {
  const isPoint = POINT_KINDS.includes(candidate.kind);
  return operations.some((current) => {
    if (current.id === candidate.id) return false;
    const currentPoint = POINT_KINDS.includes(current.kind);
    if (isPoint && currentPoint) return candidate.start === current.start;
    if (isPoint) return current.start < candidate.start && candidate.start < current.end;
    if (currentPoint) return candidate.start < current.start && current.start < candidate.end;
    return candidate.start < current.end && current.start < candidate.end;
  });
}

export function positionBlocked(position: number, operations: EditOperation[]) {
  return operations.some((operation) => {
    if (POINT_KINDS.includes(operation.kind)) return operation.start === position;
    return operation.start < position && position < operation.end;
  });
}

export function constrainSelectionFocus(anchor: number, focus: number, operations: EditOperation[]) {
  if (positionBlocked(anchor, operations)) return null;
  if (focus === anchor) return anchor;
  if (focus > anchor) {
    let constrained = focus;
    for (const operation of operations) {
      if (POINT_KINDS.includes(operation.kind)) {
        if (anchor < operation.start && operation.start < constrained) constrained = operation.start;
      } else if (anchor <= operation.start && operation.start < constrained) {
        constrained = operation.start;
      } else if (operation.start < constrained && constrained < operation.end) {
        constrained = operation.start;
      }
    }
    return constrained;
  }
  let constrained = focus;
  for (const operation of operations) {
    if (POINT_KINDS.includes(operation.kind)) {
      if (constrained < operation.start && operation.start < anchor) constrained = operation.start;
    } else if (constrained < operation.end && operation.end <= anchor) {
      constrained = Math.max(constrained, operation.end);
    } else if (operation.start < constrained && constrained < operation.end) {
      constrained = operation.end;
    }
  }
  return constrained;
}

export function selectionConflicts(start: number, end: number, operations: EditOperation[]) {
  if (start === end) return positionBlocked(start, operations);
  return operations.some((operation) => {
    if (POINT_KINDS.includes(operation.kind)) return start < operation.start && operation.start < end;
    return start < operation.end && operation.start < end;
  });
}

function containsUnsupportedMarkup(value: string) {
  return /<\/?(?:del|ins|sub)(?:\s+[^>]*)?>/.test(value);
}

function defaultParams(kind: OperationKind): Record<string, string | number> {
  if (kind === "replace") return { target: "" };
  if (kind === "insert") return { text: "" };
  if (kind === "emotion") return { mode: "type", type: "happy", level: 2 };
  if (kind === "pitch") return { semitones: 5 };
  if (kind === "rate") return { factor: 0.85 };
  if (kind === "pause") return { act: "ins", level: 2 };
  return {};
}

export function EditCanvas({ segmentId, text, operations, operationColors, onOperationsChange, editing, draft, hintVisible, onBeginEdit, onDraftChange, onSaveEdit, onCancelEdit, onDismissHint, requestedEditId = null, onRequestedEditHandled, hoveredOperationId = null, onHoverOperation, disabled = false, transcriptDisabled = false, toolbarAction }: Props) {
  const root = useRef<HTMLDivElement>(null);
  const [selection, setSelection] = useState<SelectionState | null>(null);
  const [kind, setKind] = useState<OperationKind | null>(null);
  const [params, setParams] = useState<Record<string, string | number>>({});
  const [editingId, setEditingId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [caret, setCaret] = useState<number | null>(null);
  const [anchorReady, setAnchorReady] = useState(false);
  const chars = useMemo(() => Array.from(text), [text]);
  const keyboardCaret = useRef(0);
  const keyboardAnchor = useRef<number | null>(null);
  const anchorRange = useRef<Range | null>(null);
  const pointerSelection = useRef<PointerSelection | null>(null);
  const floating = useFloating({
    open: selection !== null && anchorReady,
    placement: "bottom",
    strategy: "fixed",
    middleware: [offset(8), flip({ padding: 12 }), shift({ padding: 12 })],
    whileElementsMounted: autoUpdate,
  });

  useEffect(() => {
    setSelection(null);
    setAnchorReady(false);
    setCaret(null);
    keyboardCaret.current = 0;
    keyboardAnchor.current = null;
    pointerSelection.current = null;
    window.getSelection()?.removeAllRanges();
  }, [text, editing]);

  const capture = (preserveKeyboardAnchor = false, focusedOffset?: number, pointer?: { x: number; y: number }) => {
    if (disabled) return;
    setAnchorReady(false);
    requestAnimationFrame(() => {
      if (!root.current) return;
      const value = selectionOffsets(root.current);
      if (!value) return;
      if (selectionConflicts(value.start, value.end, operations)) {
        setSelection(null);
        setCaret(null);
        window.getSelection()?.removeAllRanges();
        return;
      }
      const canvas = root.current;
      anchorRange.current = value.range.cloneRange();
      const rootRect = canvas.getBoundingClientRect();
      const relativePointer = pointer ? { x: pointer.x - rootRect.left, y: pointer.y - rootRect.top } : undefined;
      const fallback = value.start === value.end
        ? caretAnchorRect(canvas, value.start, relativePointer)
        : (() => {
            const start = caretAnchorRect(canvas, value.start);
            const end = caretAnchorRect(canvas, value.end);
            return new DOMRect(start.left, Math.min(start.top, end.top), Math.max(1, end.right - start.left), Math.max(start.height, end.height));
          })();
      floating.refs.setPositionReference({
        contextElement: canvas,
        getBoundingClientRect: () => {
          if (value.start === value.end) return caretAnchorRect(canvas, value.start, relativePointer);
          return anchorRange.current ? rangeAnchorRect(anchorRange.current, canvas, fallback) : fallback;
        },
      });
      setSelection({
        start: value.start,
        end: value.end,
      });
      setCaret(value.start === value.end ? value.start : null);
      keyboardCaret.current = focusedOffset ?? value.end;
      if (!preserveKeyboardAnchor) keyboardAnchor.current = null;
      setKind(null);
      setEditingId(null);
      setError("");
      setAnchorReady(true);
    });
  };

  const setBrowserSelection = (start: number, end: number) => {
    if (!root.current || !chars.length) return;
    const boundary = (position: number): [Node, number] => {
      if (position >= chars.length) {
        const last = root.current!.querySelector<HTMLElement>(`[data-source-index="${chars.length - 1}"]`)!;
        const node = last.firstChild!;
        return [node, node.textContent?.length || 0];
      }
      const element = root.current!.querySelector<HTMLElement>(`[data-source-index="${position}"]`)!;
      return [element.firstChild!, 0];
    };
    const range = document.createRange();
    const [startNode, startOffset] = boundary(Math.min(start, end));
    const [endNode, endOffset] = boundary(Math.max(start, end));
    range.setStart(startNode, startOffset);
    range.setEnd(endNode, endOffset);
    const browserSelection = window.getSelection();
    browserSelection?.removeAllRanges();
    browserSelection?.addRange(range);
  };

  const keyboardSelect = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return;
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const current = keyboardCaret.current;
    const boundaries = textBoundaries(text);
    const previousBoundary = boundaries.filter((item) => item < current).at(-1) ?? 0;
    const nextBoundary = boundaries.find((item) => item > current) ?? chars.length;
    const requested = event.key === "Home"
      ? 0
      : event.key === "End"
        ? chars.length
        : event.key === "ArrowRight" ? nextBoundary : previousBoundary;
    if (event.shiftKey) {
      if (keyboardAnchor.current == null) keyboardAnchor.current = current;
      const next = constrainSelectionFocus(keyboardAnchor.current, requested, operations);
      if (next == null) return;
      keyboardCaret.current = next;
      setBrowserSelection(keyboardAnchor.current, next);
      capture(true, next);
    } else {
      const next = constrainSelectionFocus(current, requested, operations);
      if (next == null || positionBlocked(next, operations)) return;
      keyboardAnchor.current = null;
      keyboardCaret.current = next;
      setBrowserSelection(next, next);
      capture(false, next);
    }
  };

  const updatePointerSelection = (clientX: number, clientY: number) => {
    if (!root.current || !pointerSelection.current) return null;
    const raw = caretOffsetFromPoint(root.current, clientX, clientY);
    const direction = raw === pointerSelection.current.anchor ? 0 : raw > pointerSelection.current.anchor ? 1 : -1;
    const requested = snapBoundary(text, raw, direction);
    const focus = constrainSelectionFocus(pointerSelection.current.anchor, requested, operations);
    if (focus == null) return null;
    setBrowserSelection(pointerSelection.current.anchor, focus);
    const start = Math.min(pointerSelection.current.anchor, focus);
    const end = Math.max(pointerSelection.current.anchor, focus);
    setSelection({ start, end });
    setCaret(start === end ? start : null);
    keyboardCaret.current = focus;
    return focus;
  };

  const beginPointerSelection = (event: React.PointerEvent<HTMLDivElement>) => {
    if (disabled || event.button !== 0 || !root.current) return;
    if (activateOperation(event)) return;
    event.preventDefault();
    const anchor = snapBoundary(text, caretOffsetFromPoint(root.current, event.clientX, event.clientY), 0);
    setSelection(null);
    setAnchorReady(false);
    setKind(null);
    setEditingId(null);
    if (positionBlocked(anchor, operations)) {
      pointerSelection.current = null;
      setCaret(null);
      window.getSelection()?.removeAllRanges();
      return;
    }
    pointerSelection.current = { anchor, pointerId: event.pointerId };
    event.currentTarget.setPointerCapture?.(event.pointerId);
    setBrowserSelection(anchor, anchor);
    setSelection({ start: anchor, end: anchor });
    setCaret(anchor);
  };

  const movePointerSelection = (event: React.PointerEvent<HTMLDivElement>) => {
    if (pointerSelection.current?.pointerId !== event.pointerId) return;
    event.preventDefault();
    updatePointerSelection(event.clientX, event.clientY);
  };

  const finishPointerSelection = (event: React.PointerEvent<HTMLDivElement>) => {
    if (pointerSelection.current?.pointerId !== event.pointerId) return;
    event.preventDefault();
    const focus = updatePointerSelection(event.clientX, event.clientY);
    pointerSelection.current = null;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
    if (focus == null) return;
    capture(false, focus, { x: event.clientX, y: event.clientY });
  };

  const selectWord = (event: React.MouseEvent<HTMLDivElement>) => {
    if (disabled || !root.current) return;
    event.preventDefault();
    const range = wordRangeAt(text, caretOffsetFromPoint(root.current, event.clientX, event.clientY));
    if (selectionConflicts(range.start, range.end, operations)) return;
    setBrowserSelection(range.start, range.end);
    keyboardCaret.current = range.end;
    capture(false, range.end, { x: event.clientX, y: event.clientY });
  };

  const choose = (value: OperationKind) => {
    setKind(value);
    setParams(defaultParams(value));
    setError("");
  };

  const save = () => {
    if (!selection || !kind) return;
    const isPoint = POINT_KINDS.includes(kind);
    const candidate: EditOperation = {
      id: editingId || createId(),
      segment_id: segmentId,
      kind,
      start: selection.start,
      end: isPoint ? selection.start : selection.end,
      params,
    };
    if (!isPoint && candidate.start === candidate.end) {
      setError("Select a text range.");
      return;
    }
    if ((kind === "replace" && !String(params.target || "")) || (kind === "insert" && !String(params.text || ""))) {
      setError("Text is required.");
      return;
    }
    if (kind === "replace" && containsUnsupportedMarkup(String(params.target || ""))) {
      setError("Replacement text contains unsupported markup.");
      return;
    }
    if (kind === "insert" && containsUnsupportedMarkup(String(params.text || ""))) {
      setError("Inserted text contains unsupported markup.");
      return;
    }
    if (kind === "emotion" && params.mode === "description") {
      if (!String(params.desc || "").trim()) {
        setError("Description is required.");
        return;
      }
      candidate.params = { desc: params.desc };
    } else if (kind === "emotion") {
      const { mode, ...emotion } = params;
      candidate.params = emotion;
    }
    if (operationConflicts(candidate, operations)) {
      return;
    }
    const next = editingId
      ? operations.map((operation) => operation.id === editingId ? candidate : operation)
      : [...operations, candidate];
    onOperationsChange(next);
    setSelection(null);
    setKind(null);
    setEditingId(null);
    window.getSelection()?.removeAllRanges();
  };

  const editOperation = (operation: EditOperation, anchor: HTMLElement) => {
    anchorRange.current = null;
    setAnchorReady(false);
    floating.refs.setReference(anchor);
    setSelection({
      start: operation.start,
      end: operation.end,
    });
    setKind(operation.kind);
    if (operation.kind === "emotion") {
      if (operation.params.desc) setParams({ mode: "description", desc: operation.params.desc });
      else setParams({ type: operation.params.type, level: operation.params.level ?? 2, mode: "type" });
    } else setParams({ ...operation.params });
    setEditingId(operation.id);
    setAnchorReady(true);
  };

  const activateOperation = (event: React.PointerEvent<HTMLDivElement>) => {
    const anchor = (event.target as Element).closest<HTMLElement>("[data-operation-id]");
    const operation = operations.find((item) => item.id === anchor?.dataset.operationId);
    if (!anchor || !operation) return false;
    event.preventDefault();
    event.stopPropagation();
    pointerSelection.current = null;
    window.getSelection()?.removeAllRanges();
    editOperation(operation, anchor);
    return true;
  };

  useEffect(() => {
    if (!requestedEditId || editing) return;
    const operation = operations.find((item) => item.id === requestedEditId);
    if (!operation || !root.current) return;
    const anchor = Array.from(
      root.current.querySelectorAll<HTMLElement>("[data-operation-id]"),
    ).find((element) => element.dataset.operationId === operation.id);
    if (!anchor) return;
    anchor.scrollIntoView?.({ block: "nearest", inline: "nearest" });
    editOperation(operation, anchor);
    onRequestedEditHandled?.();
  }, [editing, operations, requestedEditId]);

  const pointAt = new Map<number, EditOperation>();
  for (const operation of operations) {
    if (POINT_KINDS.includes(operation.kind)) pointAt.set(operation.start, operation);
  }

  return (
    <div className="edit-canvas-shell">
      <div className="edit-canvas-toolbar">
        <span>Source Transcript</span>
        {editing ? <div className="transcript-edit-actions"><button className="icon-button transcript-tool-button" aria-label="Cancel Source Transcript edit" title="Cancel" onClick={onCancelEdit}><X size={14} /></button><button className="icon-button transcript-tool-button transcript-tool-button--save" aria-label="Save Source Transcript" title="Save" onClick={onSaveEdit}><Save size={13} /></button></div> : <button className="icon-button transcript-tool-button" aria-label="Edit Source Transcript" title="Edit Source Transcript" disabled={transcriptDisabled} onClick={onBeginEdit}><Pencil size={14} /></button>}
        {!editing && toolbarAction}
      </div>
      {!editing && hintVisible && !transcriptDisabled && <TranscriptHint onDismiss={onDismissHint} />}
      {editing ? <textarea
        autoFocus
        className="source-transcript-editor"
        aria-label="Source Transcript"
        value={draft}
        placeholder="Enter the transcript paired with Source Audio."
        onChange={(event) => onDraftChange(event.target.value)}
        onPaste={(event) => {
          event.preventDefault();
          const plain = event.clipboardData.getData("text/plain");
          const target = event.currentTarget;
          const start = target.selectionStart;
          const end = target.selectionEnd;
          onDraftChange(draft.slice(0, start) + plain + draft.slice(end));
        }}
      /> : <>
      <div
        ref={root}
        className={`edit-canvas ${disabled ? "edit-canvas--disabled" : ""}`}
        role="textbox"
        aria-readonly="true"
        aria-disabled={disabled}
        tabIndex={0}
        onPointerDown={beginPointerSelection}
        onPointerMove={movePointerSelection}
        onPointerUp={finishPointerSelection}
        onPointerCancel={() => { pointerSelection.current = null; setSelection(null); setAnchorReady(false); }}
        onKeyDown={keyboardSelect}
        onDoubleClick={selectWord}
        onMouseMove={(event) => {
          const operationId = (event.target as Element).closest<HTMLElement>("[data-operation-id]")?.dataset.operationId;
          onHoverOperation?.(operationId || null);
        }}
        onMouseLeave={() => onHoverOperation?.(null)}
        onContextMenu={(event) => { event.preventDefault(); capture(); }}
      >
        {chars.map((char, index) => {
          const active = operations.find((operation) => SPAN_KINDS.includes(operation.kind) && operation.start <= index && index < operation.end);
          const selected = editingId === null && selection !== null && selection.start < selection.end && selection.start <= index && index < selection.end;
          const point = pointAt.get(index);
          const activeStyle = active ? operationStyle(active.kind) : null;
          const pointStyle = point ? operationStyle(point.kind) : null;
          const marker = active && index === active.end - 1 ? operationGlyph(active) : null;
          return (
            <span key={index}>
              {caret === index && <span className="canvas-caret" aria-hidden="true" />}
              {point && <span data-operation-id={point.id} className={`point-mark point-mark--${point.kind} ${hoveredOperationId === point.id ? "operation-mark--linked" : ""}`} style={{ "--op-color": operationColors[point.id] || pointStyle?.color, "--op-bg": pointStyle?.background } as React.CSSProperties} contentEditable={false}>{point.kind === "insert" ? String(point.params.text || "") : operationGlyph(point)}</span>}
              <span
                data-source-index={index}
                data-operation-id={active?.id}
                className={`source-char ${active ? `source-char--active source-char--${active.kind}` : ""} ${selected ? "source-char--selected" : ""} ${active && hoveredOperationId === active.id ? "operation-mark--linked" : ""}`}
                style={active ? { "--op-color": operationColors[active.id] || activeStyle?.color, "--op-bg": activeStyle?.background } as React.CSSProperties : undefined}
              >{char}</span>
              {active?.kind === "replace" && index === active.end - 1 && (
                <span data-operation-id={active.id} className={`replacement-mark ${hoveredOperationId === active.id ? "operation-mark--linked" : ""}`} style={{ "--op-color": operationColors[active.id] || activeStyle?.color, "--op-bg": activeStyle?.background } as React.CSSProperties} contentEditable={false}>{active.params.target}</span>
              )}
              {marker && <span data-operation-id={active!.id} className={`semantic-mark ${hoveredOperationId === active!.id ? "operation-mark--linked" : ""}`} style={{ "--op-color": operationColors[active!.id] || activeStyle?.color, "--op-bg": activeStyle?.background } as React.CSSProperties} contentEditable={false}>{marker}</span>}
            </span>
          );
        })}
        {caret === chars.length && <span className="canvas-caret" aria-hidden="true" />}
        {pointAt.get(chars.length) && (() => {
          const point = pointAt.get(chars.length)!;
          const style = operationStyle(point.kind);
          return <span data-operation-id={point.id} className={`point-mark point-mark--${point.kind} ${hoveredOperationId === point.id ? "operation-mark--linked" : ""}`} style={{ "--op-color": operationColors[point.id] || style.color, "--op-bg": style.background } as React.CSSProperties}>{point.kind === "insert" ? String(point.params.text || "") : operationGlyph(point)}</span>;
        })()}
        {!text && <span className="canvas-placeholder">{transcriptDisabled ? "Choose or upload Source Audio first." : "Enter the transcript paired with Source Audio."}</span>}
      </div>

      {selection && anchorReady && (
        <div
          ref={floating.refs.setFloating}
          className="operation-popover"
          style={{ ...floating.floatingStyles, visibility: floating.isPositioned ? "visible" : "hidden" }}
        >
          <div className="popover-head">
            <span>{editingId ? "Edit operation" : "Add operation"}</span>
            <button className="icon-button" onClick={() => setSelection(null)}><X size={14} /></button>
          </div>
          {!kind ? (
            <div className="operation-grid">
              {(selection.start === selection.end ? POINT_KINDS : SPAN_KINDS).map((item) => (
                <button
                  key={item}
                  className={`operation-choice operation-choice--${operationCategory(item)}`}
                  style={{ "--op-color": operationStyle(item).color, "--op-bg": operationStyle(item).background } as React.CSSProperties}
                  onClick={() => choose(item)}
                >
                  {item === "emotion" && <span aria-hidden="true">😊</span>}
                  {item === "pitch" && <span aria-hidden="true">🎵</span>}
                  {item === "rate" && <span aria-hidden="true">⏱</span>}
                  {item === "pause" && <span aria-hidden="true">⏸</span>}
                  {item}
                </button>
              ))}
            </div>
          ) : (
            <div className="operation-form">
              <div className="form-title"><button className="icon-button operation-back" aria-label="Back to operation types" onClick={() => { setKind(null); setError(""); }}><ArrowLeft size={14} /></button><span>{kind}</span></div>
              {(kind === "replace" || kind === "insert") && (
                <input
                  autoFocus
                  value={String(params[kind === "replace" ? "target" : "text"] || "")}
                  placeholder={kind === "replace" ? "Replacement" : "Text to insert"}
                  onChange={(event) => setParams({ [kind === "replace" ? "target" : "text"]: event.target.value })}
                />
              )}
              {kind === "emotion" && (
                <>
                  <div className="control-row">
                    <select value={String(params.mode || "type")} onChange={(event) => {
                      const mode = event.target.value;
                      setParams(mode === "description" ? { mode, desc: "" } : { mode, type: "happy", level: 2 });
                    }}>
                      <option value="type">Emotion</option><option value="description">Description</option>
                    </select>
                    {params.mode !== "description" && (
                      <select value={String(params.type)} onChange={(event) => setParams({ ...params, type: event.target.value })}>
                        {EMOTION_TYPES.map((value) => <option key={value}>{value}</option>)}
                      </select>
                    )}
                  </div>
                  {params.mode === "description" && <input value={String(params.desc || "")} placeholder="Emotion description" onChange={(event) => setParams({ ...params, desc: event.target.value })} />}
                </>
              )}
              {kind === "pitch" && <Slider label="Pitch level" value={Math.max(0, PITCH_LEVELS.findIndex((item) => item.semitones === Number(params.semitones)))} min={0} max={PITCH_LEVELS.length - 1} display={pitchLevelLabel(Number(params.semitones))} onChange={(index) => setParams({ semitones: PITCH_LEVELS[index].semitones })} />}
              {kind === "rate" && <Slider label="Rate" value={Math.max(0, RATE_VALUES.indexOf(Number(params.factor)))} min={0} max={RATE_VALUES.length - 1} display={`${Number(params.factor).toFixed(2)}×`} onChange={(index) => setParams({ factor: RATE_VALUES[index] })} />}
              {kind === "pause" && (
                <>
                  <div className="segmented">
                    <button className={params.act === "ins" ? "active" : ""} onClick={() => setParams({ ...params, act: "ins" })}>Insert</button>
                    <button className={params.act === "red" ? "active" : ""} onClick={() => setParams({ ...params, act: "red" })}>Reduce</button>
                  </div>
                  <Slider label="Level" value={Number(params.level)} min={1} max={3} display={['','Short','Medium','Long'][Number(params.level)]} onChange={(value) => setParams({ ...params, level: value })} />
                </>
              )}
              {error && <div className="field-error">{error}</div>}
              <div className="popover-actions">
                {editingId && <button className="danger-button" onClick={() => { onOperationsChange(operations.filter((item) => item.id !== editingId)); setSelection(null); }}><Trash2 size={13} /> Delete</button>}
                <button className="primary-small" onClick={save}><Check size={13} /> {editingId ? "Update" : "Add"}</button>
              </div>
            </div>
          )}
        </div>
      )}
      </>}
    </div>
  );
}

function Slider({ label, value, min, max, display, onChange }: { label: string; value: number; min: number; max: number; display?: string; onChange: (value: number) => void }) {
  return (
    <label className="slider-control">
      <span>{label}<output>{display || value}</output></span>
      <input type="range" min={min} max={max} step={1} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}
