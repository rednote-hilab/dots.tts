import { X } from "lucide-react";
import type { EditOperation } from "../types";
import { operationCategory, operationStyle, pitchLevelLabel } from "./EditCanvas";

function operationDetails(operation: EditOperation) {
  if (operation.kind === "replace") return `Replace with “${operation.params.target}”`;
  if (operation.kind === "insert") return `Insert “${operation.params.text}”`;
  if (operation.kind === "delete") return "Delete selected transcript";
  if (operation.kind === "emotion") {
    if (operation.params.desc) return `Custom · ${operation.params.desc}`;
    return `${operation.params.type} · level ${operation.params.level ?? 2}`;
  }
  if (operation.kind === "pitch") return pitchLevelLabel(Number(operation.params.semitones));
  if (operation.kind === "rate") return `${Number(operation.params.factor).toFixed(2)}× speed`;
  return `${operation.params.act === "red" ? "Reduce" : "Insert"} · level ${operation.params.level}`;
}

function categoryLabel(operation: EditOperation) {
  if (operation.kind === "replace") return "Replace";
  if (operation.kind === "insert") return "Insert";
  if (operation.kind === "delete") return "Delete";
  return operation.kind[0].toUpperCase() + operation.kind.slice(1);
}

export function OperationList({ operations, enhance = false, colors, segmentNumbers, hoveredOperationId, onHoverOperation, onEdit, onRemove, onRemoveEnhance, onClear }: {
  operations: EditOperation[];
  enhance?: boolean;
  colors: Record<string, string>;
  segmentNumbers: Record<string, number>;
  hoveredOperationId?: string | null;
  onHoverOperation?: (id: string | null) => void;
  onEdit: (operation: EditOperation) => void;
  onRemove: (operation: EditOperation) => void;
  onRemoveEnhance?: () => void;
  onClear: () => void;
}) {
  if (!operations.length && !enhance) return null;
  return <div className="operation-list operation-list--global" aria-label="Edit operations">
    <button className="clear-operations" onClick={onClear}>Clear all</button>
    {enhance && <button className="operation-chip operation-chip--enhance" title="Entire Source Audio" onClick={() => onRemoveEnhance?.()}><span className="operation-chip__index">G</span><span className="operation-chip__copy"><span className="operation-chip__category">Enhance</span><span className="operation-chip__details">Entire source</span></span><X className="operation-chip__remove" size={13} aria-label="Remove Enhance operation" /></button>}
    {operations.map((operation, index) => {
      const style = operationStyle(operation.kind);
      return <button
          key={operation.id}
          className={`operation-chip operation-chip--${operationCategory(operation.kind)} ${hoveredOperationId === operation.id ? "operation-chip--linked" : ""}`}
          style={{ "--op-color": colors[operation.id] || style.color, "--op-bg": style.background } as React.CSSProperties}
          title={`Segment ${segmentNumbers[operation.segment_id]} · ${operation.start}:${operation.end}`}
          onMouseEnter={() => onHoverOperation?.(operation.id)}
          onMouseLeave={() => onHoverOperation?.(null)}
          onFocus={() => onHoverOperation?.(operation.id)}
          onBlur={() => onHoverOperation?.(null)}
          onClick={() => onEdit(operation)}
        >
          <span className="operation-chip__index">{index + 1}</span>
          <span className="operation-chip__copy"><span className="operation-chip__category">{categoryLabel(operation)}</span><span className="operation-chip__details">{operationDetails(operation)}</span></span>
          <X
            className="operation-chip__remove"
            size={13}
            aria-label={`Remove ${categoryLabel(operation)} operation`}
            onClick={(event) => {
              event.stopPropagation();
              onRemove(operation);
            }}
          />
        </button>;
    })}
  </div>;
}
