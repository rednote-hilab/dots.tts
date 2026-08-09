import { forwardRef, useCallback, useEffect, useRef, useState } from "react";
import type { Destination } from "../types";
import { DRAG_TYPE } from "./AudioCard";

interface Props {
  destination: Destination;
  className?: string;
  children: React.ReactNode;
  onRoute: (source: string, destination: Destination) => void;
  onUpload: (file: File) => void;
  uploadsEnabled?: boolean;
}

export const ReferenceDropZone = forwardRef<HTMLDivElement, Props>(function ReferenceDropZone(
  { destination, className = "", children, onRoute, onUpload, uploadsEnabled = true },
  forwardedRef,
) {
  const depth = useRef(0);
  const [active, setActive] = useState(false);
  const clear = useCallback(() => {
    depth.current = 0;
    setActive(false);
  }, []);
  useEffect(() => {
    window.addEventListener("dragend", clear);
    window.addEventListener("drop", clear);
    window.addEventListener("blur", clear);
    return () => {
      window.removeEventListener("dragend", clear);
      window.removeEventListener("drop", clear);
      window.removeEventListener("blur", clear);
      depth.current = 0;
    };
  }, [clear]);
  const accepts = (event: React.DragEvent) =>
    event.dataTransfer.types.includes(DRAG_TYPE)
    || (uploadsEnabled && event.dataTransfer.types.includes("Files"));

  return (
    <div
      ref={forwardedRef}
      className={`${className} reference-drop-zone ${active ? "reference-drop-zone--active" : ""}`}
      onDragEnter={(event) => {
        if (!accepts(event)) return;
        event.preventDefault();
        depth.current += 1;
        setActive(true);
      }}
      onDragOver={(event) => {
        if (!accepts(event)) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
      }}
      onDragLeave={(event) => {
        if (!accepts(event)) return;
        depth.current = Math.max(0, depth.current - 1);
        if (depth.current === 0) setActive(false);
      }}
      onDrop={(event) => {
        event.preventDefault();
        event.stopPropagation();
        clear();
        const source = event.dataTransfer.getData(DRAG_TYPE);
        if (source) onRoute(source, destination);
        else if (uploadsEnabled && event.dataTransfer.files[0]) onUpload(event.dataTransfer.files[0]);
      }}
    >
      {children}
      {active && <div className="drop-feedback">Drop as {destination === "prompt" ? "Voice Prompt" : "Edit Source"}</div>}
    </div>
  );
});
