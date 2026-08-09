import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { AudioReference } from "../types";
import { classifySegmentDrop, segmentDropIntent, SourceAudioTimeline } from "./SourceAudioTimeline";

vi.mock("wavesurfer.js", () => ({
  default: {
    create: () => ({ on: () => () => {}, destroy: () => {}, playPause: () => {}, pause: () => {} }),
  },
}));

const reference: AudioReference = {
  source_id: "edit_source",
  audio_url: "/source.wav",
  text: "first",
  revision_id: null,
  duration_seconds: 2,
  composition_version: "v1",
  use_xvector: true,
  origin: { kind: "custom", name: null },
  segments: [{
    id: "first",
    source_id: "first",
    audio_url: "/first.wav",
    transcript: "first",
    duration_seconds: 2,
    revision_id: null,
    multiple_speakers: false,
    allow_xvector: true,
  }],
};

function setup() {
  const handlers = {
    onRoute: vi.fn(),
    onInsert: vi.fn(),
    onMove: vi.fn(),
    onReplace: vi.fn(),
    onRemove: vi.fn(),
    onUploadWhole: vi.fn(),
    onUploadAt: vi.fn(),
    onUploadReplace: vi.fn(),
  };
  const rendered = render(<SourceAudioTimeline reference={reference} enhance={false} onEnhanceToggle={() => {}} renderTranscript={(id) => <div>{id} transcript</div>} {...handlers} />);
  return { ...rendered, handlers };
}

describe("SourceAudioTimeline", () => {
  it("maps segment thirds to before, replace, and after", () => {
    expect(classifySegmentDrop(120, 100, 300)).toBe("before");
    expect(classifySegmentDrop(250, 100, 300)).toBe("replace");
    expect(classifySegmentDrop(390, 100, 300)).toBe("after");
  });

  it("moves an existing source segment instead of copying it", () => {
    expect(segmentDropIntent("after", true)).toBe("move");
    expect(segmentDropIntent("replace", true)).toBe("none");
    expect(segmentDropIntent("after", false)).toBe("insert");
    expect(segmentDropIntent("replace", false)).toBe("replace");
    expect(segmentDropIntent("after", false, false)).toBe("none");
  });

  it("shows upload gaps only while another segment can be added", () => {
    const { getAllByRole, rerender } = setup();
    expect(getAllByRole("button", { name: /Insert audio/ })).toHaveLength(2);
    const full = {
      ...reference,
      use_xvector: false,
      segments: [
        reference.segments[0],
        { ...reference.segments[0], id: "second", source_id: "second" },
        { ...reference.segments[0], id: "third", source_id: "third" },
      ],
    };
    rerender(<SourceAudioTimeline reference={full} enhance={false} onEnhanceToggle={() => {}} renderTranscript={(id) => <div>{id} transcript</div>} onRoute={() => {}} onInsert={() => {}} onMove={() => {}} onReplace={() => {}} onRemove={() => {}} onUploadWhole={() => {}} onUploadAt={() => {}} onUploadReplace={() => {}} />);
    expect(getAllByRole("button", { name: /Remove Segment/ })).toHaveLength(3);
    expect(() => getAllByRole("button", { name: /Insert audio/ })).toThrow();
  });

  it("opens one background noise control on the single source segment", () => {
    const onNoise = vi.fn();
    const noisy = {
      ...reference,
      noise: { item_id: "ambient", kind: "presets" as const, snr_db: 15 },
    };
    const { getByRole } = render(
      <SourceAudioTimeline
        reference={noisy}
        renderTranscript={(id) => <div>{id} transcript</div>}
        onRoute={() => {}}
        onInsert={() => {}}
        onMove={() => {}}
        onReplace={() => {}}
        onRemove={() => {}}
        onUploadWhole={() => {}}
        onUploadAt={() => {}}
        onUploadReplace={() => {}}
        onNoise={onNoise}
        enhance={false}
        onEnhanceToggle={() => {}}
      />,
    );
    fireEvent.click(getByRole("button", { name: "Edit background noise, 15 dB" }));
    expect(onNoise).toHaveBeenCalledWith();
    expect(() => getByRole("button", { name: /Segment.*background noise/ })).toThrow();
  });

  it("keeps the hint, Noise, and Enhance on the segment when there is only one", () => {
    const onEnhanceToggle = vi.fn();
    const { getByText, getByRole } = render(
      <SourceAudioTimeline
        reference={reference}
        enhance
        onEnhanceToggle={onEnhanceToggle}
        globalHint={<div>Global transcript hint</div>}
        renderTranscript={(id) => <div>{id} transcript</div>}
        onRoute={() => {}}
        onInsert={() => {}}
        onMove={() => {}}
        onReplace={() => {}}
        onRemove={() => {}}
        onUploadWhole={() => {}}
        onUploadAt={() => {}}
        onUploadReplace={() => {}}
        onNoise={() => {}}
      />,
    );
    expect(() => getByText("Full Source Audio")).toThrow();
    expect(getByRole("button", { name: "Add background noise to Source Segment" })).toBeInTheDocument();
    expect(getByText("Global transcript hint")).toBeInTheDocument();
    fireEvent.click(getByRole("button", { name: "Enhance" }));
    expect(onEnhanceToggle).toHaveBeenCalledOnce();
  });

  it("hides file controls but keeps preset segment insertion when uploads are disabled", () => {
    const chooseSegment = vi.fn();
    const { queryByRole, queryAllByRole } = render(
      <SourceAudioTimeline
        reference={reference}
        uploadsEnabled={false}
        enhance={false}
        onEnhanceToggle={() => {}}
        renderTranscript={(id) => <div>{id} transcript</div>}
        onRoute={() => {}}
        onInsert={() => {}}
        onMove={() => {}}
        onReplace={() => {}}
        onRemove={() => {}}
        onUploadWhole={() => {}}
        onUploadAt={chooseSegment}
        onUploadReplace={() => {}}
      />,
    );
    expect(queryByRole("button", { name: "Choose audio" })).not.toBeInTheDocument();
    expect(queryAllByRole("button", { name: /Insert audio/ })).toHaveLength(2);
    fireEvent.click(queryAllByRole("button", { name: /Insert audio/ })[0]);
    expect(chooseSegment).toHaveBeenCalledWith(0);
    expect(queryByRole("button", { name: "Enhance" })).toBeInTheDocument();
  });
});
