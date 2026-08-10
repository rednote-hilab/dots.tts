import { beforeEach, describe, expect, it, vi } from "vitest";

const transport = vi.hoisted(() => ({ getSharedGradioClient: vi.fn() }));

vi.mock("./gradioTransport", () => transport);

import { studioApi } from "./api";

function response(body: unknown, init: { ok?: boolean; status?: number; statusText?: string } = {}) {
  return {
    ok: init.ok ?? true,
    status: init.status ?? 200,
    statusText: init.statusText ?? "OK",
    json: async () => body,
  } as Response;
}

async function* recognitionEvents(events: unknown[]) {
  for (const event of events) yield event;
}

describe("studio API queue transport", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    transport.getSharedGradioClient.mockReset();
  });

  it("recognizes with the shared client and removes the temporary upload", async () => {
    const submit = vi.fn(() => recognitionEvents([
      { type: "status", stage: "pending" },
      { type: "data", data: [JSON.stringify({ text: "识别文本", language: "Chinese" })] },
    ]));
    transport.getSharedGradioClient.mockResolvedValue({ submit });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response({ job_id: "recognition-1" }))
      .mockResolvedValueOnce(response(null, { status: 204, statusText: "No Content" }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(studioApi.recognize("session", new File(["wav"], "sample.wav"))).resolves.toEqual({
      text: "识别文本",
      language: "Chinese",
    });

    expect(submit).toHaveBeenCalledWith(
      "/recognize",
      ["recognition-1"],
      undefined,
      null,
      true,
    );
    expect(fetchMock).toHaveBeenLastCalledWith(
      "/api/session/session/recognition/recognition-1",
      { method: "DELETE" },
    );
  });

  it("keeps the inference error and still cleans up", async () => {
    transport.getSharedGradioClient.mockResolvedValue({
      submit: () => recognitionEvents([
        { type: "status", stage: "error", message: "ASR ran out of memory" },
      ]),
    });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response({ job_id: "recognition-2" }))
      .mockResolvedValueOnce(response(null, { status: 204, statusText: "No Content" }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(studioApi.recognize("session", new File(["wav"], "sample.wav")))
      .rejects.toThrow("Recognition inference failed. ASR ran out of memory");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("reports queue transport failures separately and still cleans up", async () => {
    transport.getSharedGradioClient.mockResolvedValue({
      submit: () => recognitionEvents([Promise.reject(new TypeError("Failed to fetch"))]),
    });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response({ job_id: "recognition-3" }))
      .mockResolvedValueOnce(response(null, { status: 204, statusText: "No Content" }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(studioApi.recognize("session", new File(["wav"], "sample.wav")))
      .rejects.toThrow("Recognition could not enter or read from the local inference queue");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("surfaces cleanup failure after successful recognition", async () => {
    transport.getSharedGradioClient.mockResolvedValue({
      submit: () => recognitionEvents([
        { type: "data", data: [JSON.stringify({ text: "hello", language: "English" })] },
      ]),
    });
    vi.stubGlobal("fetch", vi.fn()
      .mockResolvedValueOnce(response({ job_id: "recognition-4" }))
      .mockResolvedValueOnce(response(null, { ok: false, status: 500, statusText: "Server Error" })));

    await expect(studioApi.recognize("session", new File(["wav"], "sample.wav")))
      .rejects.toThrow("temporary upload cleanup failed");
  });

  it("submits generation directly to the local inference queue", async () => {
    const complete = { phase: "complete", progress: 1 };
    const submit = vi.fn(() => recognitionEvents([
      { type: "data", data: [JSON.stringify(complete)] },
    ]));
    transport.getSharedGradioClient.mockResolvedValue({ submit });
    const events: unknown[] = [];

    await studioApi.tts(
      "session",
      "hello",
      { ode_method: "euler", num_steps: 2, guidance_scale: 1, speaker_scale: 1, use_xvector: true, seed: 3, model_id: "model" },
      (event) => events.push(event),
    );

    expect(submit).toHaveBeenCalledWith(
      "/synthesize_tts",
      ["session", "hello", expect.any(String)],
      undefined,
      null,
      true,
    );
    expect(events).toEqual([complete]);
  });

});
