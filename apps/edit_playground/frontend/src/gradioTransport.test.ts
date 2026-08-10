import { beforeEach, describe, expect, it, vi } from "vitest";

const gradio = vi.hoisted(() => ({ connect: vi.fn() }));

vi.mock("@gradio/client", () => ({ Client: { connect: gradio.connect } }));

describe("shared Gradio transport", () => {
  beforeEach(() => {
    vi.resetModules();
    gradio.connect.mockReset();
  });

  it("reuses one connected local client", async () => {
    const client = { submit: vi.fn() };
    gradio.connect.mockResolvedValue(client);
    const { getSharedGradioClient } = await import("./gradioTransport");

    const first = getSharedGradioClient();
    const second = getSharedGradioClient();

    await expect(first).resolves.toBe(client);
    await expect(second).resolves.toBe(client);
    expect(gradio.connect).toHaveBeenCalledTimes(1);
    expect(gradio.connect).toHaveBeenCalledWith(window.location.origin, {
      credentials: "include",
      events: ["data", "status"],
    });
  });

  it("allows a later retry after connection failure", async () => {
    const client = { submit: vi.fn() };
    gradio.connect.mockRejectedValueOnce(new Error("network")).mockResolvedValueOnce(client);
    const { getSharedGradioClient } = await import("./gradioTransport");

    await expect(getSharedGradioClient()).rejects.toThrow("network");
    await expect(getSharedGradioClient()).resolves.toBe(client);
    expect(gradio.connect).toHaveBeenCalledTimes(2);
  });
});
