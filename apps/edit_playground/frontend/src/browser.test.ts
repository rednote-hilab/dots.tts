import { afterEach, describe, expect, it, vi } from "vitest";
import { createId, readPersistentValue, readSessionValue, writePersistentValue, writeSessionValue } from "./browser";

describe("public HTTP browser compatibility", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("uses randomUUID when the secure-context API is available", () => {
    const randomUUID = vi.fn(() => "11111111-1111-4111-8111-111111111111" as `${string}-${string}-${string}-${string}-${string}`);
    vi.stubGlobal("crypto", { randomUUID });
    expect(createId()).toBe("11111111-1111-4111-8111-111111111111");
    expect(randomUUID).toHaveBeenCalledOnce();
  });

  it("constructs UUID v4 with getRandomValues on an insecure origin", () => {
    vi.stubGlobal("crypto", {
      getRandomValues: (values: Uint8Array) => {
        values.fill(0);
        return values;
      },
    });
    expect(createId()).toBe("00000000-0000-4000-8000-000000000000");
  });

  it("still returns distinct IDs when crypto is unavailable", () => {
    vi.stubGlobal("crypto", undefined);
    const first = createId();
    const second = createId();
    expect(first).not.toBe(second);
    expect(first.length).toBeGreaterThan(10);
  });

  it("falls back to memory when sessionStorage is blocked", () => {
    vi.stubGlobal("sessionStorage", {
      getItem: () => { throw new DOMException("blocked", "SecurityError"); },
      setItem: () => { throw new DOMException("blocked", "SecurityError"); },
    });
    writeSessionValue("blocked-storage-test", "available");
    expect(readSessionValue("blocked-storage-test")).toBe("available");
  });

  it("persists first-use completion in browser-local storage", () => {
    localStorage.clear();
    writePersistentValue("onboarding-test", "complete");
    expect(readPersistentValue("onboarding-test")).toBe("complete");
  });
});
