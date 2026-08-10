let fallbackCounter = 0;
const memorySession = new Map<string, string>();
const memoryPersistent = new Map<string, string>();

function uuidFromBytes(bytes: Uint8Array) {
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, "0"));
  return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
}

export function createId() {
  const webCrypto = globalThis.crypto;
  if (typeof webCrypto?.randomUUID === "function") return webCrypto.randomUUID();
  if (typeof webCrypto?.getRandomValues === "function") {
    return uuidFromBytes(webCrypto.getRandomValues(new Uint8Array(16)));
  }
  fallbackCounter += 1;
  return `${Date.now().toString(36)}-${fallbackCounter.toString(36)}-${Math.random().toString(36).slice(2, 14)}`;
}

export function readSessionValue(key: string) {
  try {
    return globalThis.sessionStorage?.getItem(key) ?? memorySession.get(key) ?? null;
  } catch {
    return memorySession.get(key) ?? null;
  }
}

export function writeSessionValue(key: string, value: string) {
  memorySession.set(key, value);
  try {
    globalThis.sessionStorage?.setItem(key, value);
  } catch {
    // Browser policy may disable storage; memory keeps this page usable.
  }
}

export function readPersistentValue(key: string) {
  try {
    if (globalThis.localStorage) return globalThis.localStorage.getItem(key);
    return memoryPersistent.get(key) ?? null;
  } catch {
    return memoryPersistent.get(key) ?? null;
  }
}

export function writePersistentValue(key: string, value: string) {
  memoryPersistent.set(key, value);
  try {
    globalThis.localStorage?.setItem(key, value);
  } catch {
    // Browser policy may disable storage; memory keeps this page usable.
  }
}
