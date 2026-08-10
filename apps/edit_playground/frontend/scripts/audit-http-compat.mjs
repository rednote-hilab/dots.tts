import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "../src");
const checks = [
  [/(?<!globalThis\.)\bcrypto\.randomUUID\s*\(/, "unguarded crypto.randomUUID"],
  [/navigator\.clipboard/, "secure-context Clipboard API"],
  [/navigator\.mediaDevices|\bgetUserMedia\s*\(/, "secure-context media capture"],
  [/navigator\.serviceWorker/, "secure-context Service Worker"],
  [/\b(?:PublicKeyCredential|CredentialsContainer)\b/, "secure-context WebAuthn"],
  [/\bshow(?:Open|Save)FilePicker\s*\(/, "secure-context File System Access API"],
  [/\b(?:localhost|127\.0\.0\.1|wss:\/\/|https:\/\/)/, "hard-coded deployment origin"],
];

const files = [];
function visit(directory) {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) visit(target);
    else if (/\.(?:ts|tsx)$/.test(entry.name) && !entry.name.endsWith(".test.ts") && !entry.name.endsWith(".test.tsx")) files.push(target);
  }
}
visit(root);

const failures = [];
for (const file of files) {
  const source = fs.readFileSync(file, "utf8");
  for (const [pattern, label] of checks) {
    if (pattern.test(source)) failures.push(`${path.relative(root, file)}: ${label}`);
  }
}
if (failures.length) {
  console.error(`Public HTTP compatibility audit failed:\n${failures.join("\n")}`);
  process.exit(1);
}
console.log(`Public HTTP compatibility audit passed (${files.length} source files).`);
