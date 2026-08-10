import { Client } from "@gradio/client";

let sharedClient: Promise<Client> | null = null;

export function getSharedGradioClient(): Promise<Client> {
  if (sharedClient) return sharedClient;

  const pending = Client.connect(window.location.origin, {
    credentials: "include",
    events: ["data", "status"],
  });
  sharedClient = pending;
  void pending.catch(() => {
    if (sharedClient === pending) sharedClient = null;
  });
  return pending;
}
