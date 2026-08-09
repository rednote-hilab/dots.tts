import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import type { EditGenerationSettings, GenerationSettings } from "../types";
import { GeneratorSettingsPanel } from "./GeneratorSettings";

const defaults: GenerationSettings = {
  ode_method: "euler",
  num_steps: 10,
  guidance_scale: 1,
  speaker_scale: 1.5,
  use_xvector: false,
  seed: 3,
};

function Harness({ available = true }: { available?: boolean }) {
  const [settings, setSettings] = useState(defaults);
  return <GeneratorSettingsPanel settings={settings} odeMethods={["euler"]} onChange={setSettings} speakerGuidanceAvailable={available} />;
}

const editDefaults: EditGenerationSettings = {
  ...defaults,
  use_xvector: "auto",
};

function AutoHarness({ available = true, autoEnabled = true }: { available?: boolean; autoEnabled?: boolean }) {
  const [settings, setSettings] = useState(editDefaults);
  return <GeneratorSettingsPanel settings={settings} odeMethods={["euler"]} onChange={setSettings} speakerGuidanceAvailable={available} speakerGuidanceMode="auto" autoSpeakerGuidanceEnabled={autoEnabled} />;
}

describe("GeneratorSettingsPanel", () => {
  it("keeps the TTS boolean control enabled at scale 1.5", () => {
    render(<Harness />);
    const checkbox = screen.getByRole("checkbox", { name: "Use Speaker Guidance" });
    const scale = screen.getByRole("slider", { name: "Scale" });

    expect(checkbox).not.toBeChecked();
    fireEvent.click(checkbox);
    expect(checkbox).toBeChecked();
    expect(scale).toBeEnabled();
    expect(scale).toHaveValue("1.5");
  });

  it("cannot enable speaker guidance when the source does not support it", () => {
    render(<Harness available={false} />);

    expect(screen.getByRole("checkbox", { name: "Use Speaker Guidance" })).toBeDisabled();
    expect(screen.getByRole("slider", { name: "Scale" })).toBeDisabled();
    expect(screen.getByText("Speaker guidance unavailable for this audio")).toBeInTheDocument();
  });

  it("offers Auto, On and Off while showing the resolved Auto state", () => {
    render(<AutoHarness />);
    const auto = screen.getByRole("button", { name: "Auto" });
    const on = screen.getByRole("button", { name: "On" });
    const off = screen.getByRole("button", { name: "Off" });
    const scale = screen.getByRole("slider", { name: "Scale" });

    expect(auto).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("Auto · On")).toBeInTheDocument();
    expect(scale).toBeEnabled();

    fireEvent.click(off);
    expect(off).toHaveAttribute("aria-pressed", "true");
    expect(scale).toBeDisabled();
    expect(scale.closest("label")?.querySelector("output")).toBeNull();

    fireEvent.click(on);
    expect(on).toHaveAttribute("aria-pressed", "true");
    expect(scale).toBeEnabled();
    expect(scale).toHaveValue("1.5");
  });

  it("shows Auto Off with a thumbless scale for emotion-only edits", () => {
    render(<AutoHarness autoEnabled={false} />);
    const scale = screen.getByRole("slider", { name: "Scale" });

    expect(screen.getByText("Auto · Off")).toBeInTheDocument();
    expect(scale).toBeDisabled();
    expect(scale.closest("label")).toHaveClass("range-setting--thumbless");
    expect(scale.closest("label")?.querySelector("output")).toBeNull();
  });

  it("preserves the requested mode while source capability forces Off", () => {
    const { rerender } = render(<AutoHarness available={false} />);

    expect(screen.getByText("Forced Off")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Auto" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Auto" })).toBeDisabled();

    rerender(<AutoHarness available />);
    expect(screen.getByRole("button", { name: "Auto" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("Auto · On")).toBeInTheDocument();
  });
});
