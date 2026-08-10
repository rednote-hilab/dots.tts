import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { ONBOARDING_STORAGE_KEY, OnboardingTour } from "./OnboardingTour";

function mountTargets() {
  const targets = document.createElement("div");
  targets.innerHTML = `
    <div data-tour="latest-results"></div>
    <section class="edit-section">
      <label class="preset-select"><select></select></label>
      <div class="audio-card"><button aria-label="Choose audio"></button></div>
      <div class="reference-panel"></div>
      <div class="edit-canvas"></div>
    </section>
  `;
  document.body.appendChild(targets);
  for (const target of targets.querySelectorAll<HTMLElement>("div, button")) {
    target.getBoundingClientRect = () => new DOMRect(40, 40, 240, 80);
  }
  return targets;
}

describe("OnboardingTour", () => {
  beforeEach(() => localStorage.clear());

  it("walks through no more than four bilingual spotlight steps", async () => {
    const targets = mountTargets();
    render(<OnboardingTour />);
    expect(await screen.findByText("Latest results · 最新结果")).toBeInTheDocument();
    for (const title of [
      "Upload · 上传",
      "Drag and drop · 拖拽",
      "Add an edit · 添加编辑操作",
    ]) {
      fireEvent.click(screen.getByRole("button", { name: "Next · 下一步" }));
      expect(await screen.findByText(title)).toBeInTheDocument();
    }
    fireEvent.click(screen.getByRole("button", { name: "Done · 完成" }));
    expect(screen.queryByRole("dialog", { name: "First-use guide" })).not.toBeInTheDocument();
    expect(localStorage.getItem(ONBOARDING_STORAGE_KEY)).toBe("complete");
    targets.remove();
  });

  it("skips all steps and stays dismissed for the browser", async () => {
    const targets = mountTargets();
    const first = render(<OnboardingTour />);
    fireEvent.click(await screen.findByRole("button", { name: "Skip all · 全部跳过" }));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "First-use guide" })).not.toBeInTheDocument());
    first.unmount();
    render(<OnboardingTour />);
    expect(screen.queryByRole("dialog", { name: "First-use guide" })).not.toBeInTheDocument();
    targets.remove();
  });

  it("uses preset and internal-reuse guidance when uploads are disabled", async () => {
    const targets = mountTargets();
    render(<OnboardingTour uploadsEnabled={false} />);
    expect(await screen.findByText("Latest results · 最新结果")).toBeInTheDocument();
    for (const title of [
      "Choose a preset · 选择预置",
      "Reuse and drag · 复用与拖拽",
      "Add an edit · 添加编辑操作",
    ]) {
      fireEvent.click(screen.getByRole("button", { name: "Next · 下一步" }));
      expect(await screen.findByText(title)).toBeInTheDocument();
    }
    expect(screen.queryByText("Upload · 上传")).not.toBeInTheDocument();
    expect(screen.queryByText(/Recognition/)).not.toBeInTheDocument();
    targets.remove();
  });
});
