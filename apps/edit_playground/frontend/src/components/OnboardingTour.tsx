import { useEffect, useMemo, useState } from "react";
import { readPersistentValue, writePersistentValue } from "../browser";

export const ONBOARDING_STORAGE_KEY = "dots-tts-edit-onboarding-v1";

const SHARED_FIRST_STEP = {
  selector: "[data-tour='latest-results']",
  title: "Latest results · 最新结果",
  copy: "Listen to the newest output here, then reuse it as an edit source or voice prompt. / 在这里试听最新结果，并可将它继续用作编辑源音频或声音提示。",
};

const UPLOAD_STEPS = [
  SHARED_FIRST_STEP,
  {
    selector: ".edit-section .audio-card button[aria-label='Choose audio']",
    title: "Upload · 上传",
    copy: "Open the audio library and upload a file. Recognition can fill the transcript for you. / 打开音频库并上传文件，也可以用 Recognition 自动填写转录。",
  },
  {
    selector: ".edit-section .reference-panel",
    title: "Drag and drop · 拖拽",
    copy: "Drop an audio file or drag a previous result into Edit Source. / 可将音频文件或历史结果直接拖入 Edit Source。",
  },
  {
    selector: ".edit-section .edit-canvas",
    title: "Add an edit · 添加编辑操作",
    copy: "Select words for span edits, or click a boundary for insertion and pause edits. / 选中文字添加区间编辑；点击文字边界添加插入或停顿编辑。",
  },
];

const NO_UPLOAD_STEPS = [
  SHARED_FIRST_STEP,
  {
    selector: ".edit-section .preset-select",
    title: "Choose a preset · 选择预置",
    copy: "Start with a curated source preset and its transcript. / 从精选的源音频预置及其转录开始。",
  },
  {
    selector: ".edit-section .reference-panel",
    title: "Reuse and drag · 复用与拖拽",
    copy: "Drag a previous result into Edit Source, or route it from its menu. / 将历史结果拖入 Edit Source，或从结果菜单中复用。",
  },
  {
    selector: ".edit-section .edit-canvas",
    title: "Add an edit · 添加编辑操作",
    copy: "Select words for span edits, or click a boundary for insertion and pause edits. / 选中文字添加区间编辑；点击文字边界添加插入或停顿编辑。",
  },
];

interface Rect {
  top: number;
  left: number;
  right: number;
  bottom: number;
  width: number;
  height: number;
}

export function OnboardingTour({ uploadsEnabled = true }: { uploadsEnabled?: boolean }) {
  const steps = uploadsEnabled ? UPLOAD_STEPS : NO_UPLOAD_STEPS;
  const [step, setStep] = useState(() => readPersistentValue(ONBOARDING_STORAGE_KEY) ? -1 : 0);
  const [rect, setRect] = useState<Rect | null>(null);
  const item = step >= 0 ? steps[step] : null;

  useEffect(() => {
    if (!item) return;
    const update = () => {
      const target = document.querySelector<HTMLElement>(item.selector);
      if (!target) return setRect(null);
      const value = target.getBoundingClientRect();
      const padding = 8;
      setRect({
        top: Math.max(8, value.top - padding),
        left: Math.max(8, value.left - padding),
        right: Math.min(window.innerWidth - 8, value.right + padding),
        bottom: Math.min(window.innerHeight - 8, value.bottom + padding),
        width: Math.min(window.innerWidth - 16, value.width + padding * 2),
        height: Math.min(window.innerHeight - 16, value.height + padding * 2),
      });
      target.scrollIntoView?.({ behavior: "smooth", block: "center" });
    };
    const timer = window.setTimeout(update, 120);
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [item]);

  const cardStyle = useMemo(() => {
    if (!rect) return { left: 24, top: 24 };
    const width = Math.min(390, window.innerWidth - 32);
    const below = rect.bottom + 14;
    const top = below + 190 < window.innerHeight
      ? below
      : Math.max(16, rect.top - 190);
    const left = Math.min(
      window.innerWidth - width - 16,
      Math.max(16, rect.left + rect.width / 2 - width / 2),
    );
    return { left, top, width };
  }, [rect]);

  if (!item || !rect) return null;
  const finish = () => {
    writePersistentValue(ONBOARDING_STORAGE_KEY, "complete");
    setStep(-1);
  };
  const next = () => {
    if (step === steps.length - 1) finish();
    else setStep((value) => value + 1);
  };

  return <div className="onboarding-tour" role="dialog" aria-modal="true" aria-label="First-use guide">
    <div className="tour-shade tour-shade--top" style={{ height: rect.top }} />
    <div className="tour-shade tour-shade--left" style={{ top: rect.top, width: rect.left, height: rect.height }} />
    <div className="tour-shade tour-shade--right" style={{ top: rect.top, left: rect.right, height: rect.height }} />
    <div className="tour-shade tour-shade--bottom" style={{ top: rect.bottom }} />
    <div className="tour-highlight" style={{ top: rect.top, left: rect.left, width: rect.width, height: rect.height }} />
    <section className="tour-card" style={cardStyle}>
      <span className="tour-progress">{step + 1} / {steps.length}</span>
      <h2>{item.title}</h2>
      <p>{item.copy}</p>
      <footer><button className="quiet-button" onClick={finish}>Skip all · 全部跳过</button><button className="primary-button" onClick={next}>{step === steps.length - 1 ? "Done · 完成" : "Next · 下一步"}</button></footer>
    </section>
  </div>;
}
