import { render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { AppErrorBoundary } from "./AppErrorBoundary";

function BrokenStudio(): never {
  throw new Error("startup failure");
}

afterEach(() => vi.restoreAllMocks());

it("shows a recoverable startup screen instead of an empty root", () => {
  vi.spyOn(console, "error").mockImplementation(() => {});
  render(<AppErrorBoundary><BrokenStudio /></AppErrorBoundary>);
  expect(screen.getByRole("alert")).toHaveTextContent("Edit Playground could not start.");
  expect(screen.getByRole("button", { name: "Reload" })).toBeInTheDocument();
});
