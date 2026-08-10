import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  failed: boolean;
}

export class AppErrorBoundary extends Component<Props, State> {
  state: State = { failed: false };

  static getDerivedStateFromError(): State {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Studio startup failed", error, info.componentStack);
  }

  render() {
    if (this.state.failed) {
      return (
        <main className="fatal-startup" role="alert">
          <div className="brand-mark">d</div>
          <h1>Edit Playground could not start.</h1>
          <button type="button" onClick={() => window.location.reload()}>Reload</button>
        </main>
      );
    }
    return this.props.children;
  }
}
