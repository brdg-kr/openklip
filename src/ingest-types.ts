export type IngestPhase =
  | "probe"
  | "proxy"
  | "audio"
  | "frames"
  | "index"
  | "media-index"
  | "transcribe"
  | "finalize"
  | "done";

export interface IngestProgress {
  message: string;
  phase: IngestPhase;
  /** 1-based step index (the work phases, excluding "done"). */
  step: number;
  total: number;
}
