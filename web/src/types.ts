// Mirrors the JSON written by vision/timeline.py.
export type Box = [number, number, number, number]; // x, y, w, h as 0-1 fractions

export type Detection = {
  label: string;
  group: "person" | "animal" | "vehicle" | "bag" | "other";
  score: number;
  box: Box;
};

export type Frame = {
  t: number;
  box: Box | null;
  area: number;
  blur: number;
  bright: number;
  ir: boolean;
  objs?: Detection[];
};

export type Decision = {
  action: "alert" | "ignore" | "recheck" | "log";
  reason: string;
  source: string;
};

export type ClipEvent = {
  id: number;
  type: "motion" | "scene_change" | "night_mode" | "day_mode" | "blurry" | "dark" | "cut";
  t_start: number;
  t_end: number;
  box: Box | null;
  say: string;
  decision: Decision;
  label?: string | null;
  objects?: Record<string, number>;
};

export type Timeline = {
  clip: string;
  video: string;
  width: number;
  height: number;
  fps: number;
  duration: number;
  thresholds: { blur_min: number; dark_max: number };
  frames: Frame[];
  events: ClipEvent[];
};

export type IndexEntry = { name: string; timeline: string; events: number; duration: number };
