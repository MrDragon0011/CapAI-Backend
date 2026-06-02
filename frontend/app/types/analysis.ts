export type ActionType = "shooting" | "passing" | "swimming" | "goalie";

export interface Metric {
  metric: string;
  description: string;
  value: number;
  unit: string;
  elite_min: number;
  elite_max: number;
  status: "elite" | "below_elite" | "above_elite";
  feedback: string;
}

export interface PoseFrame {
  t: number;
  lm: [number, number, number][];
}

export interface LandmarkSequence {
  fps: number;
  frames: PoseFrame[];
}

export interface AnalysisResult {
  action: ActionType;
  label: string;
  overall_elite_score_pct: number;
  priority_focus: string;
  total_frames_analysed: number;
  metrics: Metric[];
  landmarks: LandmarkSequence;
}
