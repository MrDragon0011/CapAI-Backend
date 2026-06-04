"use client";

import { motion, AnimatePresence } from "framer-motion";
import { AnalysisResult, ActionType, Metric } from "../types/analysis";
import { ACTION_CONFIG } from "./ActionBadge";

const BODY_FOCUS: Record<ActionType, string[]> = {
  shooting: ["Right Arm Chain", "Hip Drive", "Trunk Rotation", "Release Height"],
  passing: ["Bilateral Arms", "Wrist Release", "Follow-Through", "Torso Lean"],
  swimming: ["Stroke Symmetry", "High-Elbow Catch", "Hip Roll", "Head Position"],
  goalie: ["Eggbeater Kick", "Vertical Stability", "Lateral Reach", "Arm Width"],
};

const ACTION_DESC: Record<ActionType, string> = {
  shooting:
    "Explosive arm extension, trunk rotation, and hip-to-shoulder kinetic chain loading are the defining biomechanical signatures of an elite power shot.",
  passing:
    "Crisp bilateral arm mechanics, clean release point, and deceptive follow-through separate elite passers from average ones.",
  swimming:
    "Symmetric stroke mechanics, high-elbow catch position, and coordinated hip roll maximise propulsive efficiency through the water.",
  goalie:
    "Sustained eggbeater elevation, minimal vertical oscillation, and explosive lateral reach are the cornerstones of elite goalkeeping.",
};

const STATUS_COLOR: Record<Metric["status"], string> = {
  elite: "text-emerald-400",
  below_elite: "text-amber-400",
  above_elite: "text-rose-400",
};

const STATUS_LABEL: Record<Metric["status"], string> = {
  elite: "✓ Elite",
  below_elite: "↓ Below",
  above_elite: "↑ Above",
};

function TopMetric({ metric, index }: { metric: Metric; index: number }) {
  const displayValue =
    metric.unit === "degrees"
      ? `${metric.value.toFixed(1)}°`
      : metric.unit === "ratio"
      ? metric.value.toFixed(3)
      : metric.value.toFixed(3);

  return (
    <motion.div
      initial={{ opacity: 0, x: 12 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.2 + index * 0.08 }}
      className="flex items-center justify-between gap-3 py-3 border-b last:border-0"
      style={{ borderColor: "var(--border)" }}
    >
      <div className="min-w-0">
        <p className="text-[12px] text-white/60 truncate">{metric.description}</p>
        <p className="text-[11px] text-white/30 mt-0.5 leading-snug line-clamp-2">{metric.feedback}</p>
      </div>
      <div className="shrink-0 text-right">
        <p className="text-base font-bold text-white">{displayValue}</p>
        <p className={`text-[10px] font-medium ${STATUS_COLOR[metric.status]}`}>
          {STATUS_LABEL[metric.status]}
        </p>
      </div>
    </motion.div>
  );
}

interface Props {
  result: AnalysisResult;
}

export function ActionInsightPanel({ result }: Props) {
  const { action, label, priority_focus, metrics, overall_elite_score_pct } = result;
  const cfg = ACTION_CONFIG[action];
  const focus = BODY_FOCUS[action];
  const desc = ACTION_DESC[action];

  const nonElite = metrics.filter((m) => m.status !== "elite");
  const eliteMetrics = metrics.filter((m) => m.status === "elite");
  const topMetrics = [...nonElite, ...eliteMetrics].slice(0, 4);

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={action}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -20 }}
        transition={{ duration: 0.4 }}
        className="flex flex-col gap-4 h-full"
      >
        <div
          className="p-5"
          style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
        >
          <div className="flex items-start justify-between mb-4">
            <div>
              <p className="font-mono text-[10px] uppercase tracking-widest text-white/25 mb-1">
                Action detected
              </p>
              <h3 className="text-xl font-black text-white tracking-tight">{label.toUpperCase()}</h3>
            </div>
            <div className="text-right">
              <p className="font-mono text-[10px] text-white/25 mb-1">Elite Score</p>
              <p className="text-3xl font-black text-white">{overall_elite_score_pct}%</p>
            </div>
          </div>
          <p className="text-[12px] text-white/40 leading-relaxed">{desc}</p>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="p-5"
          style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
        >
          <p className="font-mono text-[10px] uppercase tracking-widest text-white/25 mb-3">Body Focus</p>
          <div className="flex flex-wrap gap-2">
            {focus.map((tag, i) => (
              <motion.span
                key={tag}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 + i * 0.06 }}
                className="px-2.5 py-1 font-mono text-[10px] text-white/50"
                style={{ border: "1px solid var(--border-strong)" }}
              >
                {tag}
              </motion.span>
            ))}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="p-5"
          style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
        >
          <p className="font-mono text-[10px] uppercase tracking-widest text-white/25 mb-2">Critical Insight</p>
          <div className="flex gap-3">
            <div className="shrink-0 w-px self-stretch" style={{ background: "var(--accent)" }} />
            <p className="text-[13px] text-white/65 leading-relaxed">{priority_focus}</p>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="p-5 flex-1"
          style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
        >
          <p className="font-mono text-[10px] uppercase tracking-widest text-white/25 mb-1">Key Metrics</p>
          <div>
            {topMetrics.map((m, i) => (
              <TopMetric key={m.metric} metric={m} index={i} />
            ))}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
