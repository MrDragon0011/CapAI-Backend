"use client";

import { motion } from "framer-motion";
import { Metric } from "../types/analysis";

const STATUS_STYLES = {
  elite: {
    bar: "bg-emerald-400",
    badge: "text-emerald-400 border border-emerald-400/40",
    label: "Elite",
    glow: "",
  },
  below_elite: {
    bar: "bg-amber-400",
    badge: "text-amber-400 border border-amber-400/40",
    label: "Below Elite",
    glow: "",
  },
  above_elite: {
    bar: "bg-rose-400",
    badge: "text-rose-400 border border-rose-400/40",
    label: "Above Elite",
    glow: "",
  },
};

function normalise(value: number, min: number, max: number) {
  return Math.min(1, Math.max(0, (value - min) / (max - min + 1e-8)));
}

export function MetricCard({ metric, index }: { metric: Metric; index: number }) {
  const s = STATUS_STYLES[metric.status];
  const pct = normalise(metric.value, metric.elite_min, metric.elite_max);
  const displayValue =
    metric.unit === "degrees"
      ? `${metric.value.toFixed(1)}°`
      : metric.unit === "ratio"
      ? metric.value.toFixed(3)
      : metric.value.toFixed(4);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.07, duration: 0.4 }}
      className="relative overflow-hidden p-5"
      style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
    >
      <div className="flex items-start justify-between gap-3 mb-3">
        <p className="text-[13px] font-medium text-white/70 leading-snug">{metric.description}</p>
        <span className={`shrink-0 font-mono text-[10px] px-2 py-0.5 ${s.badge}`}>
          {s.label}
        </span>
      </div>

      <div className="flex items-end justify-between mb-3">
        <span className="text-2xl font-bold text-white">{displayValue}</span>
        <span className="font-mono text-[10px] text-white/25 uppercase tracking-wider">{metric.unit}</span>
      </div>

      <div className="h-px bg-white/[0.08] overflow-hidden relative" style={{ height: 2 }}>
        <motion.div
          className={`h-full ${s.bar}`}
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(100, pct * 100)}%` }}
          transition={{ duration: 0.8, delay: index * 0.07 + 0.2, ease: "easeOut" }}
        />
      </div>

      <div className="flex justify-between mt-1">
        <span className="text-[10px] text-white/25">{metric.elite_min}{metric.unit === "degrees" ? "°" : ""}</span>
        <span className="text-[10px] text-white/25">{metric.elite_max}{metric.unit === "degrees" ? "°" : ""} elite range</span>
      </div>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: index * 0.07 + 0.3 }}
        className="mt-3 text-[12px] leading-relaxed text-white/45"
      >
        {metric.feedback}
      </motion.p>
    </motion.div>
  );
}
