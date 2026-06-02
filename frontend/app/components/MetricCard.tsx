"use client";

import { motion } from "framer-motion";
import { Metric } from "../types/analysis";

const STATUS_STYLES = {
  elite: {
    bar: "bg-emerald-400",
    badge: "bg-emerald-400/15 text-emerald-300 border border-emerald-400/30",
    label: "Elite",
    glow: "shadow-emerald-500/10",
  },
  below_elite: {
    bar: "bg-amber-400",
    badge: "bg-amber-400/15 text-amber-300 border border-amber-400/30",
    label: "Below Elite",
    glow: "shadow-amber-500/10",
  },
  above_elite: {
    bar: "bg-rose-400",
    badge: "bg-rose-400/15 text-rose-300 border border-rose-400/30",
    label: "Above Elite",
    glow: "shadow-rose-500/10",
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
      className={`relative rounded-2xl border border-white/[0.07] bg-white/[0.04] backdrop-blur-xl p-5 shadow-lg ${s.glow} overflow-hidden`}
    >
      <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-white/[0.03] to-transparent pointer-events-none" />

      <div className="flex items-start justify-between gap-3 mb-3">
        <p className="text-[13px] font-medium text-white/70 leading-snug">{metric.description}</p>
        <span className={`shrink-0 text-[11px] font-semibold px-2 py-0.5 rounded-full ${s.badge}`}>
          {s.label}
        </span>
      </div>

      <div className="flex items-end justify-between mb-3">
        <span className="text-2xl font-bold text-white">{displayValue}</span>
        <span className="text-[11px] text-white/30 uppercase tracking-wider">{metric.unit}</span>
      </div>

      <div className="h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${s.bar}`}
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
