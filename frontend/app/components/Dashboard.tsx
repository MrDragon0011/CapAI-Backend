"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AnalysisResult, ActionType } from "../types/analysis";
import { UploadZone } from "./UploadZone";
import { ActionBadge, ACTION_CONFIG } from "./ActionBadge";
import { ScoreRing } from "./ScoreRing";
import { MetricCard } from "./MetricCard";

const ACTION_HEADINGS: Record<ActionType, { title: string; subtitle: string }> = {
  shooting: {
    title: "Power Shot Analysis",
    subtitle: "Biomechanical breakdown of your shooting mechanics against elite standards",
  },
  passing: {
    title: "Pass Mechanics",
    subtitle: "Arm extension, hand position and follow-through compared to elite technique",
  },
  swimming: {
    title: "Stroke Efficiency",
    subtitle: "Symmetry, catch position and body alignment across your swim cycle",
  },
  goalie: {
    title: "Goalkeeper Technique",
    subtitle: "Eggbeater power, blocking stance and lateral coverage metrics",
  },
};

export function Dashboard() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const action = result?.action as ActionType | undefined;
  const cfg = action ? ACTION_CONFIG[action] : null;
  const headings = action ? ACTION_HEADINGS[action] : null;

  return (
    <div className="min-h-screen bg-[#080810] text-white font-sans">
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99,60,255,0.18) 0%, transparent 70%)",
        }}
      />

      <div className="relative z-10 max-w-5xl mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <p className="text-[11px] uppercase tracking-[0.25em] text-white/30 mb-2">CapAI</p>
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-white to-white/50 bg-clip-text text-transparent">
            Movement Intelligence
          </h1>
          <p className="text-white/40 mt-2 text-sm">
            Upload a water polo clip to receive real-time biomechanical coaching feedback
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-10"
        >
          <UploadZone
            onResult={(data) => { setResult(data as AnalysisResult); setError(null); }}
            onError={(msg) => { setError(msg); setResult(null); }}
            loading={loading}
            setLoading={setLoading}
          />
        </motion.div>

        <AnimatePresence>
          {error && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mb-8 rounded-xl border border-rose-500/30 bg-rose-500/10 px-5 py-4 text-rose-300 text-sm"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence mode="wait">
          {result && action && cfg && headings && (
            <motion.div
              key={action}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className="relative rounded-3xl border border-white/[0.07] bg-white/[0.03] backdrop-blur-2xl overflow-hidden mb-8">
                <div
                  className="absolute inset-0 pointer-events-none opacity-30"
                  style={{
                    background: `radial-gradient(ellipse 60% 80% at 90% 50%, ${
                      action === "shooting"
                        ? "rgba(249,115,22,0.25)"
                        : action === "passing"
                        ? "rgba(56,189,248,0.25)"
                        : action === "swimming"
                        ? "rgba(45,212,191,0.25)"
                        : "rgba(167,139,250,0.25)"
                    } 0%, transparent 70%)`,
                  }}
                />

                <div className="relative z-10 flex flex-col sm:flex-row items-center sm:items-start gap-8 p-8">
                  <div className="shrink-0">
                    <ScoreRing score={result.overall_elite_score_pct} action={action} />
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="mb-3">
                      <ActionBadge action={action} label={result.label} />
                    </div>
                    <h2 className="text-2xl font-bold text-white mt-3 mb-1">{headings.title}</h2>
                    <p className="text-white/40 text-sm mb-5">{headings.subtitle}</p>

                    <div className="rounded-xl border border-white/[0.07] bg-white/[0.04] px-5 py-4">
                      <p className="text-[11px] uppercase tracking-widest text-white/30 mb-1">
                        Priority Focus
                      </p>
                      <p className="text-sm text-white/70 leading-relaxed">
                        {result.priority_focus}
                      </p>
                    </div>
                  </div>

                  <div className="shrink-0 flex flex-col items-center gap-1 text-center">
                    <span className="text-3xl font-bold text-white">{result.total_frames_analysed}</span>
                    <span className="text-[11px] uppercase tracking-widest text-white/30">frames</span>
                  </div>
                </div>
              </div>

              <div className="mb-4 flex items-center gap-3">
                <h3 className="text-sm font-semibold uppercase tracking-widest text-white/30">
                  Metric Breakdown
                </h3>
                <div className="flex-1 h-px bg-white/[0.06]" />
                <div className="flex gap-3 text-[11px]">
                  {(["elite", "below_elite", "above_elite"] as const).map((s) => (
                    <span
                      key={s}
                      className={`flex items-center gap-1.5 ${
                        s === "elite"
                          ? "text-emerald-400"
                          : s === "below_elite"
                          ? "text-amber-400"
                          : "text-rose-400"
                      }`}
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-current" />
                      {s === "elite" ? "Elite" : s === "below_elite" ? "Below" : "Above"}
                    </span>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {result.metrics.map((m, i) => (
                  <MetricCard key={m.metric} metric={m} index={i} />
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
