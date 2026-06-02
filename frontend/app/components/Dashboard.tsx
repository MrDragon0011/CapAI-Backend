"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AnalysisResult, ActionType } from "../types/analysis";
import { UploadZone } from "./UploadZone";
import { ScoreRing } from "./ScoreRing";
import { MetricCard } from "./MetricCard";
import { SkeletonOverlay } from "./SkeletonOverlay";
import { ActionInsightPanel } from "./ActionInsightPanel";

export function Dashboard() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const action = result?.action as ActionType | undefined;

  return (
    <div className="min-h-screen bg-[#080810] text-white font-sans">
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99,60,255,0.15) 0%, transparent 70%)",
        }}
      />

      <div className="relative z-10 max-w-6xl mx-auto px-5 py-10">
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <p className="text-[11px] uppercase tracking-[0.25em] text-white/25 mb-1.5">CapAI</p>
          <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-white to-white/40 bg-clip-text text-transparent">
            Movement Intelligence
          </h1>
          <p className="text-white/35 mt-1.5 text-sm">
            Upload a water polo clip for real-time biomechanical coaching feedback
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.08 }}
          className="mb-6"
        >
          <UploadZone
            onResult={(data, url) => {
              setResult(data);
              setVideoUrl(url);
              setError(null);
            }}
            onError={(msg) => { setError(msg); setResult(null); setVideoUrl(null); }}
            hasResult={!!result}
          />
        </motion.div>

        <AnimatePresence>
          {error && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mb-6 rounded-xl border border-rose-500/25 bg-rose-500/08 px-5 py-4 text-rose-300 text-sm"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence mode="wait">
          {result && action && videoUrl && (
            <motion.div
              key={action}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.35 }}
            >
              <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-5 mb-6">
                <SkeletonOverlay
                  videoUrl={videoUrl}
                  landmarks={result.landmarks}
                  action={action}
                />
                <ActionInsightPanel result={result} />
              </div>

              <div className="flex items-center gap-3 mb-4">
                <h3 className="text-[11px] font-semibold uppercase tracking-widest text-white/25">
                  Full Metric Breakdown
                </h3>
                <div className="flex-1 h-px bg-white/[0.05]" />
                <div className="flex items-center gap-4 text-[10px]">
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
                      {s === "elite" ? "Elite" : s === "below_elite" ? "Below Elite" : "Above Elite"}
                    </span>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
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
