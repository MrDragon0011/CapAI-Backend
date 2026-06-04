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
    <div className="min-h-screen text-white" style={{ background: "var(--bg)" }}>
      <div className="max-w-6xl mx-auto px-5 pt-24 pb-16">
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
              className="mb-6 px-5 py-4 text-rose-300 text-sm font-mono"
              style={{ border: "1px solid rgba(248,113,113,0.3)", background: "rgba(248,113,113,0.06)" }}
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

              <div className="flex items-center gap-4 mb-4 pt-2">
                <div className="w-3 h-px" style={{ background: "var(--accent)" }} />
                <h3 className="font-mono text-[10px] uppercase tracking-[0.25em] text-white/25">
                  Full Metric Breakdown
                </h3>
                <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
                <div className="flex items-center gap-4 font-mono text-[10px]">
                  {(["elite", "below_elite", "above_elite"] as const).map((s) => (
                    <span
                      key={s}
                      className={`flex items-center gap-1.5 ${
                        s === "elite" ? "text-emerald-400" : s === "below_elite" ? "text-amber-400" : "text-rose-400"
                      }`}
                    >
                      <span className="w-1.5 h-1.5 bg-current" />
                      {s === "elite" ? "Elite" : s === "below_elite" ? "Below" : "Above"}
                    </span>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
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
