"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AnalysisResult } from "../types/analysis";

interface Props {
  onResult: (data: AnalysisResult, videoUrl: string) => void;
  onError: (msg: string) => void;
  hasResult: boolean;
}

const STAGES = [
  { label: "Uploading video", detail: "Transferring to analysis server", icon: "↑", target: 15 },
  { label: "Extracting landmarks", detail: "Mapping body position across all frames", icon: "◈", target: 45 },
  { label: "Classifying movement", detail: "Identifying action type from motion patterns", icon: "⬡", target: 65 },
  { label: "Biomechanical analysis", detail: "Benchmarking against elite standards", icon: "◎", target: 90 },
  { label: "Finalising insights", detail: "Preparing coaching feedback", icon: "✦", target: 99 },
];

const SPEEDS = [0.8, 0.4, 0.35, 0.25, 0.05];

export function UploadZone({ onResult, onError, hasResult }: Props) {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const progressRef = useRef(0);
  const rafRef = useRef<number>(0);
  const doneRef = useRef(false);
  const resultRef = useRef<{ data: AnalysisResult; videoUrl: string } | null>(null);

  const currentStage = STAGES.findIndex((s) => progressRef.current < s.target);
  const activeStage = currentStage === -1 ? STAGES.length - 1 : currentStage;

  const advanceProgress = useCallback(() => {
    const stage = STAGES.findIndex((s) => progressRef.current < s.target);
    if (stage === -1) {
      if (doneRef.current && resultRef.current) {
        setProgress(100);
        setTimeout(() => {
          setLoading(false);
          onResult(resultRef.current!.data, resultRef.current!.videoUrl);
          progressRef.current = 0;
          setProgress(0);
          doneRef.current = false;
          resultRef.current = null;
        }, 400);
      } else {
        rafRef.current = requestAnimationFrame(advanceProgress);
      }
      return;
    }
    progressRef.current = Math.min(progressRef.current + SPEEDS[stage], STAGES[stage].target);
    setProgress(Math.floor(progressRef.current));
    rafRef.current = requestAnimationFrame(advanceProgress);
  }, [onResult]);

  const submit = useCallback(
    async (file: File) => {
      const videoUrl = URL.createObjectURL(file);
      doneRef.current = false;
      resultRef.current = null;
      progressRef.current = 0;
      setProgress(0);
      setLoading(true);
      rafRef.current = requestAnimationFrame(advanceProgress);

      const form = new FormData();
      form.append("file", file);
      try {
        const res = await fetch("http://localhost:8000/analyze", { method: "POST", body: form });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          cancelAnimationFrame(rafRef.current);
          setLoading(false);
          progressRef.current = 0;
          setProgress(0);
          onError(err.detail ?? "Request failed");
        } else {
          const data = await res.json();
          resultRef.current = { data, videoUrl };
          doneRef.current = true;
        }
      } catch {
        cancelAnimationFrame(rafRef.current);
        setLoading(false);
        progressRef.current = 0;
        setProgress(0);
        onError("Could not reach the backend. Make sure it is running on port 8000.");
      }
    },
    [advanceProgress, onError]
  );

  useEffect(() => () => cancelAnimationFrame(rafRef.current), []);

  const handleFiles = (files: FileList | null) => {
    if (files?.[0]) submit(files[0]);
  };

  if (loading) {
    return (
      <div className="w-full rounded-2xl border border-white/[0.07] bg-white/[0.03] backdrop-blur-xl overflow-hidden">
        <div className="px-8 py-6">
          <div className="flex items-center justify-between mb-5">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeStage}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                className="flex items-center gap-3"
              >
                <motion.span
                  animate={{ rotate: [0, 180, 360] }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  className="text-xl text-violet-400"
                >
                  {STAGES[activeStage]?.icon}
                </motion.span>
                <div>
                  <p className="text-sm font-semibold text-white">{STAGES[activeStage]?.label}</p>
                  <p className="text-xs text-white/35 mt-0.5">{STAGES[activeStage]?.detail}</p>
                </div>
              </motion.div>
            </AnimatePresence>
            <span className="text-2xl font-bold text-white tabular-nums">{progress}%</span>
          </div>

          <div className="relative h-1.5 rounded-full bg-white/[0.06] overflow-hidden mb-6">
            <motion.div
              className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-violet-500 to-indigo-400"
              style={{ width: `${progress}%` }}
              transition={{ duration: 0.1 }}
            />
            <motion.div
              className="absolute inset-y-0 w-20 rounded-full bg-gradient-to-r from-transparent via-white/30 to-transparent"
              animate={{ x: ["-100%", "600%"] }}
              transition={{ duration: 1.8, repeat: Infinity, ease: "linear" }}
            />
          </div>

          <div className="grid grid-cols-5 gap-2">
            {STAGES.map((s, i) => (
              <div key={s.label} className="flex flex-col items-center gap-1.5">
                <div
                  className={`w-7 h-7 rounded-full flex items-center justify-center text-xs transition-all duration-300 ${
                    i < activeStage
                      ? "bg-violet-500/30 text-violet-300 border border-violet-500/40"
                      : i === activeStage
                      ? "bg-violet-500/20 text-violet-300 border border-violet-400/60 shadow-lg shadow-violet-500/20"
                      : "bg-white/[0.04] text-white/20 border border-white/[0.06]"
                  }`}
                >
                  {i < activeStage ? "✓" : s.icon}
                </div>
                <span className={`text-[9px] text-center leading-tight ${i <= activeStage ? "text-white/40" : "text-white/15"}`}>
                  {s.label.split(" ")[0]}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <motion.label
      htmlFor="video-upload"
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
      animate={{
        borderColor: dragging ? "rgba(139,92,246,0.6)" : "rgba(255,255,255,0.07)",
        backgroundColor: dragging ? "rgba(139,92,246,0.06)" : "rgba(255,255,255,0.02)",
      }}
      className={`relative flex flex-col items-center justify-center gap-4 w-full rounded-2xl border-2 border-dashed backdrop-blur-xl cursor-pointer transition-all ${
        hasResult ? "py-5 px-8" : "py-14 px-8"
      }`}
    >
      <input
        id="video-upload"
        type="file"
        accept="video/*"
        className="sr-only"
        onChange={(e) => handleFiles(e.target.files)}
      />

      {hasResult ? (
        <div className="flex items-center gap-3 text-sm">
          <span className="text-white/30">↑</span>
          <span className="text-white/50">Upload a new clip to re-analyse</span>
        </div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center gap-4 text-center"
        >
          <motion.div
            animate={{ scale: dragging ? 1.1 : 1 }}
            className="w-16 h-16 rounded-2xl bg-white/[0.05] border border-white/[0.08] flex items-center justify-center text-3xl"
          >
            🎬
          </motion.div>
          <div>
            <p className="text-white/80 font-semibold text-lg">Drop a water polo clip</p>
            <p className="text-white/30 text-sm mt-1">MP4 · MOV · AVI — drag anywhere or click to browse</p>
          </div>
          <div className="flex gap-6 text-[11px] text-white/20 uppercase tracking-widest mt-1">
            <span>Pose Extraction</span>
            <span>·</span>
            <span>Action Classification</span>
            <span>·</span>
            <span>Elite Benchmarking</span>
          </div>
        </motion.div>
      )}
    </motion.label>
  );
}
