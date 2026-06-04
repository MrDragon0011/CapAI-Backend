"use client";

import { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { ActionType, LandmarkSequence } from "../types/analysis";

const CONNECTIONS: [number, number][] = [
  [11, 12],
  [11, 13], [13, 15],
  [12, 14], [14, 16],
  [15, 17], [15, 19], [17, 19],
  [16, 18], [16, 20], [18, 20],
  [11, 23], [12, 24],
  [23, 24],
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
];

const ACTION_FOCUS: Record<ActionType, { joints: Set<number>; connections: Set<string> }> = {
  shooting: {
    joints: new Set([11, 12, 13, 14, 15, 16, 23, 24]),
    connections: new Set(["11,12", "12,14", "14,16", "11,23", "12,24", "23,24"]),
  },
  passing: {
    joints: new Set([11, 12, 13, 14, 15, 16]),
    connections: new Set(["11,12", "11,13", "13,15", "12,14", "14,16"]),
  },
  swimming: {
    joints: new Set([11, 12, 13, 14, 15, 16, 23, 24, 25, 26]),
    connections: new Set(["11,12", "11,13", "13,15", "12,14", "14,16", "23,24", "23,25", "24,26"]),
  },
  goalie: {
    joints: new Set([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]),
    connections: new Set(["11,12", "11,13", "12,14", "23,24", "23,25", "25,27", "24,26", "26,28"]),
  },
};

const ACTION_COLOR: Record<ActionType, string> = {
  shooting: "255,140,50",
  passing: "56,189,248",
  swimming: "45,212,191",
  goalie: "167,139,250",
};

interface Props {
  videoUrl: string;
  landmarks: LandmarkSequence;
  action: ActionType;
}

function findFrame(frames: LandmarkSequence["frames"], currentMs: number) {
  if (!frames.length) return null;
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (frames[mid].t <= currentMs) lo = mid;
    else hi = mid - 1;
  }
  return frames[lo];
}

export function SkeletonOverlay({ videoUrl, landmarks, action }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const color = ACTION_COLOR[action];
  const focus = ACTION_FOCUS[action];

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d")!;

    const draw = () => {
      const w = video.offsetWidth;
      const h = video.offsetHeight;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      ctx.clearRect(0, 0, w, h);

      if (!landmarks.frames.length || video.paused && video.currentTime === 0) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const currentMs = video.currentTime * 1000;
      const frame = findFrame(landmarks.frames, currentMs);
      if (!frame) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const lm = frame.lm;

      for (const [a, b] of CONNECTIONS) {
        if (a >= lm.length || b >= lm.length) continue;
        const pa = lm[a];
        const pb = lm[b];
        const vis = Math.min(pa[2], pb[2]);
        if (vis < 0.3) continue;
        const key = `${a},${b}`;
        const isFocus = focus.connections.has(key);
        ctx.beginPath();
        ctx.moveTo(pa[0] * w, pa[1] * h);
        ctx.lineTo(pb[0] * w, pb[1] * h);
        if (isFocus) {
          ctx.strokeStyle = `rgba(${color},${0.5 + vis * 0.4})`;
          ctx.lineWidth = 2.5;
          ctx.shadowColor = `rgba(${color},0.6)`;
          ctx.shadowBlur = 8;
        } else {
          ctx.strokeStyle = `rgba(255,255,255,${0.08 + vis * 0.1})`;
          ctx.lineWidth = 1;
          ctx.shadowBlur = 0;
        }
        ctx.stroke();
      }

      ctx.shadowBlur = 0;

      for (let i = 0; i < lm.length; i++) {
        const p = lm[i];
        if (!p || p[2] < 0.3) continue;
        const isFocus = focus.joints.has(i);
        const x = p[0] * w;
        const y = p[1] * h;

        if (isFocus) {
          ctx.beginPath();
          ctx.arc(x, y, 6, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${color},0.15)`;
          ctx.fill();
          ctx.beginPath();
          ctx.arc(x, y, 3.5, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${color},${0.7 + p[2] * 0.3})`;
          ctx.shadowColor = `rgba(${color},0.8)`;
          ctx.shadowBlur = 10;
          ctx.fill();
          ctx.shadowBlur = 0;
        } else {
          ctx.beginPath();
          ctx.arc(x, y, 2, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255,255,255,${0.15 + p[2] * 0.15})`;
          ctx.fill();
        }
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [landmarks, action, color, focus]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
      className="relative w-full overflow-hidden bg-black"
      style={{ border: "1px solid var(--border)" }}
    >
      <video
        ref={videoRef}
        src={videoUrl}
        className="w-full block"
        controls
        loop
        playsInline
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 pointer-events-none"
        style={{ width: "100%", height: "100%" }}
      />
      <div
        className="absolute top-0 left-0 flex items-center gap-2 px-3 py-1.5"
        style={{
          background: "rgba(0,0,0,0.85)",
          borderBottom: "1px solid var(--border)",
          borderRight: "1px solid var(--border)",
        }}
      >
        <span
          className="w-1.5 h-1.5 animate-pulse"
          style={{ background: `rgba(${ACTION_COLOR[action]},1)` }}
        />
        <span className="font-mono text-[9px] uppercase tracking-widest text-white/50">
          Skeleton Overlay
        </span>
      </div>
    </motion.div>
  );
}
