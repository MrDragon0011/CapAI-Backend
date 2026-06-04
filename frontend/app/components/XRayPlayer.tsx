"use client";

import { useRef, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const CONNECTIONS: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
  [24, 26], [26, 28],
];

export const ACTIONS: {
  key: string;
  label: string;
  accent: string;
  tags: string[];
  joints: Record<number, [number, number]>;
  highlights: number[];
}[] = [
  {
    key: "shooting",
    label: "Power Shot",
    accent: "#ff7c3a",
    tags: ["Elbow chain", "Hip drive", "Release pt."],
    joints: {
      0: [0.5, 0.09], 11: [0.37, 0.30], 12: [0.63, 0.26],
      13: [0.26, 0.46], 14: [0.72, 0.36], 15: [0.15, 0.56],
      16: [0.84, 0.26], 23: [0.42, 0.56], 24: [0.58, 0.54],
      25: [0.36, 0.72], 26: [0.62, 0.70], 27: [0.32, 0.87], 28: [0.65, 0.85],
    },
    highlights: [11, 12, 13, 14, 15, 16],
  },
  {
    key: "passing",
    label: "Outlet Pass",
    accent: "#22d3ee",
    tags: ["Release timing", "Wrist snap", "Shoulder"],
    joints: {
      0: [0.5, 0.10], 11: [0.40, 0.30], 12: [0.60, 0.30],
      13: [0.29, 0.44], 14: [0.71, 0.44], 15: [0.18, 0.42],
      16: [0.73, 0.28], 23: [0.44, 0.55], 24: [0.56, 0.55],
      25: [0.40, 0.71], 26: [0.58, 0.71], 27: [0.37, 0.85], 28: [0.61, 0.85],
    },
    highlights: [11, 12, 13, 14, 15, 16],
  },
  {
    key: "swimming",
    label: "Sprint Swim",
    accent: "#00d4ff",
    tags: ["Stroke rate", "Head pos.", "Kick amplitude"],
    joints: {
      0: [0.5, 0.12], 11: [0.34, 0.30], 12: [0.66, 0.30],
      13: [0.20, 0.26], 14: [0.80, 0.35], 15: [0.08, 0.24],
      16: [0.90, 0.38], 23: [0.40, 0.52], 24: [0.60, 0.50],
      25: [0.38, 0.67], 26: [0.63, 0.65], 27: [0.35, 0.81], 28: [0.66, 0.78],
    },
    highlights: [13, 14, 15, 16, 23, 24, 25, 26],
  },
  {
    key: "goalie",
    label: "Goalie Block",
    accent: "#a78bfa",
    tags: ["Vertical reach", "Hip height", "Reaction"],
    joints: {
      0: [0.5, 0.08], 11: [0.36, 0.28], 12: [0.64, 0.28],
      13: [0.22, 0.18], 14: [0.78, 0.18], 15: [0.12, 0.08],
      16: [0.88, 0.08], 23: [0.43, 0.55], 24: [0.57, 0.55],
      25: [0.34, 0.72], 26: [0.66, 0.72], 27: [0.28, 0.86], 28: [0.72, 0.86],
    },
    highlights: [13, 14, 15, 16, 23, 24, 25, 26],
  },
];

function SkeletonCanvas({
  action,
  revealed,
}: {
  action: (typeof ACTIONS)[0];
  revealed: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tickRef = useRef(0);
  const rafRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width;
    const H = canvas.height;
    const hl = new Set(action.highlights);

    const draw = () => {
      tickRef.current += 1;
      const t = tickRef.current * 0.04;
      ctx.clearRect(0, 0, W, H);

      const wobble = (idx: number): [number, number] => {
        const base = action.joints[idx];
        if (!base) return [0, 0];
        const amp = hl.has(idx) && revealed ? 0.014 : 0.006;
        return [
          base[0] * W + Math.sin(t + idx * 1.1) * W * amp,
          base[1] * H + Math.cos(t * 0.8 + idx * 0.8) * H * amp * 0.5,
        ];
      };

      for (const [a, b] of CONNECTIONS) {
        const pa = wobble(a);
        const pb = wobble(b);
        if (!action.joints[a] || !action.joints[b]) continue;
        const isHL = hl.has(a) && hl.has(b);
        ctx.beginPath();
        ctx.moveTo(...pa);
        ctx.lineTo(...pb);
        if (isHL && revealed) {
          ctx.strokeStyle = action.accent;
          ctx.lineWidth = 1.5;
          ctx.globalAlpha = 0.9;
        } else {
          ctx.strokeStyle = "rgba(255,255,255,0.25)";
          ctx.lineWidth = 1;
          ctx.globalAlpha = revealed ? 0.5 : 0.2;
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      for (const idx of Object.keys(action.joints).map(Number)) {
        const [px, py] = wobble(idx);
        const isHL = hl.has(idx);
        ctx.beginPath();
        ctx.arc(px, py, isHL && revealed ? 3.5 : 2, 0, Math.PI * 2);
        ctx.fillStyle = isHL && revealed ? action.accent : "rgba(255,255,255,0.35)";
        ctx.globalAlpha = revealed ? 1 : 0.2;
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [action, revealed]);

  return <canvas ref={canvasRef} width={160} height={200} className="block" />;
}

export function XRayPlayer({ action }: { action: (typeof ACTIONS)[0] }) {
  const [hovered, setHovered] = useState(false);

  return (
    <motion.div
      className="relative overflow-hidden cursor-pointer"
      style={{
        background: "var(--surface)",
        border: `1px solid ${hovered ? action.accent + "50" : "var(--border)"}`,
        transition: "border-color 0.2s",
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div
        className="px-4 pt-4 pb-0 flex items-center justify-between"
        style={{ borderBottom: "1px solid var(--border)" }}
      >
        <span
          className="text-[10px] font-bold uppercase tracking-widest"
          style={{ color: action.accent }}
        >
          {action.label}
        </span>
        <AnimatePresence>
          {hovered && (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="font-mono text-[9px] text-white/30"
            >
              SKELETON ON
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      <div className="grid-bg-fine relative flex items-center justify-center py-4"
        style={{ height: 200 }}>
        <div className="relative" style={{ width: 160, height: 200 }}>
          <SkeletonCanvas action={action} revealed={hovered} />
        </div>
      </div>

      <div style={{ borderTop: "1px solid var(--border)" }} className="px-4 py-3 space-y-1">
        {action.tags.map((tag) => (
          <div key={tag} className="flex items-center gap-2">
            <div className="w-px h-3" style={{ background: hovered ? action.accent : "var(--border-strong)" }} />
            <span className="text-[10px] text-white/40 font-mono">{tag}</span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}
