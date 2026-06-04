"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const SHOT_SKELETON_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
  [24, 26], [26, 28],
];

const ACTIONS: {
  key: string;
  label: string;
  color: string;
  glowColor: string;
  description: string;
  stat: string;
  joints: Record<number, [number, number]>;
  highlights: number[];
}[] = [
  {
    key: "shooting",
    label: "Power Shot",
    color: "#f97316",
    glowColor: "rgba(249,115,22,0.3)",
    description: "Elbow angle, shoulder rotation & hip drive across throwing chain",
    stat: "Arm velocity · Hip drive · Release point",
    joints: {
      0: [0.5, 0.09], 11: [0.37, 0.3], 12: [0.63, 0.26],
      13: [0.26, 0.46], 14: [0.72, 0.36], 15: [0.15, 0.56],
      16: [0.84, 0.26], 23: [0.42, 0.56], 24: [0.58, 0.54],
      25: [0.36, 0.72], 26: [0.62, 0.70], 27: [0.32, 0.87], 28: [0.65, 0.85],
    },
    highlights: [11, 12, 13, 14, 15, 16],
  },
  {
    key: "passing",
    label: "Outlet Pass",
    color: "#22d3ee",
    glowColor: "rgba(34,211,238,0.3)",
    description: "Two-handed release, torso rotation & weight transfer",
    stat: "Release timing · Wrist snap · Shoulder alignment",
    joints: {
      0: [0.5, 0.10], 11: [0.40, 0.30], 12: [0.60, 0.30],
      13: [0.29, 0.44], 14: [0.71, 0.44], 15: [0.18, 0.42],
      16: [0.73, 0.28], 23: [0.44, 0.55], 24: [0.56, 0.55],
      25: [0.40, 0.71], 26: [0.58, 0.71], 27: [0.37, 0.85], 28: [0.61, 0.85],
    },
    highlights: [11, 12, 15, 16, 13, 14],
  },
  {
    key: "swimming",
    label: "Sprint Swim",
    color: "#00c8b4",
    glowColor: "rgba(0,200,180,0.3)",
    description: "Stroke efficiency, head position & leg kick cadence",
    stat: "Stroke rate · Head position · Kick amplitude",
    joints: {
      0: [0.5, 0.12], 11: [0.34, 0.30], 12: [0.66, 0.30],
      13: [0.20, 0.26], 14: [0.80, 0.35], 15: [0.08, 0.24],
      16: [0.90, 0.38], 23: [0.40, 0.52], 24: [0.60, 0.50],
      25: [0.38, 0.67], 26: [0.63, 0.65], 27: [0.35, 0.81], 28: [0.66, 0.78],
    },
    highlights: [13, 14, 15, 16, 0, 25, 26],
  },
  {
    key: "goalie",
    label: "Goalie Block",
    color: "#a78bfa",
    glowColor: "rgba(167,139,250,0.3)",
    description: "Vertical reach, hip elevation & eggbeater power output",
    stat: "Vertical reach · Hip height · Reaction time",
    joints: {
      0: [0.5, 0.08], 11: [0.36, 0.28], 12: [0.64, 0.28],
      13: [0.22, 0.18], 14: [0.78, 0.18], 15: [0.12, 0.08],
      16: [0.88, 0.08], 23: [0.43, 0.55], 24: [0.57, 0.55],
      25: [0.34, 0.72], 26: [0.66, 0.72], 27: [0.28, 0.86], 28: [0.72, 0.86],
    },
    highlights: [13, 14, 15, 16, 23, 24, 25, 26],
  },
];

function PlayerFigure({
  action,
  revealed,
  size = 160,
}: {
  action: (typeof ACTIONS)[0];
  revealed: boolean;
  size?: number;
}) {
  const w = size * 0.65;
  const h = size;
  const hl = new Set(action.highlights ?? []);

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="overflow-visible">
      <defs>
        <filter id={`glow-${action.key}`}>
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <g transform={`translate(${(size - w) / 2}, ${(size - h) / 2})`} filter={`url(#glow-${action.key})`}>
        {SHOT_SKELETON_CONNECTIONS.map(([a, b]) => {
          const ja = action.joints[a];
          const jb = action.joints[b];
          if (!ja || !jb) return null;
          const isHL = hl.has(a) && hl.has(b);
          return (
            <motion.line
              key={`${a}-${b}`}
              x1={ja[0] * w} y1={ja[1] * h}
              x2={jb[0] * w} y2={jb[1] * h}
              stroke={revealed && isHL ? action.color : revealed ? "rgba(255,255,255,0.35)" : "transparent"}
              strokeWidth={revealed && isHL ? 2 : 1}
              strokeLinecap="round"
              animate={{
                stroke: revealed && isHL ? action.color : revealed ? "rgba(255,255,255,0.3)" : "transparent",
                strokeWidth: revealed && isHL ? 2 : 1,
              }}
              transition={{ duration: 0.35 }}
            />
          );
        })}
        {Object.entries(action.joints).map(([idx, [jx, jy]]) => {
          const isHL = hl.has(Number(idx));
          return (
            <motion.circle
              key={idx}
              cx={jx * w}
              cy={jy * h}
              r={isHL && revealed ? 4.5 : 3}
              fill={revealed && isHL ? action.color : revealed ? "rgba(255,255,255,0.4)" : "transparent"}
              animate={{
                fill: revealed && isHL ? action.color : revealed ? "rgba(255,255,255,0.35)" : "transparent",
                r: isHL && revealed ? 4.5 : 3,
              }}
              transition={{ duration: 0.3, delay: isHL ? 0.05 * Number(idx) : 0 }}
            />
          );
        })}
      </g>
    </svg>
  );
}

export function XRayPlayer({ action }: { action: (typeof ACTIONS)[0] }) {
  const [hovered, setHovered] = useState(false);

  return (
    <motion.div
      className="relative rounded-xl overflow-hidden cursor-pointer"
      style={{
        background: `linear-gradient(135deg, rgba(2,11,20,0.9) 0%, rgba(6,45,80,0.6) 100%)`,
        border: `1px solid ${hovered ? action.color + "40" : "rgba(255,255,255,0.08)"}`,
      }}
      animate={{ borderColor: hovered ? action.color + "40" : "rgba(255,255,255,0.08)" }}
      transition={{ duration: 0.25 }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="p-5 flex flex-col items-center gap-4">
        <div className="relative">
          <motion.div
            className="absolute inset-0 rounded-full blur-2xl"
            style={{ background: action.glowColor }}
            animate={{ opacity: hovered ? 1 : 0 }}
            transition={{ duration: 0.4 }}
          />
          <motion.div animate={{ scale: hovered ? 1.05 : 1 }} transition={{ duration: 0.4, ease: "easeOut" }}>
            <PlayerFigure action={action} revealed={hovered} size={150} />
          </motion.div>
        </div>

        <div className="text-center w-full">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-bold uppercase tracking-widest" style={{ color: action.color }}>
              {action.label}
            </span>
            <AnimatePresence>
              {hovered && (
                <motion.span
                  initial={{ opacity: 0, x: 8 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0 }}
                  className="text-[9px] uppercase tracking-widest text-white/30"
                >
                  X-RAY ON
                </motion.span>
              )}
            </AnimatePresence>
          </div>
          <p className="text-[11px] text-white/40 leading-snug">{action.description}</p>

          <AnimatePresence>
            {hovered && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="overflow-hidden"
              >
                <div className="mt-3 pt-3 border-t border-white/[0.07]">
                  <p className="text-[10px] text-white/25 leading-relaxed">{action.stat}</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  );
}

export { ACTIONS };
