"use client";

import { motion } from "framer-motion";
import { ActionType } from "../types/analysis";
import { ACTION_CONFIG } from "./ActionBadge";

const SIZE = 140;
const STROKE = 10;
const R = (SIZE - STROKE) / 2;
const CIRC = 2 * Math.PI * R;

export function ScoreRing({ score, action }: { score: number; action: ActionType }) {
  const offset = CIRC - (score / 100) * CIRC;
  const color = action === "shooting"
    ? "#f97316"
    : action === "passing"
    ? "#38bdf8"
    : action === "swimming"
    ? "#2dd4bf"
    : "#a78bfa";

  return (
    <div className="relative flex items-center justify-center" style={{ width: SIZE, height: SIZE }}>
      <svg width={SIZE} height={SIZE} className="-rotate-90">
        <circle
          cx={SIZE / 2}
          cy={SIZE / 2}
          r={R}
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={STROKE}
        />
        <motion.circle
          cx={SIZE / 2}
          cy={SIZE / 2}
          r={R}
          fill="none"
          stroke={color}
          strokeWidth={STROKE}
          strokeLinecap="round"
          strokeDasharray={CIRC}
          initial={{ strokeDashoffset: CIRC }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.2, ease: "easeOut" }}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <motion.span
          key={score}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="text-3xl font-bold text-white"
        >
          {score}
          <span className="text-base font-normal text-white/50">%</span>
        </motion.span>
        <span className="text-[10px] uppercase tracking-widest text-white/40 mt-0.5">Elite</span>
      </div>
    </div>
  );
}
