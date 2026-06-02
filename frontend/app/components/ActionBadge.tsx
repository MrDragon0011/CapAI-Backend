"use client";

import { motion } from "framer-motion";
import { ActionType } from "../types/analysis";

const ACTION_CONFIG: Record<ActionType, { color: string; glow: string; icon: string }> = {
  shooting: {
    color: "from-orange-500 to-red-500",
    glow: "shadow-orange-500/40",
    icon: "🎯",
  },
  passing: {
    color: "from-blue-400 to-cyan-400",
    glow: "shadow-blue-500/40",
    icon: "🤾",
  },
  swimming: {
    color: "from-teal-400 to-emerald-400",
    glow: "shadow-teal-500/40",
    icon: "🌊",
  },
  goalie: {
    color: "from-violet-500 to-purple-500",
    glow: "shadow-violet-500/40",
    icon: "🧤",
  },
};

export function ActionBadge({ action, label }: { action: ActionType; label: string }) {
  const cfg = ACTION_CONFIG[action];
  return (
    <motion.div
      key={action}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: "spring", stiffness: 300, damping: 20 }}
      className={`inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r ${cfg.color} shadow-lg ${cfg.glow} text-white font-semibold text-sm tracking-wide`}
    >
      <span>{cfg.icon}</span>
      <span>{label}</span>
    </motion.div>
  );
}

export { ACTION_CONFIG };
