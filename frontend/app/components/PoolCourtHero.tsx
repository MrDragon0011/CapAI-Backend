"use client";

import { useEffect, useRef, useState } from "react";
import { motion, useMotionValue, useSpring } from "framer-motion";

const SKELETON_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
  [24, 26], [26, 28],
];

const PLAYER_POSES = [
  {
    label: "Power Shot",
    color: "#f97316",
    joints: {
      0: [0.5, 0.12], 11: [0.38, 0.32], 12: [0.62, 0.28],
      13: [0.28, 0.48], 14: [0.72, 0.38], 15: [0.18, 0.58],
      16: [0.82, 0.28], 23: [0.42, 0.58], 24: [0.58, 0.56],
      25: [0.36, 0.74], 26: [0.62, 0.72], 27: [0.32, 0.88], 28: [0.65, 0.86],
    },
  },
  {
    label: "Passing",
    color: "#22d3ee",
    joints: {
      0: [0.5, 0.1], 11: [0.4, 0.3], 12: [0.6, 0.3],
      13: [0.3, 0.45], 14: [0.7, 0.45], 15: [0.2, 0.42],
      16: [0.72, 0.28], 23: [0.44, 0.56], 24: [0.56, 0.56],
      25: [0.4, 0.72], 26: [0.58, 0.72], 27: [0.36, 0.86], 28: [0.6, 0.86],
    },
  },
  {
    label: "Swimming",
    color: "#00c8b4",
    joints: {
      0: [0.5, 0.12], 11: [0.35, 0.3], 12: [0.65, 0.3],
      13: [0.22, 0.26], 14: [0.78, 0.35], 15: [0.1, 0.24],
      16: [0.88, 0.38], 23: [0.4, 0.52], 24: [0.6, 0.5],
      25: [0.38, 0.68], 26: [0.62, 0.65], 27: [0.35, 0.82], 28: [0.65, 0.78],
    },
  },
];

function SkeletonFigure({
  pose,
  x,
  y,
  scale = 1,
  opacity = 1,
  animated = false,
}: {
  pose: (typeof PLAYER_POSES)[0];
  x: number;
  y: number;
  scale?: number;
  opacity?: number;
  animated?: boolean;
}) {
  const w = 80 * scale;
  const h = 100 * scale;

  return (
    <motion.g
      transform={`translate(${x - w / 2}, ${y - h / 2})`}
      animate={animated ? { y: [0, -3, 0] } : {}}
      transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
    >
      {SKELETON_CONNECTIONS.map(([a, b]) => {
        const ja = pose.joints[a as keyof typeof pose.joints];
        const jb = pose.joints[b as keyof typeof pose.joints];
        if (!ja || !jb) return null;
        return (
          <line
            key={`${a}-${b}`}
            x1={ja[0] * w} y1={ja[1] * h}
            x2={jb[0] * w} y2={jb[1] * h}
            stroke={pose.color}
            strokeWidth={1.5}
            strokeLinecap="round"
            opacity={opacity * 0.8}
          />
        );
      })}
      {Object.entries(pose.joints).map(([idx, [jx, jy]]) => (
        <motion.circle
          key={idx}
          cx={jx * w}
          cy={jy * h}
          r={3}
          fill={pose.color}
          opacity={opacity}
          animate={animated ? { r: [3, 4, 3], opacity: [opacity, 1, opacity] } : {}}
          transition={{ duration: 1.8, repeat: Infinity, delay: Number(idx) * 0.05 }}
        />
      ))}
    </motion.g>
  );
}

export function PoolCourtHero() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [mousePos, setMousePos] = useState({ x: 0.5, y: 0.5 });
  const [showLens, setShowLens] = useState(false);
  const lensX = useMotionValue(0);
  const lensY = useMotionValue(0);
  const springX = useSpring(lensX, { stiffness: 200, damping: 25 });
  const springY = useSpring(lensY, { stiffness: 200, damping: 25 });

  const handleMouseMove = (e: React.MouseEvent) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const rx = (e.clientX - rect.left) / rect.width;
    const ry = (e.clientY - rect.top) / rect.height;
    setMousePos({ x: rx, y: ry });
    lensX.set(e.clientX - rect.left);
    lensY.set(e.clientY - rect.top);
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full overflow-hidden rounded-2xl cursor-crosshair select-none"
      style={{ height: 420, background: "linear-gradient(180deg, #041e35 0%, #062d50 35%, #083d68 65%, #041e35 100%)" }}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setShowLens(true)}
      onMouseLeave={() => setShowLens(false)}
    >
      <svg
        className="absolute inset-0 w-full h-full"
        viewBox="0 0 800 420"
        preserveAspectRatio="xMidYMid slice"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <linearGradient id="pool-water" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#041e35" />
            <stop offset="50%" stopColor="#083d68" />
            <stop offset="100%" stopColor="#041e35" />
          </linearGradient>
          <radialGradient id="lens-light" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(0,200,180,0.12)" />
            <stop offset="100%" stopColor="transparent" />
          </radialGradient>
          <filter id="xray-glow">
            <feGaussianBlur stdDeviation="1.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <clipPath id="pool-clip">
            <rect x="60" y="40" width="680" height="340" rx="8" />
          </clipPath>
        </defs>

        <rect x="60" y="40" width="680" height="340" rx="8" fill="#062d50" />

        {[1, 2, 3, 4, 5, 6, 7].map((i) => (
          <line
            key={i}
            x1={60 + i * 85}
            y1="40"
            x2={60 + i * 85}
            y2="380"
            stroke="rgba(255,255,255,0.07)"
            strokeWidth="1"
            strokeDasharray="4 6"
          />
        ))}

        <line x1="60" y1="210" x2="740" y2="210" stroke="rgba(255,255,255,0.18)" strokeWidth="1.5" />
        <line x1="60" y1="100" x2="740" y2="100" stroke="rgba(255,60,60,0.35)" strokeWidth="2" strokeDasharray="8 5" />
        <line x1="60" y1="320" x2="740" y2="320" stroke="rgba(255,60,60,0.35)" strokeWidth="2" strokeDasharray="8 5" />

        <rect x="60" y="40" width="100" height="340" fill="rgba(255,100,100,0.04)" />
        <rect x="640" y="40" width="100" height="340" fill="rgba(255,100,100,0.04)" />

        <rect x="60" y="150" width="30" height="120" rx="4" fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth="1.5" />
        <rect x="710" y="150" width="30" height="120" rx="4" fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth="1.5" />

        <circle cx="400" cy="210" r="35" fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth="1.5" />

        <rect x="60" y="40" width="680" height="340" rx="8" fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth="2" />

        <text x="95" y="36" fill="rgba(255,255,255,0.2)" fontSize="9" textAnchor="middle" fontFamily="monospace">GOAL</text>
        <text x="705" y="36" fill="rgba(255,255,255,0.2)" fontSize="9" textAnchor="middle" fontFamily="monospace">GOAL</text>
        <text x="400" y="36" fill="rgba(255,255,255,0.15)" fontSize="9" textAnchor="middle" fontFamily="monospace">CENTER</text>
        <text x="54" y="105" fill="rgba(255,80,80,0.5)" fontSize="8" textAnchor="end" fontFamily="monospace">5m</text>
        <text x="54" y="325" fill="rgba(255,80,80,0.5)" fontSize="8" textAnchor="end" fontFamily="monospace">5m</text>

        <g filter="url(#xray-glow)" clipPath="url(#pool-clip)">
          <SkeletonFigure pose={PLAYER_POSES[0]} x={160} y={160} scale={1.1} opacity={0.85} animated />
          <SkeletonFigure pose={PLAYER_POSES[1]} x={400} y={200} scale={1.0} opacity={0.75} animated />
          <SkeletonFigure pose={PLAYER_POSES[2]} x={620} y={240} scale={1.05} opacity={0.80} animated />
          <SkeletonFigure pose={PLAYER_POSES[0]} x={280} y={310} scale={0.85} opacity={0.5} />
          <SkeletonFigure pose={PLAYER_POSES[1]} x={520} y={130} scale={0.8} opacity={0.45} />
        </g>

        <motion.rect
          x="60" y="40" width="680" height="2"
          fill="rgba(0,200,180,0.5)"
          animate={{ opacity: [0, 0.6, 0], x: [60, 60] }}
          transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
        />
      </svg>

      {showLens && (
        <motion.div
          className="absolute pointer-events-none magnifier-ring rounded-full overflow-hidden"
          style={{
            width: 140,
            height: 140,
            x: springX,
            y: springY,
            translateX: "-50%",
            translateY: "-50%",
            backgroundImage: `radial-gradient(circle, rgba(0,200,180,0.05) 0%, transparent 70%)`,
            border: "1.5px solid rgba(0,200,180,0.5)",
            backdropFilter: "brightness(1.4) saturate(1.2)",
            zIndex: 10,
          }}
        >
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-px h-full bg-teal-400/20 absolute" />
            <div className="h-px w-full bg-teal-400/20 absolute" />
            <div className="w-8 h-8 rounded-full border border-teal-400/40" />
          </div>
        </motion.div>
      )}

      <div className="absolute inset-0 pointer-events-none"
        style={{ background: "radial-gradient(ellipse 60% 40% at 50% 50%, transparent 30%, rgba(2,11,20,0.6) 100%)" }} />

      <div className="absolute bottom-5 left-6 right-6 flex items-end justify-between pointer-events-none">
        {PLAYER_POSES.map((p) => (
          <div key={p.label} className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full" style={{ background: p.color, boxShadow: `0 0 6px ${p.color}` }} />
            <span className="text-[10px] text-white/40 uppercase tracking-wider">{p.label}</span>
          </div>
        ))}
        <span className="text-[10px] text-white/25 uppercase tracking-widest">Move cursor to inspect</span>
      </div>
    </div>
  );
}
