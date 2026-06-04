"use client";

import { useRef, useState, useEffect } from "react";
import { motion } from "framer-motion";

const CONNECTIONS: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
  [24, 26], [26, 28],
];

const BASE_JOINTS: Record<number, [number, number]> = {
  0: [0.5, 0.10], 11: [0.38, 0.30], 12: [0.62, 0.28],
  13: [0.27, 0.46], 14: [0.73, 0.38], 15: [0.16, 0.56],
  16: [0.83, 0.27], 23: [0.42, 0.56], 24: [0.58, 0.54],
  25: [0.36, 0.72], 26: [0.62, 0.70], 27: [0.32, 0.87], 28: [0.65, 0.85],
};

const HIGHLIGHT = new Set([11, 12, 13, 14, 15, 16]);

export function PoolCourtHero() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 50);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width;
    const H = canvas.height;
    const t = tick * 0.05;

    ctx.clearRect(0, 0, W, H);

    const wobble = (idx: number): [number, number] => {
      const base = BASE_JOINTS[idx];
      const amp = HIGHLIGHT.has(idx) ? 0.018 : 0.008;
      return [
        base[0] + Math.sin(t + idx * 1.3) * amp,
        base[1] + Math.cos(t * 0.7 + idx * 0.9) * amp * 0.6,
      ];
    };

    for (const [a, b] of CONNECTIONS) {
      const pa = wobble(a);
      const pb = wobble(b);
      const isHL = HIGHLIGHT.has(a) && HIGHLIGHT.has(b);
      ctx.beginPath();
      ctx.moveTo(pa[0] * W, pa[1] * H);
      ctx.lineTo(pb[0] * W, pb[1] * H);
      if (isHL) {
        ctx.strokeStyle = "rgba(0,212,255,0.8)";
        ctx.lineWidth = 1.5;
      } else {
        ctx.strokeStyle = "rgba(255,255,255,0.3)";
        ctx.lineWidth = 1;
      }
      ctx.stroke();
    }

    for (const [idx, base] of Object.entries(BASE_JOINTS)) {
      const [jx, jy] = wobble(Number(idx));
      const isHL = HIGHLIGHT.has(Number(idx));
      ctx.beginPath();
      ctx.arc(jx * W, jy * H, isHL ? 4 : 2.5, 0, Math.PI * 2);
      ctx.fillStyle = isHL ? "#00d4ff" : "rgba(255,255,255,0.5)";
      ctx.fill();

      if (isHL) {
        ctx.beginPath();
        ctx.arc(jx * W, jy * H, 8, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(0,212,255,0.25)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }, [tick]);

  return (
    <div
      className="relative w-full overflow-hidden"
      style={{
        height: 380,
        background: "var(--surface)",
        border: "1px solid var(--border)",
      }}
    >
      <div className="grid-bg-fine absolute inset-0" />

      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 900 380" preserveAspectRatio="xMidYMid slice">
        <rect x="80" y="40" width="740" height="300" fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth="1" />
        {[1, 2, 3, 4, 5, 6, 7].map((i) => (
          <line key={i} x1={80 + i * 92.5} y1="40" x2={80 + i * 92.5} y2="340"
            stroke="rgba(255,255,255,0.05)" strokeWidth="1" strokeDasharray="3 5" />
        ))}
        <line x1="80" y1="190" x2="820" y2="190" stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
        <line x1="80" y1="110" x2="820" y2="110" stroke="rgba(255,60,60,0.3)" strokeWidth="1" strokeDasharray="6 4" />
        <line x1="80" y1="270" x2="820" y2="270" stroke="rgba(255,60,60,0.3)" strokeWidth="1" strokeDasharray="6 4" />
        <rect x="80" y="40" width="80" height="300" fill="rgba(255,255,255,0.02)" />
        <rect x="740" y="40" width="80" height="300" fill="rgba(255,255,255,0.02)" />
        <rect x="80" y="130" width="20" height="120" fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth="1" />
        <rect x="800" y="130" width="20" height="120" fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth="1" />
        <circle cx="450" cy="190" r="30" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
        <text x="88" y="34" fill="rgba(255,255,255,0.2)" fontSize="8" fontFamily="monospace">GOAL</text>
        <text x="806" y="34" fill="rgba(255,255,255,0.2)" fontSize="8" fontFamily="monospace">GOAL</text>
        <text x="450" y="34" fill="rgba(255,255,255,0.12)" fontSize="8" textAnchor="middle" fontFamily="monospace">CENTER</text>
        <text x="75" y="113" fill="rgba(255,80,80,0.4)" fontSize="7" textAnchor="end" fontFamily="monospace">5m</text>
        <text x="75" y="273" fill="rgba(255,80,80,0.4)" fontSize="7" textAnchor="end" fontFamily="monospace">5m</text>
      </svg>

      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative" style={{ width: 220, height: 280 }}>
          <canvas ref={canvasRef} width={220} height={280} className="absolute inset-0" />
        </div>
      </div>

      <motion.div
        className="absolute top-0 left-0 right-0 h-px"
        style={{ background: "rgba(0,212,255,0.6)" }}
        animate={{ y: [0, 380, 0] }}
        transition={{ duration: 5, repeat: Infinity, ease: "linear" }}
      />

      <div className="absolute bottom-4 left-4 right-4 flex items-end justify-between">
        <div className="font-mono text-[9px] text-white/20 space-y-0.5">
          <div>LANDMARKS: 33</div>
          <div>FRAME: <span className="text-white/40">{(tick % 500).toString().padStart(3, "0")}</span></div>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full" style={{ background: "var(--accent)" }} />
          <span className="font-mono text-[9px] text-white/30 uppercase tracking-wider">Live skeleton</span>
        </div>
        <div className="font-mono text-[9px] text-white/20 text-right space-y-0.5">
          <div>ACTION: <span className="text-white/40">SHOOTING</span></div>
          <div>STATUS: <span style={{ color: "var(--accent)" }}>ANALYSING</span></div>
        </div>
      </div>
    </div>
  );
}
