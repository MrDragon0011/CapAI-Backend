"use client";

import { motion } from "framer-motion";

const STEPS = [
  {
    num: "01",
    title: "Upload your clip",
    body: "Drop any water polo video — shooting, passing, swimming, or goalie work. MP4, MOV, and AVI are all supported.",
    color: "#22d3ee",
  },
  {
    num: "02",
    title: "Pose landmark extraction",
    body: "MediaPipe HolisticLandmarker maps 33 body keypoints per sampled frame, tracking wrists, elbows, shoulders, hips, knees, and ankles through the full motion.",
    color: "#00c8b4",
  },
  {
    num: "03",
    title: "Action classification",
    body: "A scikit-learn Random Forest trained on sliding-window sequence features identifies the movement type from the landmark trajectory.",
    color: "#a78bfa",
  },
  {
    num: "04",
    title: "Elite benchmarking",
    body: "Each biomechanical metric — elbow angle, shoulder rotation, hip elevation, stroke rate — is compared against sport-science thresholds for elite water polo athletes.",
    color: "#f97316",
  },
];

export default function AboutPage() {
  return (
    <main className="min-h-screen pt-28 px-5 pb-20" style={{ background: "var(--ocean-950)" }}>
      <div className="max-w-3xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-16"
        >
          <div className="flex items-center gap-2 mb-4">
            <div className="h-px w-8" style={{ background: "var(--teal-accent)" }} />
            <span className="text-[11px] uppercase tracking-[0.3em]" style={{ color: "var(--teal-accent)" }}>
              How it works
            </span>
          </div>
          <h1 className="text-4xl font-black text-white mb-4">From pixels to coaching feedback</h1>
          <p className="text-white/40 text-base leading-relaxed">
            CapAI runs a four-stage pipeline on every uploaded clip, turning raw video into quantified biomechanical insights without any manual annotation.
          </p>
        </motion.div>

        <div className="space-y-6">
          {STEPS.map((s, i) => (
            <motion.div
              key={s.num}
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ delay: i * 0.1, duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
              className="flex gap-6 rounded-xl p-6"
              style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)" }}
            >
              <div className="shrink-0 w-12 h-12 rounded-lg flex items-center justify-center text-xs font-black"
                style={{ background: s.color + "18", color: s.color, border: `1px solid ${s.color}30` }}>
                {s.num}
              </div>
              <div>
                <h3 className="text-white font-semibold mb-1.5">{s.title}</h3>
                <p className="text-white/40 text-sm leading-relaxed">{s.body}</p>
              </div>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mt-12 rounded-xl p-6"
          style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}
        >
          <h3 className="text-white/60 text-xs font-semibold uppercase tracking-widest mb-3">Stack</h3>
          <div className="flex flex-wrap gap-2">
            {["MediaPipe", "scikit-learn", "FastAPI", "Next.js 16", "Framer Motion", "OpenCV"].map((t) => (
              <span key={t} className="px-3 py-1 rounded-md text-xs text-white/40"
                style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.07)" }}>
                {t}
              </span>
            ))}
          </div>
        </motion.div>
      </div>
    </main>
  );
}
