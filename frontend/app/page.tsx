"use client";

import Link from "next/link";
import { motion, useScroll, useTransform } from "framer-motion";
import { useRef } from "react";
import { PoolCourtHero } from "./components/PoolCourtHero";
import { XRayPlayer, ACTIONS } from "./components/XRayPlayer";

const STATS = [
  { value: "33", label: "Pose keypoints tracked" },
  { value: "4", label: "Action classes" },
  { value: "<3s", label: "Analysis turnaround" },
  { value: "500+", label: "Frames processed" },
];

function ScrollReveal({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.6, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}

export default function Home() {
  const heroRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: heroRef, offset: ["start start", "end start"] });
  const heroY = useTransform(scrollYProgress, [0, 1], ["0%", "30%"]);
  const heroOpacity = useTransform(scrollYProgress, [0, 0.7], [1, 0]);

  return (
    <main className="min-h-screen" style={{ background: "var(--ocean-950)" }}>
      <section ref={heroRef} className="relative min-h-screen flex flex-col justify-center px-5 pt-24 pb-16 overflow-hidden">
        <motion.div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: "radial-gradient(ellipse 70% 50% at 50% 0%, rgba(26,74,110,0.35) 0%, transparent 70%)",
            y: heroY,
            opacity: heroOpacity,
          }}
        />

        <div className="relative z-10 max-w-6xl mx-auto w-full">
          <motion.div
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
            className="mb-10"
          >
            <div className="flex items-center gap-2 mb-5">
              <div className="h-px w-8" style={{ background: "var(--teal-accent)" }} />
              <span className="text-[11px] uppercase tracking-[0.3em]" style={{ color: "var(--teal-accent)" }}>
                Movement Intelligence
              </span>
            </div>
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-black tracking-tight leading-[0.92] mb-5">
              <span className="text-white">See every</span>
              <br />
              <span style={{
                background: "linear-gradient(90deg, #22d3ee, #00c8b4)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}>
                biomechanical
              </span>
              <br />
              <span className="text-white">detail</span>
            </h1>
            <p className="text-white/40 text-lg max-w-lg leading-relaxed">
              Upload a water polo clip. CapAI extracts pose landmarks frame by frame, classifies the action, and benchmarks it against elite standards — in seconds.
            </p>
            <div className="flex items-center gap-4 mt-8">
              <Link
                href="/upload"
                className="inline-flex items-center gap-2.5 px-6 py-3 rounded-xl font-semibold text-sm text-white transition-all hover:scale-[1.02] active:scale-[0.98]"
                style={{
                  background: "linear-gradient(135deg, #1a4a6e, #00c8b4)",
                  boxShadow: "0 0 30px rgba(0,200,180,0.25)",
                }}
              >
                Analyse a clip
                <span className="text-white/70">→</span>
              </Link>
              <Link href="/about" className="text-sm text-white/35 hover:text-white/60 transition-colors">
                How it works
              </Link>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.25, ease: [0.22, 1, 0.36, 1] }}
          >
            <PoolCourtHero />
          </motion.div>
        </div>
      </section>

      <section className="px-5 py-16 max-w-6xl mx-auto">
        <ScrollReveal>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-20">
            {STATS.map((s, i) => (
              <motion.div
                key={s.label}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.08, duration: 0.5 }}
                className="rounded-xl px-5 py-6 text-center"
                style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)" }}
              >
                <p className="text-3xl font-black text-white mb-1">{s.value}</p>
                <p className="text-xs text-white/35 uppercase tracking-wider">{s.label}</p>
              </motion.div>
            ))}
          </div>
        </ScrollReveal>

        <ScrollReveal>
          <div className="mb-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="h-px w-6" style={{ background: "var(--teal-accent)" }} />
              <span className="text-[11px] uppercase tracking-[0.3em] text-white/30">X-Ray Analysis</span>
            </div>
            <h2 className="text-3xl font-bold text-white">Four actions. One engine.</h2>
            <p className="text-white/35 mt-2 text-sm max-w-md">
              Hover any card to see the skeleton chain analysed and the joints that matter most for that movement.
            </p>
          </div>
        </ScrollReveal>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-20">
          {ACTIONS.map((action, i) => (
            <ScrollReveal key={action.key} delay={i * 0.08}>
              <XRayPlayer action={action} />
            </ScrollReveal>
          ))}
        </div>

        <ScrollReveal>
          <div className="rounded-2xl overflow-hidden relative"
            style={{
              background: "linear-gradient(135deg, rgba(26,74,110,0.3) 0%, rgba(0,200,180,0.08) 100%)",
              border: "1px solid rgba(0,200,180,0.15)",
            }}>
            <div className="px-8 py-12 relative z-10 flex flex-col sm:flex-row items-center justify-between gap-8">
              <div>
                <h3 className="text-2xl font-bold text-white mb-2">Ready to analyse your technique?</h3>
                <p className="text-white/40 text-sm">Upload any water polo clip — shooting, passing, swimming, or goalie work.</p>
              </div>
              <Link
                href="/upload"
                className="shrink-0 inline-flex items-center gap-2 px-7 py-3.5 rounded-xl font-semibold text-sm text-white transition-all hover:scale-[1.02] active:scale-[0.98] whitespace-nowrap"
                style={{
                  background: "linear-gradient(135deg, #1a4a6e, #00c8b4)",
                  boxShadow: "0 0 30px rgba(0,200,180,0.2)",
                }}
              >
                Upload a clip →
              </Link>
            </div>
          </div>
        </ScrollReveal>
      </section>
    </main>
  );
}
