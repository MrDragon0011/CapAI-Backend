"use client";

import Link from "next/link";
import { useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { PoolCourtHero } from "./components/PoolCourtHero";
import { XRayPlayer, ACTIONS } from "./components/XRayPlayer";

const STATS = [
  { value: "33", label: "Keypoints tracked" },
  { value: "4", label: "Action classes" },
  { value: "<3s", label: "Analysis time" },
  { value: "500+", label: "Frames sampled" },
];

function ScrollReveal({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.55, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3 mb-6">
      <div className="w-4 h-px" style={{ background: "var(--accent)" }} />
      <span className="font-mono text-[10px] uppercase tracking-[0.25em] text-white/30">
        {children}
      </span>
    </div>
  );
}

export default function Home() {
  const heroRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: heroRef, offset: ["start start", "end start"] });
  const heroOpacity = useTransform(scrollYProgress, [0, 0.6], [1, 0]);

  return (
    <main className="min-h-screen" style={{ background: "var(--bg)" }}>
      <section ref={heroRef} className="relative min-h-screen flex items-center pt-14">
        <motion.div className="relative z-10 max-w-6xl mx-auto w-full px-5 py-16 grid grid-cols-1 lg:grid-cols-2 gap-12 items-center"
          style={{ opacity: heroOpacity }}>
          <div>
            <SectionLabel>Movement Intelligence</SectionLabel>
            <h1 className="text-6xl sm:text-7xl font-black tracking-tight leading-[0.88] mb-6 text-white">
              BIOMECHANICAL<br />ANALYSIS<br />
              <span style={{ color: "var(--accent)" }}>FOR WATER<br />POLO</span>
            </h1>
            <p className="text-sm text-white/40 leading-relaxed max-w-sm mb-8">
              Upload a clip. CapAI maps every joint, classifies the action, and benchmarks each metric against elite thresholds — in seconds.
            </p>
            <div className="flex items-center gap-4">
              <Link
                href="/upload"
                className="inline-flex items-center gap-3 px-5 py-3 text-xs font-bold uppercase tracking-widest transition-all hover:opacity-80"
                style={{
                  background: "var(--accent)",
                  color: "#000",
                }}
              >
                Analyse a clip
                <span>→</span>
              </Link>
              <Link
                href="/about"
                className="text-xs uppercase tracking-widest text-white/30 hover:text-white/55 transition-colors"
              >
                How it works
              </Link>
            </div>
          </div>

          <div>
            <PoolCourtHero />
          </div>
        </motion.div>
      </section>

      <section className="px-5 py-0 max-w-6xl mx-auto">
        <div style={{ borderTop: "1px solid var(--border)", borderBottom: "1px solid var(--border)" }}
          className="grid grid-cols-2 sm:grid-cols-4">
          {STATS.map((s, i) => (
            <motion.div
              key={s.label}
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.07 }}
              className="px-6 py-8 text-center"
              style={{ borderRight: i < 3 ? "1px solid var(--border)" : undefined }}
            >
              <p className="text-4xl font-black text-white mb-1 tracking-tight">{s.value}</p>
              <p className="font-mono text-[10px] text-white/25 uppercase tracking-wider">{s.label}</p>
            </motion.div>
          ))}
        </div>
      </section>

      <section className="px-5 pt-20 pb-8 max-w-6xl mx-auto">
        <ScrollReveal>
          <SectionLabel>X-Ray analysis</SectionLabel>
          <div className="flex items-end justify-between mb-8">
            <h2 className="text-3xl font-black text-white tracking-tight">
              FOUR ACTIONS.<br />ONE ENGINE.
            </h2>
            <p className="text-xs text-white/30 max-w-xs text-right hidden sm:block">
              Hover any card to see the skeleton chain light up and the joints that matter most for that movement.
            </p>
          </div>
        </ScrollReveal>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4"
          style={{ border: "1px solid var(--border)" }}>
          {ACTIONS.map((action, i) => (
            <ScrollReveal key={action.key} delay={i * 0.06}>
              <div style={{ borderRight: i < 3 ? "1px solid var(--border)" : undefined }}>
                <XRayPlayer action={action} />
              </div>
            </ScrollReveal>
          ))}
        </div>
      </section>

      <section className="px-5 pt-16 pb-20 max-w-6xl mx-auto">
        <ScrollReveal>
          <div
            className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-8 p-8"
            style={{ border: "1px solid var(--border-strong)", background: "var(--surface)" }}
          >
            <div>
              <p className="font-mono text-[10px] uppercase tracking-widest text-white/25 mb-2">
                Ready to analyse?
              </p>
              <h3 className="text-2xl font-black text-white tracking-tight">
                UPLOAD YOUR CLIP
              </h3>
              <p className="text-xs text-white/35 mt-1.5 max-w-xs leading-relaxed">
                Shooting, passing, swimming, or goalie work — any water polo clip works.
              </p>
            </div>
            <Link
              href="/upload"
              className="shrink-0 inline-flex items-center gap-3 px-6 py-3.5 text-xs font-bold uppercase tracking-widest transition-all hover:opacity-80 whitespace-nowrap"
              style={{ background: "var(--accent)", color: "#000" }}
            >
              Upload a clip →
            </Link>
          </div>
        </ScrollReveal>
      </section>
    </main>
  );
}
