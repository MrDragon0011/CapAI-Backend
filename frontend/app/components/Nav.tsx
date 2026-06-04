"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/upload", label: "Analyse" },
  { href: "/about", label: "About" },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4"
      style={{ background: "linear-gradient(180deg, rgba(2,11,20,0.95) 0%, transparent 100%)", backdropFilter: "blur(8px)" }}>
      <Link href="/" className="flex items-center gap-2.5 group">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: "linear-gradient(135deg, #1a4a6e, #00c8b4)" }}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="4" r="2" fill="white" opacity="0.9" />
            <line x1="8" y1="6" x2="8" y2="10" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
            <line x1="8" y1="8" x2="5" y2="11" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
            <line x1="8" y1="8" x2="11" y2="11" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
            <line x1="8" y1="10" x2="5.5" y2="13" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
            <line x1="8" y1="10" x2="10.5" y2="13" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </div>
        <span className="text-sm font-bold tracking-wide text-white/90 group-hover:text-white transition-colors">CapAI</span>
      </Link>

      <div className="flex items-center gap-1">
        {LINKS.map((link) => {
          const active = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`relative px-4 py-1.5 text-sm rounded-lg transition-colors ${
                active ? "text-white" : "text-white/45 hover:text-white/75"
              }`}
            >
              {active && (
                <motion.div
                  layoutId="nav-pill"
                  className="absolute inset-0 rounded-lg"
                  style={{ background: "rgba(26,74,110,0.5)", border: "1px solid rgba(0,200,180,0.2)" }}
                  transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
                />
              )}
              <span className="relative z-10">{link.label}</span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
