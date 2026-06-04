"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/upload", label: "Analyse" },
  { href: "/about", label: "About" },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <nav
      className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 h-14"
      style={{ background: "var(--bg)", borderBottom: "1px solid var(--border)" }}
    >
      <Link href="/" className="flex items-center gap-3 group">
        <div
          className="w-6 h-6 flex items-center justify-center"
          style={{ border: "1px solid var(--border-strong)" }}
        >
          <svg width="12" height="14" viewBox="0 0 12 14" fill="none">
            <circle cx="6" cy="2.5" r="1.5" fill="var(--accent)" />
            <line x1="6" y1="4" x2="6" y2="8" stroke="white" strokeWidth="1.2" />
            <line x1="6" y1="6.5" x2="3.5" y2="9" stroke="white" strokeWidth="1.2" />
            <line x1="6" y1="6.5" x2="8.5" y2="9" stroke="white" strokeWidth="1.2" />
            <line x1="6" y1="8" x2="4" y2="12" stroke="white" strokeWidth="1.2" />
            <line x1="6" y1="8" x2="8" y2="12" stroke="white" strokeWidth="1.2" />
          </svg>
        </div>
        <span className="text-xs font-bold tracking-[0.15em] text-white/80 uppercase group-hover:text-white transition-colors">
          CapAI
        </span>
      </Link>

      <div className="flex items-center">
        {LINKS.map((link, i) => {
          const active = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`px-5 h-14 flex items-center text-xs uppercase tracking-widest transition-colors ${
                active
                  ? "text-white"
                  : "text-white/35 hover:text-white/65"
              }`}
              style={{
                borderLeft: i === 0 ? "1px solid var(--border)" : undefined,
                borderRight: "1px solid var(--border)",
                borderBottom: active ? "2px solid var(--accent)" : "2px solid transparent",
              }}
            >
              {link.label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
