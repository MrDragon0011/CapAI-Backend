export function Footer() {
  return (
    <footer style={{ borderTop: "1px solid var(--border)" }} className="px-6 py-6 mt-20">
      <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
        <span className="text-xs uppercase tracking-[0.2em] text-white/20">CapAI</span>
        <p className="text-xs text-white/15 font-mono">
          pose extraction · action classification · elite benchmarking
        </p>
      </div>
    </footer>
  );
}
