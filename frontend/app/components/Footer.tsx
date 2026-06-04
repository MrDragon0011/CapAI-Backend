export function Footer() {
  return (
    <footer className="border-t border-white/[0.06] px-6 py-8 mt-20">
      <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2 text-white/30 text-xs">
          <span>CapAI</span>
          <span>·</span>
          <span>Water polo movement intelligence</span>
        </div>
        <p className="text-white/20 text-xs">
          Pose extraction · Action classification · Elite benchmarking
        </p>
      </div>
    </footer>
  );
}
