"use client";

import { useCallback, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Props {
  onResult: (data: unknown) => void;
  onError: (msg: string) => void;
  loading: boolean;
  setLoading: (v: boolean) => void;
}

export function UploadZone({ onResult, onError, loading, setLoading }: Props) {
  const [dragging, setDragging] = useState(false);

  const submit = useCallback(
    async (file: File) => {
      setLoading(true);
      const form = new FormData();
      form.append("file", file);
      try {
        const res = await fetch("http://localhost:8000/analyze", { method: "POST", body: form });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          onError(err.detail ?? "Request failed");
        } else {
          onResult(await res.json());
        }
      } catch {
        onError("Could not reach the backend. Make sure it is running on port 8000.");
      } finally {
        setLoading(false);
      }
    },
    [onResult, onError, setLoading]
  );

  const handleFiles = (files: FileList | null) => {
    if (files?.[0]) submit(files[0]);
  };

  return (
    <motion.label
      htmlFor="video-upload"
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
      animate={{ borderColor: dragging ? "rgba(139,92,246,0.7)" : "rgba(255,255,255,0.08)" }}
      className="relative flex flex-col items-center justify-center gap-4 w-full rounded-2xl border-2 border-dashed bg-white/[0.03] backdrop-blur-xl cursor-pointer py-14 px-8 transition-colors"
    >
      <input
        id="video-upload"
        type="file"
        accept="video/*"
        className="sr-only"
        onChange={(e) => handleFiles(e.target.files)}
        disabled={loading}
      />

      <AnimatePresence mode="wait">
        {loading ? (
          <motion.div
            key="loading"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            className="flex flex-col items-center gap-3"
          >
            <div className="w-10 h-10 rounded-full border-2 border-violet-400 border-t-transparent animate-spin" />
            <p className="text-sm text-white/50">Analysing movement…</p>
          </motion.div>
        ) : (
          <motion.div
            key="idle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center gap-3 text-center"
          >
            <div className="w-14 h-14 rounded-full bg-white/[0.06] flex items-center justify-center text-2xl">
              🎬
            </div>
            <div>
              <p className="text-white/80 font-medium">Drop a video or click to upload</p>
              <p className="text-white/30 text-sm mt-1">MP4, MOV, AVI · any water polo clip</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.label>
  );
}
