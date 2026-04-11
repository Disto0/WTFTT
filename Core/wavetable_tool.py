"""
Wavetable Analyzer & Converter
Analyse, visualise et exporte des fichiers wavetable WAV.
Dépendances : numpy (pip install numpy)
Tout le reste (tkinter, wave, struct) est inclus dans Python standard.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import wave
import struct
import os
import math
import numpy as np


# ─────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────
CYCLE_SIZES   = [256, 512, 1024, 2048]
EXPORT_SIZES  = [256, 512, 1024, 2048]
COLORS = {
    "bg":        "#1a1a2e",
    "panel":     "#16213e",
    "accent":    "#0f3460",
    "highlight": "#e94560",
    "text":      "#eaeaea",
    "muted":     "#7a7a9a",
    "wave":      "#4fc3f7",
    "fft":       "#81c784",
    "grid":      "#2a2a4a",
}
LABEL_COLORS = {
    "sin":      "#4fc3f7",
    "square":   "#81c784",
    "saw":      "#ffb74d",
    "triangle": "#ce93d8",
    "complex":  "#7a7a9a",
}


# ─────────────────────────────────────────────
#  Audio helpers
# ─────────────────────────────────────────────
def read_wav(path):
    with wave.open(path, "rb") as w:
        sr   = w.getframerate()
        ch   = w.getnchannels()
        sw   = w.getsampwidth()
        nf   = w.getnframes()
        raw  = w.readframes(nf)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if ch == 2:
        audio = audio[::2]
    audio /= np.iinfo(dtype).max
    return audio, sr


def write_wav(path, audio_f32, sr, sampwidth=2):
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sampwidth]
    max_val = np.iinfo(dtype).max
    data = np.clip(audio_f32, -1.0, 1.0)
    data = (data * max_val).astype(dtype)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(sampwidth)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


def resample_cycle(cycle, target_size):
    if len(cycle) == target_size:
        return cycle
    x_old = np.linspace(0, 1, len(cycle), endpoint=False)
    x_new = np.linspace(0, 1, target_size, endpoint=False)
    return np.interp(x_new, x_old, cycle)


def detect_cycle_size(audio):
    """Retourne la taille la plus probable avec un score de confiance."""
    scores = {}
    for cs in CYCLE_SIZES:
        if len(audio) < cs:
            continue
        n = len(audio) // cs
        if n == 0:
            continue
        cycles = [audio[i*cs:(i+1)*cs] for i in range(min(n, 4))]
        if len(cycles) < 2:
            scores[cs] = 0.5
            continue
        sims = []
        for i in range(len(cycles)-1):
            a, b = cycles[i], cycles[i+1]
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            if norm > 0:
                sims.append(abs(np.dot(a, b) / norm))
        scores[cs] = np.mean(sims) if sims else 0.0
    if not scores:
        return 2048, {}
    best = max(scores, key=scores.get)
    return best, scores


def classify_cycle(cycle):
    fft = np.abs(np.fft.rfft(cycle))
    if fft.max() == 0:
        return "complex", fft
    fft_norm = fft / fft.max()
    fund = fft[1] if len(fft) > 1 else 1
    total = sum(fft[1:10]) if len(fft) > 10 else 1
    odds  = sum(fft[k] for k in range(1, 10, 2) if k < len(fft))
    evens = sum(fft[k] for k in range(2, 10, 2) if k < len(fft))
    h3 = fft[3] if len(fft) > 3 else 0
    h5 = fft[5] if len(fft) > 5 else 0
    odd_ratio = odds / total if total > 0 else 0
    if odd_ratio > 0.7:
        if fund > 0 and h3 / fund < 0.15:
            label = "sin"
        elif h3 > 0 and h5 / h3 < 0.25:
            label = "triangle"
        else:
            label = "square"
    elif evens / (odds + evens + 1e-9) > 0.25:
        label = "saw"
    else:
        label = "complex"
    return label, fft_norm[:16]


# ─────────────────────────────────────────────
#  Application principale
# ─────────────────────────────────────────────
class WavetableTool(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wavetable Analyzer")
        self.configure(bg=COLORS["bg"])
        self.geometry("980x720")
        self.minsize(800, 600)
        self.resizable(True, True)

        # État
        self.audio       = None
        self.sr          = 44100
        self.cycle_size  = tk.IntVar(value=2048)
        self.current_idx = 0
        self.cycles      = []
        self.filepath    = None
        self.mode        = tk.StringVar(value="wavetable")  # wavetable | batch

        self._build_ui()

    # ── UI ──────────────────────────────────
    def _build_ui(self):
        # Titre
        hdr = tk.Frame(self, bg=COLORS["bg"], pady=8)
        hdr.pack(fill="x", padx=16)
        tk.Label(hdr, text="WAVETABLE ANALYZER", font=("Consolas", 14, "bold"),
                 bg=COLORS["bg"], fg=COLORS["highlight"]).pack(side="left")
        tk.Label(hdr, text="& CONVERTER", font=("Consolas", 14),
                 bg=COLORS["bg"], fg=COLORS["muted"]).pack(side="left", padx=(4, 0))

        # Mode tabs
        mode_frame = tk.Frame(self, bg=COLORS["panel"], pady=6)
        mode_frame.pack(fill="x", padx=16, pady=(0, 8))
        for label, val in [("  Wavetable (fichier unique)  ", "wavetable"),
                           ("  Batch (dossier → wavetable)  ", "batch")]:
            b = tk.Radiobutton(mode_frame, text=label, variable=self.mode, value=val,
                               command=self._on_mode_change,
                               bg=COLORS["panel"], fg=COLORS["text"],
                               selectcolor=COLORS["accent"],
                               activebackground=COLORS["panel"],
                               font=("Consolas", 10), indicatoron=False,
                               relief="flat", padx=10, pady=4,
                               bd=0, highlightthickness=0)
            b.pack(side="left", padx=4)

        # Corps principal : panneau gauche + droite
        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        # ── Panneau gauche ──
        self.left = tk.Frame(body, bg=COLORS["panel"], width=240)
        self.left.pack(side="left", fill="y", padx=(0, 8))
        self.left.pack_propagate(False)
        self._build_left_panel()

        # ── Panneau droite ──
        self.right = tk.Frame(body, bg=COLORS["bg"])
        self.right.pack(side="left", fill="both", expand=True)
        self._build_right_panel()

        # Barre de statut
        self.status_var = tk.StringVar(value="Chargez un fichier WAV pour commencer.")
        tk.Label(self, textvariable=self.status_var, font=("Consolas", 9),
                 bg=COLORS["accent"], fg=COLORS["text"], anchor="w", padx=10
                 ).pack(fill="x", side="bottom")

    def _build_left_panel(self):
        pad = {"padx": 12, "pady": 4}

        tk.Label(self.left, text="FICHIER", font=("Consolas", 9, "bold"),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w", padx=12, pady=(12,2))

        self.file_label = tk.Label(self.left, text="Aucun fichier", font=("Consolas", 9),
                                   bg=COLORS["panel"], fg=COLORS["text"],
                                   wraplength=210, justify="left")
        self.file_label.pack(anchor="w", padx=12, pady=(0, 4))

        self._btn("Ouvrir WAV...", self._open_file).pack(fill="x", padx=12, pady=2)

        # Séparateur
        tk.Frame(self.left, bg=COLORS["grid"], height=1).pack(fill="x", padx=12, pady=8)

        tk.Label(self.left, text="TAILLE DE CYCLE", font=("Consolas", 9, "bold"),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w", padx=12, pady=(0,4))

        self.detect_label = tk.Label(self.left, text="—", font=("Consolas", 9),
                                     bg=COLORS["panel"], fg=COLORS["wave"])
        self.detect_label.pack(anchor="w", padx=12, pady=(0, 6))

        for cs in CYCLE_SIZES:
            rb = tk.Radiobutton(self.left, text=f"{cs} samples",
                                variable=self.cycle_size, value=cs,
                                command=self._on_cycle_size_change,
                                bg=COLORS["panel"], fg=COLORS["text"],
                                selectcolor=COLORS["accent"],
                                activebackground=COLORS["panel"],
                                font=("Consolas", 10), indicatoron=True)
            rb.pack(anchor="w", padx=12)

        # Séparateur
        tk.Frame(self.left, bg=COLORS["grid"], height=1).pack(fill="x", padx=12, pady=8)

        # Infos cycle
        tk.Label(self.left, text="INFOS", font=("Consolas", 9, "bold"),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w", padx=12, pady=(0,4))
        self.info_label = tk.Label(self.left, text="—", font=("Consolas", 9),
                                   bg=COLORS["panel"], fg=COLORS["text"],
                                   justify="left", wraplength=210)
        self.info_label.pack(anchor="w", padx=12)

        # Séparateur
        tk.Frame(self.left, bg=COLORS["grid"], height=1).pack(fill="x", padx=12, pady=8)

        # Export
        tk.Label(self.left, text="EXPORT", font=("Consolas", 9, "bold"),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w", padx=12, pady=(0,4))

        tk.Label(self.left, text="Taille cycle export :", font=("Consolas", 9),
                 bg=COLORS["panel"], fg=COLORS["text"]).pack(anchor="w", padx=12)

        self.export_size = tk.IntVar(value=2048)
        export_combo = ttk.Combobox(self.left, textvariable=self.export_size,
                                    values=EXPORT_SIZES, state="readonly", width=10,
                                    font=("Consolas", 10))
        export_combo.pack(anchor="w", padx=12, pady=4)

        self._btn("Exporter WAV séparés",   self._export_separate).pack(fill="x", padx=12, pady=2)
        self._btn("Exporter WAV unifié",    self._export_unified).pack(fill="x", padx=12, pady=2)

    def _build_right_panel(self):
        # Navigation cycles
        nav = tk.Frame(self.right, bg=COLORS["bg"])
        nav.pack(fill="x", pady=(0, 6))

        self._btn("◀ Préc.", self._prev_cycle, small=True).pack(side="left")
        self.nav_label = tk.Label(nav, text="— / —", font=("Consolas", 11, "bold"),
                                  bg=COLORS["bg"], fg=COLORS["text"], padx=16)
        self.nav_label.pack(side="left")
        self._btn("Suiv. ▶", self._next_cycle, small=True).pack(side="left")

        self.cycle_badge = tk.Label(nav, text="", font=("Consolas", 10, "bold"),
                                    bg=COLORS["bg"], fg=COLORS["highlight"], padx=8)
        self.cycle_badge.pack(side="left")

        # Canvases : oscilloscope + FFT
        canvases = tk.Frame(self.right, bg=COLORS["bg"])
        canvases.pack(fill="both", expand=True)
        canvases.columnconfigure(0, weight=3)
        canvases.columnconfigure(1, weight=2)
        canvases.rowconfigure(0, weight=1)

        wave_frame = tk.Frame(canvases, bg=COLORS["panel"])
        wave_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(wave_frame, text="OSCILLOSCOPE", font=("Consolas", 8),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w", padx=8, pady=(6,0))
        self.wave_canvas = tk.Canvas(wave_frame, bg=COLORS["panel"],
                                     highlightthickness=0)
        self.wave_canvas.pack(fill="both", expand=True, padx=4, pady=(0,4))
        self.wave_canvas.bind("<Configure>", lambda e: self._draw_wave())

        fft_frame = tk.Frame(canvases, bg=COLORS["panel"])
        fft_frame.grid(row=0, column=1, sticky="nsew")
        tk.Label(fft_frame, text="SPECTRE FFT", font=("Consolas", 8),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w", padx=8, pady=(6,0))
        self.fft_canvas = tk.Canvas(fft_frame, bg=COLORS["panel"],
                                    highlightthickness=0)
        self.fft_canvas.pack(fill="both", expand=True, padx=4, pady=(0,4))
        self.fft_canvas.bind("<Configure>", lambda e: self._draw_fft())

        # Miniatures
        tk.Label(self.right, text="TOUS LES CYCLES", font=("Consolas", 8),
                 bg=COLORS["bg"], fg=COLORS["muted"]).pack(anchor="w", pady=(6,2))
        thumb_outer = tk.Frame(self.right, bg=COLORS["bg"], height=64)
        thumb_outer.pack(fill="x")
        thumb_outer.pack_propagate(False)

        self.thumb_scroll = tk.Canvas(thumb_outer, bg=COLORS["bg"],
                                      highlightthickness=0, height=64)
        sb = ttk.Scrollbar(thumb_outer, orient="horizontal",
                           command=self.thumb_scroll.xview)
        self.thumb_scroll.configure(xscrollcommand=sb.set)
        sb.pack(side="bottom", fill="x")
        self.thumb_scroll.pack(fill="both", expand=True)
        self.thumb_frame = tk.Frame(self.thumb_scroll, bg=COLORS["bg"])
        self.thumb_scroll.create_window((0, 0), window=self.thumb_frame, anchor="nw")
        self.thumb_frame.bind("<Configure>", lambda e: self.thumb_scroll.configure(
            scrollregion=self.thumb_scroll.bbox("all")))

    def _btn(self, text, cmd, small=False):
        font = ("Consolas", 9) if small else ("Consolas", 10)
        return tk.Button(self, text=text, command=cmd, font=font,
                         bg=COLORS["accent"], fg=COLORS["text"],
                         activebackground=COLORS["highlight"],
                         activeforeground="#ffffff",
                         relief="flat", bd=0, padx=8, pady=4,
                         cursor="hand2")

    # ── Événements ───────────────────────────
    def _on_mode_change(self):
        if self.mode.get() == "batch":
            self._open_batch()

    def _on_cycle_size_change(self):
        if self.audio is not None:
            self._slice_cycles()
            self._refresh_display()

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Ouvrir un fichier wavetable",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.audio, self.sr = read_wav(path)
            self.filepath = path
            self.file_label.config(text=os.path.basename(path))
            self._auto_detect()
            self._slice_cycles()
            self._refresh_display()
            self.status_var.set(f"Chargé : {os.path.basename(path)} — {len(self.audio)} samples @ {self.sr} Hz")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de lire le fichier :\n{e}")

    def _open_batch(self):
        folder = filedialog.askdirectory(title="Choisir un dossier de waveforms")
        if not folder:
            self.mode.set("wavetable")
            return
        wavs = sorted([f for f in os.listdir(folder) if f.lower().endswith(".wav")])
        if not wavs:
            messagebox.showwarning("Batch", "Aucun fichier WAV trouvé dans ce dossier.")
            self.mode.set("wavetable")
            return
        # Charger et concaténer toutes les waveforms
        cycles_audio = []
        target = self.export_size.get()
        for fname in wavs:
            try:
                a, _ = read_wav(os.path.join(folder, fname))
                # Prendre un seul cycle (le premier)
                if len(a) >= target:
                    cycles_audio.append(resample_cycle(a[:target], target))
                else:
                    cycles_audio.append(resample_cycle(a, target))
            except Exception:
                pass
        if not cycles_audio:
            messagebox.showerror("Batch", "Aucun fichier valide trouvé.")
            self.mode.set("wavetable")
            return
        self.audio = np.concatenate(cycles_audio)
        self.filepath = os.path.join(folder, "batch_result.wav")
        self.cycle_size.set(target)
        self.file_label.config(text=f"Batch: {len(wavs)} fichiers")
        self._slice_cycles()
        self._refresh_display()
        self.status_var.set(f"Batch : {len(wavs)} fichiers → {len(cycles_audio)} cycles de {target} samples")
        self.mode.set("wavetable")

    def _auto_detect(self):
        best, scores = detect_cycle_size(self.audio)
        self.cycle_size.set(best)
        score = scores.get(best, 0)
        lines = []
        for cs in CYCLE_SIZES:
            s = scores.get(cs, 0)
            n = len(self.audio) // cs
            marker = " ◀" if cs == best else ""
            lines.append(f"{cs}: {n} cycles (conf. {s:.2f}){marker}")
        self.detect_label.config(text=f"Auto-détecté : {best}")
        tip = "\n".join(lines)
        self.detect_label.bind("<Enter>", lambda e, t=tip: self.status_var.set(t))
        self.detect_label.bind("<Leave>", lambda e: self.status_var.set(""))

    def _slice_cycles(self):
        if self.audio is None:
            return
        cs = self.cycle_size.get()
        n = len(self.audio) // cs
        self.cycles = [self.audio[i*cs:(i+1)*cs] for i in range(n)]
        self.current_idx = 0

    def _refresh_display(self):
        if not self.cycles:
            return
        self.nav_label.config(text=f"Cycle  {self.current_idx+1}  /  {len(self.cycles)}")
        cycle = self.cycles[self.current_idx]
        label, fft = classify_cycle(cycle)
        self.cycle_badge.config(text=label.upper(),
                                fg=LABEL_COLORS.get(label, COLORS["muted"]))
        n_total = len(self.audio)
        cs = self.cycle_size.get()
        info = (f"Total samples : {n_total}\n"
                f"Cycles : {len(self.cycles)}\n"
                f"Taille : {cs} samples\n"
                f"SR : {self.sr} Hz\n"
                f"Durée cycle : {cs/self.sr*1000:.1f} ms")
        self.info_label.config(text=info)
        self._draw_wave()
        self._draw_fft()
        self._build_thumbs()

    def _draw_wave(self):
        c = self.wave_canvas
        c.delete("all")
        if not self.cycles:
            return
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10 or h < 10:
            return
        # Grille
        for y_frac in [0.25, 0.5, 0.75]:
            y = int(h * y_frac)
            c.create_line(0, y, w, y, fill=COLORS["grid"], width=1)
        # Ligne zéro
        c.create_line(0, h//2, w, h//2, fill=COLORS["muted"], width=1, dash=(4,4))
        # Waveform
        samples = self.cycles[self.current_idx]
        pad = 10
        pts = []
        for i, s in enumerate(samples):
            x = pad + (i / (len(samples)-1)) * (w - 2*pad)
            y = h//2 - s * (h//2 - pad)
            pts.extend([x, y])
        if len(pts) >= 4:
            c.create_line(*pts, fill=COLORS["wave"], width=1.5, smooth=True)

    def _draw_fft(self):
        c = self.fft_canvas
        c.delete("all")
        if not self.cycles:
            return
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10 or h < 10:
            return
        _, fft = classify_cycle(self.cycles[self.current_idx])
        n = min(len(fft), 12)
        pad = 12
        slot_w = (w - 2*pad) / n
        bar_w  = max(4, int(slot_w * 0.7))
        labels = ["F","2","3","4","5","6","7","8","9","10","11","12"]
        for i in range(n):
            bh = int(fft[i] * (h - 32))
            x  = int(pad + i * slot_w + (slot_w - bar_w) / 2)
            color = COLORS["highlight"] if i == 0 else COLORS["fft"]
            c.create_rectangle(x, h-20-bh, x+bar_w, h-20,
                                fill=color, outline="")
            c.create_text(x + bar_w//2, h-8, text=labels[i],
                          font=("Consolas", 8), fill=COLORS["muted"])

    def _build_thumbs(self):
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        for i, cycle in enumerate(self.cycles):
            bg = COLORS["highlight"] if i == self.current_idx else COLORS["panel"]
            frame = tk.Frame(self.thumb_frame, bg=bg, padx=1, pady=1)
            frame.pack(side="left", padx=2)
            thumb = tk.Canvas(frame, width=48, height=40, bg=COLORS["panel"],
                               highlightthickness=0, cursor="hand2")
            thumb.pack()
            label, _ = classify_cycle(cycle)
            # Dessiner miniature
            pts = []
            for j, s in enumerate(cycle):
                x = (j / (len(cycle)-1)) * 48
                y = 20 - s * 16
                pts.extend([x, y])
            if len(pts) >= 4:
                color = LABEL_COLORS.get(label, COLORS["muted"])
                thumb.create_line(*pts, fill=color, width=1)
            idx = i
            thumb.bind("<Button-1>", lambda e, idx=idx: self._goto_cycle(idx))
            # Label type sous miniature
            lbl_color = LABEL_COLORS.get(label, COLORS["muted"])
            tk.Label(frame, text=label[:3].upper(), font=("Consolas", 7),
                     bg=bg, fg=lbl_color).pack()

    # ── Navigation ───────────────────────────
    def _prev_cycle(self):
        if self.cycles:
            self.current_idx = (self.current_idx - 1) % len(self.cycles)
            self._refresh_display()

    def _next_cycle(self):
        if self.cycles:
            self.current_idx = (self.current_idx + 1) % len(self.cycles)
            self._refresh_display()

    def _goto_cycle(self, idx):
        self.current_idx = idx
        self._refresh_display()

    # ── Export ───────────────────────────────
    def _get_export_cycles(self):
        if not self.cycles:
            messagebox.showwarning("Export", "Aucun cycle chargé.")
            return None
        target = self.export_size.get()
        return [resample_cycle(c, target) for c in self.cycles]

    def _export_separate(self):
        cycles = self._get_export_cycles()
        if cycles is None:
            return
        folder = filedialog.askdirectory(title="Choisir le dossier d'export")
        if not folder:
            return
        base = os.path.splitext(os.path.basename(self.filepath or "wavetable"))[0]
        target = self.export_size.get()
        for i, c in enumerate(cycles):
            label, _ = classify_cycle(self.cycles[i])
            fname = f"{base}_cycle{i+1:02d}_{label}_{target}.wav"
            write_wav(os.path.join(folder, fname), c, self.sr)
        self.status_var.set(f"Export OK : {len(cycles)} fichiers WAV dans {folder}")
        messagebox.showinfo("Export", f"{len(cycles)} fichiers exportés dans :\n{folder}")

    def _export_unified(self):
        cycles = self._get_export_cycles()
        if cycles is None:
            return
        target = self.export_size.get()
        base = os.path.splitext(os.path.basename(self.filepath or "wavetable"))[0]
        default_name = f"{base}_{len(cycles)}cycles_{target}.wav"
        path = filedialog.asksaveasfilename(
            title="Exporter WAV unifié",
            initialfile=default_name,
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        unified = np.concatenate(cycles)
        write_wav(path, unified, self.sr)
        self.status_var.set(f"Export OK : {os.path.basename(path)} ({len(cycles)} cycles × {target} samples)")
        messagebox.showinfo("Export", f"WAV unifié exporté :\n{path}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = WavetableTool()
    app.mainloop()
