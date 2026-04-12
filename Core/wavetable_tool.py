"""
Wavetable Analyzer & Converter  —  v5
Analyzes, visualizes and exports WAV wavetable files.

Supported WAV formats : PCM int 8 / 16 / 24 / 32-bit,  IEEE float 32-bit
Detected WT chunks    : clm  (Serum / Deluge / Vital)
                        srge (Surge XT)
                        uhWT (u-he: Hive, Zebra)

Modes:
  Open File       — load a single wavetable bank
  Open Waveforms  — pick multiple single-cycle WAVs, assemble into one bank
  Open Banks      — pick multiple wavetable banks, browse one page per bank

Layout:
  Part A — top toolbar  (mode buttons + Clear)
  Part B — left panel   (global settings, file info, export controls)
  Part C — right area   (oscilloscope, FFT, cycle thumbnails)
  Part D — status bar   (aligned with Part C, not full-width)

Dependencies : numpy
Launch       : uv run --with numpy wavetable_tool.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import struct
import io
import os
import wave
import numpy as np


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
CYCLE_SIZES  = [256, 512, 1024, 2048]
EXPORT_SIZES = [256, 512, 1024, 2048]

# Dark theme palette
C = {
    "bg":        "#1a1a2e",
    "panel":     "#16213e",
    "accent":    "#0f3460",
    "hot":       "#e94560",   # active mode highlight
    "text":      "#eaeaea",
    "muted":     "#7a7a9a",
    "wave":      "#4fc3f7",
    "fft":       "#81c784",
    "grid":      "#2a2a4a",
}

LABEL_COLORS = {
    "sin":       "#4fc3f7",
    "triangle":  "#ce93d8",
    "square":    "#81c784",
    "saw":       "#ffb74d",
    "undefined": "#e0a060",
    "complex":   "#7a7a9a",
}

CLM_PAYLOAD_SIZE = 30  # bytes in our written clm chunk payload


# ---------------------------------------------------------------------------
#  WAV reading  (PCM 8/16/24/32-bit + IEEE float 32-bit)
# ---------------------------------------------------------------------------
def _decode_pcm24(data: bytes) -> np.ndarray:
    """Convert raw 24-bit PCM bytes to float32 in [-1, 1]."""
    raw    = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
    sign   = ((raw[:, 2] & 0x80) >> 7).astype(np.uint8) * 0xff
    padded = np.column_stack([raw, sign.reshape(-1, 1)]).flatten()
    return np.frombuffer(padded.tobytes(), dtype=np.int32).astype(np.float32) / 8388608.0


def read_wav(path: str) -> tuple:
    """
    Read a WAV file and return (audio_f32, sample_rate, bit_depth, chunk_info).

    audio_f32  : mono float32 ndarray in [-1, 1]
    chunk_info : dict with keys 'clm ', 'srge', 'uhWT' (bytes or None)
    """
    with open(path, "rb") as f:
        raw = f.read()

    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("Not a valid WAV/RIFF file.")

    fmt_off   = raw.find(b"fmt ")
    if fmt_off == -1:
        raise ValueError("No 'fmt ' chunk found.")
    audio_fmt = struct.unpack("<H", raw[fmt_off +  8: fmt_off + 10])[0]
    channels  = struct.unpack("<H", raw[fmt_off + 10: fmt_off + 12])[0]
    sr        = struct.unpack("<I", raw[fmt_off + 12: fmt_off + 16])[0]
    bit_depth = struct.unpack("<H", raw[fmt_off + 22: fmt_off + 24])[0]
    sampwidth = bit_depth // 8

    if audio_fmt not in (1, 3):
        raise ValueError(
            f"Unsupported WAV format code {audio_fmt}. "
            "Only PCM integer (1) and IEEE float (3) are supported.")

    data_off  = raw.find(b"data")
    if data_off == -1:
        raise ValueError("No 'data' chunk found.")
    data_size  = struct.unpack("<I", raw[data_off + 4: data_off + 8])[0]
    data_bytes = raw[data_off + 8: data_off + 8 + data_size]

    if audio_fmt == 3:
        audio = np.frombuffer(data_bytes, dtype=np.float32).copy()
    elif bit_depth == 24:
        audio = _decode_pcm24(data_bytes)
    else:
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth)
        if dtype is None:
            raise ValueError(f"Unsupported PCM bit depth: {bit_depth}")
        audio = np.frombuffer(data_bytes, dtype=dtype).astype(np.float32)
        audio /= np.iinfo(dtype).max

    if channels == 2:
        audio = audio[::2]   # keep left channel

    # Scan for known wavetable metadata chunks
    chunk_info: dict = {"clm ": None, "srge": None, "uhWT": None}
    pos = 12
    while pos < len(raw) - 8:
        cid  = raw[pos: pos + 4]
        if len(cid) < 4:
            break
        size = struct.unpack("<I", raw[pos + 4: pos + 8])[0]
        key  = cid.decode("ascii", errors="replace")
        if key in chunk_info:
            chunk_info[key] = raw[pos + 8: pos + 8 + size]
        pos += 8 + size
        if size == 0:
            pos += 1

    return audio, sr, bit_depth, chunk_info


# ---------------------------------------------------------------------------
#  Wavetable chunk parsers
# ---------------------------------------------------------------------------
def parse_clm(payload) -> int | None:
    """Extract cycle size from a 'clm ' chunk payload (Serum/Deluge/Vital)."""
    if not payload:
        return None
    text = payload.decode("ascii", errors="ignore").strip()
    if text.startswith("<!>"):
        try:
            return int(text[3:].split()[0])
        except (ValueError, IndexError):
            pass
    return None


def parse_srge(payload) -> int | None:
    """Extract cycle size from a 'srge' chunk (Surge XT): uint32 at offset 0."""
    if payload and len(payload) >= 4:
        try:
            return struct.unpack("<I", payload[:4])[0]
        except struct.error:
            pass
    return None


def best_chunk_cycle_size(chunk_info: dict) -> tuple:
    """Return (cycle_size | None, source_label) from known WT chunks."""
    cs = parse_clm(chunk_info.get("clm "))
    if cs:
        return cs, "clm"
    cs = parse_srge(chunk_info.get("srge"))
    if cs:
        return cs, "srge"
    if chunk_info.get("uhWT"):
        return None, "uhWT"
    return None, ""


# ---------------------------------------------------------------------------
#  CLM chunk writer
# ---------------------------------------------------------------------------
def build_clm_chunk(cycle_size: int) -> bytes:
    marker  = f"<!>{cycle_size}".encode("ascii")
    payload = marker + b" " * (CLM_PAYLOAD_SIZE - len(marker))
    return b"clm " + struct.pack("<I", CLM_PAYLOAD_SIZE) + payload


def write_wav_with_clm(path: str, audio: np.ndarray, sr: int, cs: int):
    buf = io.BytesIO()
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    raw      = buf.getvalue()
    data_off = raw.find(b"data")
    if data_off == -1:
        with open(path, "wb") as f: f.write(raw)
        return
    clm     = build_clm_chunk(cs)
    new_raw = raw[:data_off] + clm + raw[data_off:]
    new_raw = new_raw[:4] + struct.pack("<I", len(new_raw) - 8) + new_raw[8:]
    with open(path, "wb") as f: f.write(new_raw)


def write_wav_plain(path: str, audio: np.ndarray, sr: int):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
#  Audio analysis
# ---------------------------------------------------------------------------
def resample_cycle(cycle: np.ndarray, target: int) -> np.ndarray:
    if len(cycle) == target:
        return cycle.copy()
    return np.interp(
        np.linspace(0, 1, target,     endpoint=False),
        np.linspace(0, 1, len(cycle), endpoint=False),
        cycle)


def detect_cycle_size(audio: np.ndarray) -> tuple:
    """Cosine-similarity based cycle size detection. Returns (best, scores)."""
    scores = {}
    for cs in CYCLE_SIZES:
        n = len(audio) // cs
        if n < 2:
            continue
        slices = [audio[i * cs:(i + 1) * cs] for i in range(min(n, 4))]
        sims   = []
        for i in range(len(slices) - 1):
            a, b = slices[i], slices[i + 1]
            norm = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
            if norm > 0:
                sims.append(abs(float(np.dot(a, b))) / norm)
        scores[cs] = float(np.mean(sims)) if sims else 0.0
    if not scores:
        return 2048, {}
    return max(scores, key=scores.get), scores


def classify_cycle(cycle: np.ndarray) -> tuple:
    """
    Classify via FFT harmonic analysis.
    Returns (label, fft_norm[:16]).

    Thresholds:
      sin       — fund dominant, H2 < 5 %, H3 < 5 %
      triangle  — odd-only (>85 %), H5/H3 < 0.45  (theoretical: (3/5)² = 0.36)
      square    — odd-only (>80 %), H5/H3 ≥ 0.45  (theoretical: 3/5 = 0.60)
      saw       — even harmonics present (>20 %)
      undefined — some harmonic content but pattern unclear
      complex   — broadband / noise-like
    """
    fft = np.abs(np.fft.rfft(cycle))
    if fft.max() == 0:
        return "complex", fft[:16]
    fft_n = fft / fft.max()
    fund  = float(fft[1]) if len(fft) > 1 else 1.0
    h2    = float(fft[2]) if len(fft) > 2 else 0.0
    h3    = float(fft[3]) if len(fft) > 3 else 0.0
    h5    = float(fft[5]) if len(fft) > 5 else 0.0
    total = float(np.sum(fft[1:10])) if len(fft) > 10 else 1.0
    odds  = float(sum(fft[k] for k in range(1, 10, 2) if k < len(fft)))
    evens = float(sum(fft[k] for k in range(2, 10, 2) if k < len(fft)))
    odd_r  = odds / total if total > 0 else 0.0
    even_r = evens / (odds + evens + 1e-9)

    if fund > 0 and h2 / fund < 0.05 and h3 / fund < 0.05:
        return "sin", fft_n[:16]
    if odd_r > 0.85 and h3 > 0 and h5 / h3 < 0.45:
        return "triangle", fft_n[:16]
    if odd_r > 0.80 and h3 > 0 and h5 / h3 >= 0.45:
        return "square", fft_n[:16]
    if even_r > 0.20 and total / fft.max() > 0.3:
        return "saw", fft_n[:16]
    if total / fft.max() > 0.15:
        return "undefined", fft_n[:16]
    return "complex", fft_n[:16]


# ---------------------------------------------------------------------------
#  Data model
# ---------------------------------------------------------------------------
class Bank:
    """One loaded wavetable file (or assembled waveform collection)."""
    def __init__(self, path, audio, sr, bit_depth, chunk_info):
        self.path       = path
        self.audio      = audio
        self.sr         = sr
        self.bit_depth  = bit_depth
        self.chunk_info = chunk_info
        self.cycle_size = 2048
        self.cycles: list[np.ndarray] = []

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    def slice(self, cs: int):
        self.cycle_size = cs
        n = len(self.audio) // cs
        self.cycles = [self.audio[i * cs:(i + 1) * cs] for i in range(n)]


# ---------------------------------------------------------------------------
#  Application
# ---------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wavetable Analyzer & Converter  v5")
        self.configure(bg=C["bg"])
        self.geometry("1100x780")
        self.minsize(900, 640)

        # State
        self.banks:     list[Bank] = []
        self.bank_idx:  int        = 0
        self.cycle_idx: int        = 0
        self.mode:      str        = ""   # "file" | "waveforms" | "banks"

        # Tkinter vars (global settings)
        self.cs_var          = tk.IntVar(value=2048)
        self.export_size_var = tk.IntVar(value=2048)
        self.export_n_var    = tk.IntVar(value=0)
        self.export_clm_var  = tk.BooleanVar(value=True)

        self._build()

    # ── convenience ─────────────────────────────────────────────────────────
    @property
    def bank(self) -> Bank | None:
        return self.banks[self.bank_idx] if self.banks else None

    @property
    def cycles(self) -> list:
        return self.bank.cycles if self.bank else []

    # ── UI construction ──────────────────────────────────────────────────────
    def _build(self):
        # ── Part A — top toolbar ────────────────────────────────────────────
        self.toolbar = tk.Frame(self, bg=C["panel"], pady=6)
        self.toolbar.pack(fill="x", padx=0, pady=0)

        # Title
        tk.Label(self.toolbar, text="WAVETABLE ANALYZER",
                 font=("Consolas", 13, "bold"),
                 bg=C["panel"], fg=C["hot"]).pack(side="left", padx=(12, 4))
        tk.Label(self.toolbar, text="v5",
                 font=("Consolas", 11),
                 bg=C["panel"], fg=C["muted"]).pack(side="left", padx=(0, 16))

        # Mode buttons stored for highlight management
        self.mode_btns = {}
        for label, key in [("Open File", "file"),
                            ("Open Waveforms", "waveforms"),
                            ("Open Banks", "banks")]:
            b = tk.Button(self.toolbar, text=label,
                          font=("Consolas", 10),
                          bg=C["accent"], fg=C["text"],
                          activebackground=C["hot"],
                          activeforeground="#fff",
                          relief="flat", bd=0, padx=12, pady=5,
                          cursor="hand2",
                          command=lambda k=key: self._open_mode(k))
            b.pack(side="left", padx=4)
            self.mode_btns[key] = b

        # Clear — right side
        tk.Button(self.toolbar, text="Clear",
                  font=("Consolas", 10),
                  bg=C["hot"], fg="#fff",
                  activebackground="#c0304a",
                  activeforeground="#fff",
                  relief="flat", bd=0, padx=12, pady=5,
                  cursor="hand2",
                  command=self._clear).pack(side="right", padx=12)

        # ── Body: Part B (left) + Part C+D (right) ──────────────────────────
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True)

        # Part B — left panel
        self.panel_b = tk.Frame(body, bg=C["panel"], width=230)
        self.panel_b.pack(side="left", fill="y")
        self.panel_b.pack_propagate(False)
        self._build_panel_b()

        # Right column: Part C + Part D stacked
        right_col = tk.Frame(body, bg=C["bg"])
        right_col.pack(side="left", fill="both", expand=True)

        # Part C — visualisation
        self.panel_c = tk.Frame(right_col, bg=C["bg"])
        self.panel_c.pack(fill="both", expand=True, padx=10, pady=(8, 4))
        self._build_panel_c()

        # Part D — status bar (aligned with Part C, not full-width)
        self.status_var = tk.StringVar(value="Load a WAV file to get started.")
        tk.Label(right_col, textvariable=self.status_var,
                 font=("Consolas", 9),
                 bg=C["accent"], fg=C["text"],
                 anchor="w", padx=10).pack(fill="x", padx=10, pady=(0, 6))

    def _build_panel_b(self):
        p = self.panel_b

        # FILE INFO
        self._lbl_section(p, "FILE INFO")
        self.file_lbl = tk.Label(p, text="No file loaded",
                                 font=("Consolas", 9), bg=C["panel"], fg=C["text"],
                                 wraplength=210, justify="left")
        self.file_lbl.pack(anchor="w", padx=10, pady=(0, 4))

        # Bank navigator — only shown in Open Banks mode
        self.bank_nav_frame = tk.Frame(p, bg=C["panel"])
        self.bank_nav_frame.pack(fill="x", padx=10, pady=(0, 4))
        self._sbtn("◀", self._prev_bank).pack(side="left", in_=self.bank_nav_frame)
        self.bank_nav_lbl = tk.Label(
            self.bank_nav_frame, text="",
            font=("Consolas", 9, "bold"),
            bg=C["panel"], fg=C["hot"], padx=8)
        self.bank_nav_lbl.pack(side="left", in_=self.bank_nav_frame)
        self._sbtn("▶", self._next_bank).pack(side="left", in_=self.bank_nav_frame)
        self.bank_nav_frame.pack_forget()

        self._sep(p)

        # WT METADATA
        self._lbl_section(p, "WT METADATA")
        self.meta_lbl = tk.Label(p, text="—",
                                 font=("Consolas", 9), bg=C["panel"], fg=C["wave"],
                                 wraplength=210, justify="left")
        self.meta_lbl.pack(anchor="w", padx=10, pady=(0, 4))
        self._sep(p)

        # ANALYSIS CYCLE SIZE
        self._lbl_section(p, "ANALYSIS CYCLE SIZE")
        self.detect_lbl = tk.Label(p, text="—",
                                   font=("Consolas", 9), bg=C["panel"], fg=C["muted"])
        self.detect_lbl.pack(anchor="w", padx=10, pady=(0, 2))
        for cs in CYCLE_SIZES:
            tk.Radiobutton(p, text=f"{cs} samples",
                           variable=self.cs_var, value=cs,
                           command=self._on_cs_change,
                           bg=C["panel"], fg=C["text"],
                           selectcolor=C["accent"],
                           activebackground=C["panel"],
                           font=("Consolas", 9)).pack(anchor="w", padx=10)
        self._sep(p)

        # FILE INFO details
        self._lbl_section(p, "FILE INFO")
        self.info_lbl = tk.Label(p, text="—",
                                 font=("Consolas", 9), bg=C["panel"], fg=C["text"],
                                 justify="left", wraplength=210)
        self.info_lbl.pack(anchor="w", padx=10)
        self._sep(p)

        # EXPORT
        self._lbl_section(p, "EXPORT")

        tk.Label(p, text="Output cycle size:",
                 font=("Consolas", 9), bg=C["panel"], fg=C["text"]).pack(
                     anchor="w", padx=10)
        ttk.Combobox(p, textvariable=self.export_size_var,
                     values=EXPORT_SIZES, state="readonly", width=9,
                     font=("Consolas", 9)).pack(anchor="w", padx=10, pady=(2, 8))

        # Cycles to export: label + [−] N [+] on one row
        row = tk.Frame(p, bg=C["panel"])
        row.pack(fill="x", padx=10, pady=(0, 2))
        tk.Label(row, text="Cycles:", font=("Consolas", 9),
                 bg=C["panel"], fg=C["text"]).pack(side="left")
        self._sbtn("−", self._dec_n).pack(side="left", padx=(6, 2))
        tk.Label(row, textvariable=self.export_n_var, width=3,
                 font=("Consolas", 10, "bold"),
                 bg=C["panel"], fg=C["hot"]).pack(side="left")
        self._sbtn("+", self._inc_n).pack(side="left", padx=(2, 0))
        tk.Label(p, text="(0 = all)", font=("Consolas", 8),
                 bg=C["panel"], fg=C["muted"]).pack(anchor="w", padx=10, pady=(0, 6))

        self._sep(p)

        # WAVETABLE HEADER
        self._lbl_section(p, "WAVETABLE HEADER")
        tk.Checkbutton(p, text="Write 'clm' chunk",
                       variable=self.export_clm_var,
                       command=self._on_clm_toggle,
                       bg=C["panel"], fg=C["text"],
                       selectcolor=C["accent"],
                       activebackground=C["panel"],
                       font=("Consolas", 9)).pack(anchor="w", padx=10)
        self.clm_desc_lbl = tk.Label(p, text=self._clm_text(),
                                     font=("Consolas", 8), bg=C["panel"], fg=C["muted"],
                                     wraplength=210, justify="left")
        self.clm_desc_lbl.pack(anchor="w", padx=10, pady=(0, 6))
        self._sep(p)

        # Export buttons
        for txt, cmd in [("Export current cycle",    self._exp_solo),
                         ("Export separate WAVs",    self._exp_separate),
                         ("Export unified WAV",      self._exp_unified)]:
            self._btn(p, txt, cmd).pack(fill="x", padx=10, pady=2)

        # "Export all banks" — only shown in Open Banks mode
        self.exp_all_btn = self._btn(p, "Export all banks", self._exp_all_banks)
        self.exp_all_btn.pack(fill="x", padx=10, pady=2)
        self.exp_all_btn.pack_forget()

    def _build_panel_c(self):
        p = self.panel_c

        # ── Oscilloscope + FFT ──
        vis_row = tk.Frame(p, bg=C["bg"])
        vis_row.pack(fill="both", expand=True)
        vis_row.columnconfigure(0, weight=3)
        vis_row.columnconfigure(1, weight=2)
        vis_row.rowconfigure(0, weight=1)

        wf = tk.Frame(vis_row, bg=C["panel"])
        wf.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(wf, text="OSCILLOSCOPE", font=("Consolas", 8),
                 bg=C["panel"], fg=C["muted"]).pack(anchor="w", padx=8, pady=(4, 0))
        self.wave_cv = tk.Canvas(wf, bg=C["panel"], highlightthickness=0)
        self.wave_cv.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.wave_cv.bind("<Configure>", lambda e: self._draw_wave())

        ff = tk.Frame(vis_row, bg=C["panel"])
        ff.grid(row=0, column=1, sticky="nsew")
        tk.Label(ff, text="FFT SPECTRUM", font=("Consolas", 8),
                 bg=C["panel"], fg=C["muted"]).pack(anchor="w", padx=8, pady=(4, 0))
        self.fft_cv = tk.Canvas(ff, bg=C["panel"], highlightthickness=0)
        self.fft_cv.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.fft_cv.bind("<Configure>", lambda e: self._draw_fft())

        # ── Cycle label + badge ──
        info_row = tk.Frame(p, bg=C["bg"])
        info_row.pack(fill="x", pady=(4, 0))
        self.cycle_nav_lbl = tk.Label(info_row, text="— / —",
                                      font=("Consolas", 11, "bold"),
                                      bg=C["bg"], fg=C["text"])
        self.cycle_nav_lbl.pack(side="left")
        self.cycle_badge = tk.Label(info_row, text="",
                                    font=("Consolas", 10, "bold"),
                                    bg=C["bg"], fg=C["hot"], padx=10)
        self.cycle_badge.pack(side="left")
        # Prev / Next cycle buttons
        self._sbtn("◀", self._prev_cycle).pack(side="left", padx=(12, 2))
        self._sbtn("▶", self._next_cycle).pack(side="left")

        # ── ALL CYCLES thumbnails ──
        tk.Label(p, text="ALL CYCLES",
                 font=("Consolas", 8), bg=C["bg"], fg=C["muted"]).pack(
                     anchor="w", pady=(6, 2))
        thumb_outer = tk.Frame(p, bg=C["bg"], height=72)
        thumb_outer.pack(fill="x")
        thumb_outer.pack_propagate(False)
        self.thumb_cv = tk.Canvas(thumb_outer, bg=C["bg"],
                                  highlightthickness=0, height=72)
        sb = ttk.Scrollbar(thumb_outer, orient="horizontal",
                           command=self.thumb_cv.xview)
        self.thumb_cv.configure(xscrollcommand=sb.set)
        sb.pack(side="bottom", fill="x")
        self.thumb_cv.pack(fill="both", expand=True)
        self.thumb_frame = tk.Frame(self.thumb_cv, bg=C["bg"])
        self.thumb_cv.create_window((0, 0), window=self.thumb_frame, anchor="nw")
        self.thumb_frame.bind(
            "<Configure>",
            lambda e: self.thumb_cv.configure(
                scrollregion=self.thumb_cv.bbox("all")))

    # ── UI helpers ───────────────────────────────────────────────────────────
    def _btn(self, parent, text, cmd):
        return tk.Button(parent, text=text, command=cmd,
                         font=("Consolas", 9),
                         bg=C["accent"], fg=C["text"],
                         activebackground=C["hot"],
                         activeforeground="#fff",
                         relief="flat", bd=0, padx=6, pady=4, cursor="hand2")

    def _sbtn(self, text, cmd):
        return tk.Button(self, text=text, command=cmd,
                         font=("Consolas", 9),
                         bg=C["accent"], fg=C["text"],
                         activebackground=C["hot"],
                         activeforeground="#fff",
                         relief="flat", bd=0, padx=6, pady=2, cursor="hand2")

    def _lbl_section(self, parent, text):
        tk.Label(parent, text=text, font=("Consolas", 8, "bold"),
                 bg=C["panel"], fg=C["muted"]).pack(
                     anchor="w", padx=10, pady=(8, 2))

    def _sep(self, parent):
        tk.Frame(parent, bg=C["grid"], height=1).pack(fill="x", padx=10, pady=3)

    def _clm_text(self) -> str:
        if self.export_clm_var.get():
            return (f"cycle={self.export_size_var.get()}\n"
                    f"Deluge · Serum · Vital")
        return "Plain WAV — no WT header."

    def _set_mode(self, mode: str):
        """Update mode and refresh toolbar button highlights."""
        self.mode = mode
        for key, btn in self.mode_btns.items():
            btn.configure(bg=C["hot"] if key == mode else C["accent"])

    # ── Event handlers ───────────────────────────────────────────────────────
    def _on_cs_change(self):
        b = self.bank
        if b:
            b.slice(self.cs_var.get())
            self.cycle_idx = 0
            self.export_n_var.set(0)
            self._refresh()

    def _on_clm_toggle(self):
        self.clm_desc_lbl.config(text=self._clm_text())

    def _inc_n(self):
        v = self.export_n_var.get()
        if self.cycles:
            self.export_n_var.set(min(v + 1, len(self.cycles)))

    def _dec_n(self):
        self.export_n_var.set(max(0, self.export_n_var.get() - 1))

    # ── Loading ──────────────────────────────────────────────────────────────
    def _load_bank(self, path: str) -> Bank:
        audio, sr, bd, ci = read_wav(path)
        b = Bank(path, audio, sr, bd, ci)
        cs, _ = best_chunk_cycle_size(ci)
        if cs and cs in CYCLE_SIZES:
            b.slice(cs)
        else:
            best, _ = detect_cycle_size(audio)
            b.slice(best)
        return b

    def _open_mode(self, mode: str):
        if mode == "file":
            self._open_file()
        elif mode == "waveforms":
            self._open_waveforms()
        elif mode == "banks":
            self._open_banks()

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open a wavetable bank",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.banks    = [self._load_bank(path)]
            self.bank_idx = 0
            self._set_mode("file")
            self._activate(0)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _open_waveforms(self):
        paths = filedialog.askopenfilenames(
            title="Select single-cycle waveform WAV files",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not paths:
            return
        target = self.export_size_var.get()
        cycles_audio, errors = [], []
        for p in sorted(paths):
            try:
                audio, _, _, _ = read_wav(p)
                src = audio[:target] if len(audio) >= target else audio
                cycles_audio.append(resample_cycle(src, target))
            except Exception as e:
                errors.append(f"{os.path.basename(p)}: {e}")
        if errors:
            messagebox.showwarning("Some files skipped", "\n".join(errors))
        if not cycles_audio:
            messagebox.showerror("Error", "No valid files.")
            return
        b = Bank(path=sorted(paths)[0],
                 audio=np.concatenate(cycles_audio),
                 sr=44100, bit_depth=16, chunk_info={})
        b.slice(target)
        self.banks    = [b]
        self.bank_idx = 0
        self._set_mode("waveforms")
        self._activate(0)
        self.file_lbl.config(text=f"{len(cycles_audio)} waveforms assembled")

    def _open_banks(self):
        paths = filedialog.askopenfilenames(
            title="Select wavetable bank WAV files",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not paths:
            return
        loaded, errors = [], []
        for p in sorted(paths):
            try:
                loaded.append(self._load_bank(p))
            except Exception as e:
                errors.append(f"{os.path.basename(p)}: {e}")
        if errors:
            messagebox.showwarning("Some files skipped", "\n".join(errors))
        if not loaded:
            messagebox.showerror("Error", "No valid files.")
            return
        self.banks    = loaded
        self.bank_idx = 0
        self._set_mode("banks")
        self._activate(0)

    def _clear(self):
        self.banks    = []
        self.bank_idx = 0
        self.cycle_idx = 0
        self.mode = ""
        for btn in self.mode_btns.values():
            btn.configure(bg=C["accent"])
        self.file_lbl.config(text="No file loaded")
        self.meta_lbl.config(text="—")
        self.detect_lbl.config(text="—")
        self.info_lbl.config(text="—")
        self.cycle_nav_lbl.config(text="— / —")
        self.cycle_badge.config(text="")
        self.bank_nav_frame.pack_forget()
        self.exp_all_btn.pack_forget()
        self.wave_cv.delete("all")
        self.fft_cv.delete("all")
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        self.status_var.set("Cleared.")

    # ── Bank activation ──────────────────────────────────────────────────────
    def _activate(self, idx: int):
        """Switch to bank[idx] and refresh everything."""
        self.bank_idx  = idx
        self.cycle_idx = 0
        b = self.bank
        if b is None:
            return

        # Sync cycle-size radio to this bank
        self.cs_var.set(b.cycle_size)
        self.export_n_var.set(0)

        # Bank navigator visibility
        if self.mode == "banks" and len(self.banks) > 1:
            self.bank_nav_lbl.config(text=f"{idx + 1} / {len(self.banks)}")
            self.bank_nav_frame.pack(fill="x", padx=10, pady=(0, 4))
            self.exp_all_btn.pack(fill="x", padx=10, pady=2)
        else:
            self.bank_nav_frame.pack_forget()
            self.exp_all_btn.pack_forget()

        self._update_panel_b()
        self._refresh()
        self.status_var.set(
            f"{b.name}  |  {len(b.audio)} samples @ {b.sr} Hz  |  "
            f"{b.bit_depth}-bit  |  {len(b.cycles)} cycles × {b.cycle_size} samp")

    def _update_panel_b(self):
        b = self.bank
        if not b:
            return
        self.file_lbl.config(text=b.name)

        # Metadata
        ci    = b.chunk_info
        parts = []
        c = parse_clm(ci.get("clm "))
        if c:
            parts.append(f"clm  → {c} samp/cycle (Serum)")
        s = parse_srge(ci.get("srge"))
        if s:
            parts.append(f"srge → {s} samp/cycle (Surge)")
        if ci.get("uhWT"):
            parts.append("uhWT → present (u-he)")
        self.meta_lbl.config(
            text="\n".join(parts) if parts else "No WT chunk found")

        # Detection label
        cs_from_chunk, src = best_chunk_cycle_size(ci)
        if cs_from_chunk:
            self.detect_lbl.config(
                text=f"From '{src}' chunk: {cs_from_chunk}")
        else:
            _, scores = detect_cycle_size(b.audio)
            best = b.cycle_size
            tip_lines = []
            for sz in CYCLE_SIZES:
                n = len(b.audio) // sz
                tip_lines.append(
                    f"{sz}: {n} cyc  conf={scores.get(sz,0):.2f}"
                    f"{'  ◀' if sz == best else ''}")
            self.detect_lbl.config(text=f"Auto-detected: {best}")
            tip = "\n".join(tip_lines)
            self.detect_lbl.bind("<Enter>",
                                 lambda e, t=tip: self.status_var.set(t))
            self.detect_lbl.bind("<Leave>",
                                 lambda e: self._restore_status())

        # File info
        cs = b.cycle_size
        self.info_lbl.config(text=(
            f"Total  : {len(b.audio)} samples\n"
            f"Cycles : {len(b.cycles)}\n"
            f"Cycle  : {cs} samples\n"
            f"SR     : {b.sr} Hz\n"
            f"Depth  : {b.bit_depth}-bit\n"
            f"Dur.   : {cs / b.sr * 1000:.1f} ms/cycle"
        ))
        self.clm_desc_lbl.config(text=self._clm_text())

    def _restore_status(self):
        b = self.bank
        if b:
            self.status_var.set(
                f"{b.name}  |  {len(b.audio)} samples @ {b.sr} Hz  |  "
                f"{b.bit_depth}-bit  |  {len(b.cycles)} cycles × {b.cycle_size} samp")

    # ── Navigation ───────────────────────────────────────────────────────────
    def _prev_bank(self):
        if self.banks:
            self._activate((self.bank_idx - 1) % len(self.banks))

    def _next_bank(self):
        if self.banks:
            self._activate((self.bank_idx + 1) % len(self.banks))

    def _prev_cycle(self):
        if self.cycles:
            self.cycle_idx = (self.cycle_idx - 1) % len(self.cycles)
            self._refresh()

    def _next_cycle(self):
        if self.cycles:
            self.cycle_idx = (self.cycle_idx + 1) % len(self.cycles)
            self._refresh()

    def _goto_cycle(self, idx: int):
        self.cycle_idx = idx
        self._refresh()

    # ── Display ──────────────────────────────────────────────────────────────
    def _refresh(self):
        if not self.cycles:
            return
        n     = len(self.cycles)
        label, _ = classify_cycle(self.cycles[self.cycle_idx])
        self.cycle_nav_lbl.config(
            text=f"Cycle  {self.cycle_idx + 1}  /  {n}")
        self.cycle_badge.config(
            text=label.upper(),
            fg=LABEL_COLORS.get(label, C["muted"]))
        self._draw_wave()
        self._draw_fft()
        self._build_thumbs()

    def _draw_wave(self):
        cv = self.wave_cv
        cv.delete("all")
        if not self.cycles:
            return
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 10 or h < 10:
            return
        for yf in [0.25, 0.5, 0.75]:
            cv.create_line(0, int(h * yf), w, int(h * yf), fill=C["grid"])
        cv.create_line(0, h // 2, w, h // 2, fill=C["muted"], dash=(4, 4))
        s = self.cycles[self.cycle_idx]
        pad = 10
        pts = []
        for i, v in enumerate(s):
            pts.extend([pad + i / max(len(s) - 1, 1) * (w - 2 * pad),
                        h // 2 - float(v) * (h // 2 - pad)])
        if len(pts) >= 4:
            cv.create_line(*pts, fill=C["wave"], width=1.5, smooth=True)

    def _draw_fft(self):
        cv = self.fft_cv
        cv.delete("all")
        if not self.cycles:
            return
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 10 or h < 10:
            return
        _, fft = classify_cycle(self.cycles[self.cycle_idx])
        n      = min(len(fft), 12)
        pad    = 10
        slot   = (w - 2 * pad) / max(n, 1)
        bw     = max(4, int(slot * 0.7))
        lbls   = ["F","2","3","4","5","6","7","8","9","10","11","12"]
        for i in range(n):
            bh = int(float(fft[i]) * (h - 28))
            x  = int(pad + i * slot + (slot - bw) / 2)
            cv.create_rectangle(x, h - 18 - bh, x + bw, h - 18,
                                fill=C["hot"] if i == 0 else C["fft"],
                                outline="")
            cv.create_text(x + bw // 2, h - 6, text=lbls[i],
                           font=("Consolas", 7), fill=C["muted"])

    def _build_thumbs(self):
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        for i, cyc in enumerate(self.cycles):
            label, _ = classify_cycle(cyc)
            border    = C["hot"] if i == self.cycle_idx else C["panel"]
            frm = tk.Frame(self.thumb_frame, bg=border, padx=1, pady=1)
            frm.pack(side="left", padx=2)
            th = tk.Canvas(frm, width=48, height=44,
                           bg=C["panel"], highlightthickness=0, cursor="hand2")
            th.pack()
            color = LABEL_COLORS.get(label, C["muted"])
            pts   = []
            for j, v in enumerate(cyc):
                pts.extend([(j / max(len(cyc) - 1, 1)) * 48,
                            22 - float(v) * 18])
            if len(pts) >= 4:
                th.create_line(*pts, fill=color, width=1)
            idx = i
            th.bind("<Button-1>", lambda e, idx=idx: self._goto_cycle(idx))
            tk.Label(frm, text=label[:4].upper(),
                     font=("Consolas", 7), bg=border, fg=color).pack()

    # ── Export ───────────────────────────────────────────────────────────────
    def _prep_cycles(self, b: Bank | None = None):
        src = b or self.bank
        if not src or not src.cycles:
            messagebox.showwarning("Export", "No cycles loaded.")
            return None, None
        target = self.export_size_var.get()
        n      = self.export_n_var.get()
        cycs   = src.cycles[:n] if 0 < n <= len(src.cycles) else src.cycles
        return [resample_cycle(c, target) for c in cycs], src

    def _write(self, path: str, audio: np.ndarray, sr: int, cs: int):
        if self.export_clm_var.get():
            write_wav_with_clm(path, audio, sr, cs)
        else:
            write_wav_plain(path, audio, sr)

    def _exp_solo(self):
        if not self.cycles:
            messagebox.showwarning("Export", "No cycles loaded.")
            return
        b      = self.bank
        target = self.export_size_var.get()
        label, _ = classify_cycle(self.cycles[self.cycle_idx])
        suf    = "_clm" if self.export_clm_var.get() else ""
        name   = (f"{os.path.splitext(b.name)[0]}"
                  f"_cycle{self.cycle_idx + 1:02d}_{label}_{target}{suf}.wav")
        path = filedialog.asksaveasfilename(
            title="Export current cycle", initialfile=name,
            defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        self._write(path, resample_cycle(self.cycles[self.cycle_idx], target),
                    b.sr, target)
        self.status_var.set(f"Exported: {os.path.basename(path)}")

    def _exp_separate(self):
        cycs, b = self._prep_cycles()
        if not cycs:
            return
        folder = filedialog.askdirectory(title="Choose export folder")
        if not folder:
            return
        target = self.export_size_var.get()
        suf    = "_clm" if self.export_clm_var.get() else ""
        base   = os.path.splitext(b.name)[0]
        for i, c in enumerate(cycs):
            label, _ = classify_cycle(b.cycles[i])
            fname = f"{base}_cycle{i + 1:02d}_{label}_{target}{suf}.wav"
            self._write(os.path.join(folder, fname), c, b.sr, target)
        msg = f"{len(cycs)} files → {folder}"
        self.status_var.set(msg)
        messagebox.showinfo("Export complete", msg)

    def _exp_unified(self, b: Bank | None = None):
        cycs, src = self._prep_cycles(b)
        if not cycs:
            return
        target  = self.export_size_var.get()
        suf     = "_clm" if self.export_clm_var.get() else ""
        default = f"{os.path.splitext(src.name)[0]}_{len(cycs)}x{target}{suf}.wav"
        path = filedialog.asksaveasfilename(
            title="Save unified wavetable WAV", initialfile=default,
            defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        self._write(path, np.concatenate(cycs), src.sr, target)
        msg = f"{os.path.basename(path)}  |  {len(cycs)} × {target} samp"
        self.status_var.set(msg)
        messagebox.showinfo("Export complete", msg)

    def _exp_all_banks(self):
        if not self.banks:
            return
        folder = filedialog.askdirectory(title="Export all banks to folder")
        if not folder:
            return
        target = self.export_size_var.get()
        suf    = "_clm" if self.export_clm_var.get() else ""
        ok, errors = 0, []
        for bank in self.banks:
            cycs, _ = self._prep_cycles(bank)
            if not cycs:
                continue
            fname = f"{os.path.splitext(bank.name)[0]}_{len(cycs)}x{target}{suf}.wav"
            try:
                self._write(os.path.join(folder, fname),
                            np.concatenate(cycs), bank.sr, target)
                ok += 1
            except Exception as e:
                errors.append(f"{bank.name}: {e}")
        if errors:
            messagebox.showwarning("Some exports failed", "\n".join(errors))
        msg = f"{ok}/{len(self.banks)} banks exported → {folder}"
        self.status_var.set(msg)
        messagebox.showinfo("Export all complete", msg)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    App().mainloop()
