"""
Wavetable Analyzer & Converter  —  v3
Analyzes, visualizes and exports WAV wavetable files.

Supported WAV formats : PCM int (8/16/32-bit), IEEE float 32-bit
Detected WT chunks    : clm  (Serum / Deluge / Vital)
                        srge (Surge XT)
                        uhWT (u-he: Hive, Zebra…)

Dependencies : numpy
Launch       : uv run --with numpy wavetable_tool.py
               python wavetable_tool.py   (if numpy is already installed)
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
    "active":    "#1a3a5c",
}
LABEL_COLORS = {
    "sin":      "#4fc3f7",
    "square":   "#81c784",
    "saw":      "#ffb74d",
    "triangle": "#ce93d8",
    "complex":  "#7a7a9a",
}

# CLM chunk: 38 bytes total (id[4] + size_field[4] + payload[30])
CLM_PAYLOAD_SIZE = 30


# ---------------------------------------------------------------------------
#  WAV reading  (supports PCM int + IEEE float 32)
# ---------------------------------------------------------------------------
def read_wav(path: str) -> tuple:
    """
    Read a mono or stereo WAV file (PCM int or IEEE float32) and return
    (float32 mono audio array, sample_rate, detected_chunk_info_dict).

    chunk_info keys: 'clm', 'srge', 'uhWT' — each maps to the raw payload
    bytes if found, or None.
    """
    with open(path, "rb") as f:
        raw = f.read()

    # Validate RIFF / WAVE header
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("Not a valid WAV/RIFF file.")

    # Parse fmt chunk
    fmt_off = raw.find(b"fmt ")
    if fmt_off == -1:
        raise ValueError("No 'fmt ' chunk found.")
    audio_fmt  = struct.unpack("<H", raw[fmt_off + 8 : fmt_off + 10])[0]
    channels   = struct.unpack("<H", raw[fmt_off + 10: fmt_off + 12])[0]
    sr         = struct.unpack("<I", raw[fmt_off + 12: fmt_off + 16])[0]
    bit_depth  = struct.unpack("<H", raw[fmt_off + 22: fmt_off + 24])[0]
    sampwidth  = bit_depth // 8

    # Supported formats: 1 = PCM int, 3 = IEEE float
    if audio_fmt not in (1, 3):
        raise ValueError(
            f"Unsupported WAV format code {audio_fmt}. "
            f"Only PCM integer (1) and IEEE float (3) are supported."
        )

    # Parse data chunk
    data_off = raw.find(b"data")
    if data_off == -1:
        raise ValueError("No 'data' chunk found.")
    data_size  = struct.unpack("<I", raw[data_off + 4: data_off + 8])[0]
    data_bytes = raw[data_off + 8: data_off + 8 + data_size]

    # Decode samples
    if audio_fmt == 3:  # IEEE float32
        audio = np.frombuffer(data_bytes, dtype=np.float32).copy()
    else:               # PCM integer
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth)
        if dtype is None:
            raise ValueError(f"Unsupported PCM bit depth: {bit_depth}")
        audio = np.frombuffer(data_bytes, dtype=dtype).astype(np.float32)
        audio /= np.iinfo(dtype).max

    # Downmix stereo to mono (keep left channel)
    if channels == 2:
        audio = audio[::2]

    # Scan for known wavetable metadata chunks.
    # WAV chunk ids are 4 bytes; clm uses "clm " (with trailing space).
    chunk_info = {"clm ": None, "srge": None, "uhWT": None}
    pos = 12
    while pos < len(raw) - 8:
        cid  = raw[pos:pos + 4]
        if len(cid) < 4:
            break
        size = struct.unpack("<I", raw[pos + 4: pos + 8])[0]
        key  = cid.decode("ascii", errors="replace")
        if key in chunk_info:
            chunk_info[key] = raw[pos + 8: pos + 8 + size]
        pos += 8 + size
        if size == 0:
            pos += 1  # guard against infinite loop on malformed files

    return audio, sr, chunk_info


# ---------------------------------------------------------------------------
#  Wavetable chunk parsers
# ---------------------------------------------------------------------------
def parse_clm_chunk(payload: bytes) -> int | None:
    """
    Extract cycle size from a 'clm ' chunk payload (Serum / Deluge / Vital).
    Serum payloads vary in length (30 or 48 bytes depending on version);
    the number always follows the '<!>' marker and ends at the first space.
    """
    text = payload.decode("ascii", errors="ignore").strip()
    if text.startswith("<!>"):
        try:
            return int(text[3:].split()[0])
        except (ValueError, IndexError):
            pass
    return None


def parse_srge_chunk(payload: bytes) -> int | None:
    """
    Extract cycle size from a 'srge' chunk (Surge XT).
    Surge stores the cycle size as a little-endian uint32 at offset 0.
    """
    if len(payload) >= 4:
        try:
            return struct.unpack("<I", payload[:4])[0]
        except struct.error:
            pass
    return None


def detect_wt_chunk_cycle_size(chunk_info: dict) -> tuple:
    """
    Try to extract the cycle size from any known wavetable chunk.
    Returns (cycle_size: int | None, source_name: str).
    """
    # Priority: clm > srge > uhWT (uhWT format is undocumented, skip parsing)
    clm_payload = chunk_info.get("clm ")
    if clm_payload:
        cs = parse_clm_chunk(clm_payload)
        if cs:
            return cs, "clm"

    srge_payload = chunk_info.get("srge")
    if srge_payload:
        cs = parse_srge_chunk(srge_payload)
        if cs:
            return cs, "srge"

    if chunk_info.get("uhWT"):
        # uhWT format is proprietary; cycle size cannot be reliably extracted
        return None, "uhWT (proprietary — size unknown)"

    return None, ""


# ---------------------------------------------------------------------------
#  CLM chunk writer
# ---------------------------------------------------------------------------
def build_clm_chunk(cycle_size: int) -> bytes:
    """
    Build the 38-byte 'clm ' chunk used by Serum, Deluge and Vital.
    Layout: id[4] + size_field[4] + payload[30]
    Payload: '<!>NNNN' left-aligned, space-padded to 30 bytes.
    """
    marker  = f"<!>{cycle_size}".encode("ascii")
    payload = marker + b" " * (CLM_PAYLOAD_SIZE - len(marker))
    return b"clm " + struct.pack("<I", CLM_PAYLOAD_SIZE) + payload


def write_wav_with_clm(path: str, audio_f32: np.ndarray,
                       sr: int, cycle_size: int, sampwidth: int = 2) -> None:
    """
    Write a 16-bit PCM mono WAV file and inject a 'clm ' chunk between
    the fmt and data chunks so the file is recognized as a wavetable by
    Deluge, Serum and Vital.
    """
    buf    = io.BytesIO()
    maxval = np.iinfo(np.int16).max
    pcm    = (np.clip(audio_f32, -1.0, 1.0) * maxval).astype(np.int16)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(sampwidth)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    raw = buf.getvalue()

    data_off = raw.find(b"data")
    if data_off == -1:
        with open(path, "wb") as f:
            f.write(raw)
        return

    clm     = build_clm_chunk(cycle_size)
    new_raw = raw[:data_off] + clm + raw[data_off:]
    # Fix RIFF size field (bytes 4–7)
    new_raw = new_raw[:4] + struct.pack("<I", len(new_raw) - 8) + new_raw[8:]
    with open(path, "wb") as f:
        f.write(new_raw)


def write_wav_plain(path: str, audio_f32: np.ndarray, sr: int) -> None:
    """Write a plain 16-bit PCM mono WAV file without any wavetable metadata."""
    maxval = np.iinfo(np.int16).max
    pcm    = (np.clip(audio_f32, -1.0, 1.0) * maxval).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
#  Audio analysis helpers
# ---------------------------------------------------------------------------
def resample_cycle(cycle: np.ndarray, target: int) -> np.ndarray:
    """Linearly resample one waveform cycle to a different sample count."""
    if len(cycle) == target:
        return cycle.copy()
    x_old = np.linspace(0, 1, len(cycle), endpoint=False)
    x_new = np.linspace(0, 1, target,     endpoint=False)
    return np.interp(x_new, x_old, cycle)


def detect_cycle_size(audio: np.ndarray) -> tuple:
    """
    Estimate the most likely wavetable cycle size via cosine similarity
    between successive candidate cycles.
    Returns (best_size: int, scores: dict[int, float]).
    """
    scores = {}
    for cs in CYCLE_SIZES:
        n = len(audio) // cs
        if n < 2:
            continue
        cycles = [audio[i * cs:(i + 1) * cs] for i in range(min(n, 4))]
        sims   = []
        for i in range(len(cycles) - 1):
            a, b = cycles[i], cycles[i + 1]
            norm = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
            if norm > 0:
                sims.append(abs(float(np.dot(a, b))) / norm)
        scores[cs] = float(np.mean(sims)) if sims else 0.0
    if not scores:
        return 2048, {}
    return max(scores, key=scores.get), scores


def classify_cycle(cycle: np.ndarray) -> tuple:
    """
    Classify a waveform cycle as sin / square / saw / triangle / complex
    via FFT harmonic analysis.
    Returns (label: str, normalized_fft_array[:16]).
    """
    fft = np.abs(np.fft.rfft(cycle))
    if fft.max() == 0:
        return "complex", fft[:16]
    fft_norm = fft / fft.max()
    fund  = float(fft[1]) if len(fft) > 1 else 1.0
    total = float(sum(fft[1:10])) if len(fft) > 10 else 1.0
    odds  = float(sum(fft[k] for k in range(1, 10, 2) if k < len(fft)))
    evens = float(sum(fft[k] for k in range(2, 10, 2) if k < len(fft)))
    h3    = float(fft[3]) if len(fft) > 3 else 0.0
    h5    = float(fft[5]) if len(fft) > 5 else 0.0
    odd_ratio = odds / total if total > 0 else 0.0
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


# ---------------------------------------------------------------------------
#  Main application
# ---------------------------------------------------------------------------
class WavetableTool(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wavetable Analyzer & Converter  v3")
        self.configure(bg=COLORS["bg"])
        self.geometry("1080x800")
        self.minsize(860, 640)

        # Current file state
        self.audio       = None
        self.sr          = 44100
        self.chunk_info  = {}
        self.cycle_size  = tk.IntVar(value=2048)
        self.current_idx = 0
        self.cycles      = []
        self.filepath    = None

        # Multi-file list  [(path, audio, sr, chunk_info), ...]
        self.file_list      = []
        self.file_list_idx  = 0

        # Mode
        self.mode = tk.StringVar(value="single")

        # Export options
        self.export_size     = tk.IntVar(value=2048)
        self.export_n_cycles = tk.IntVar(value=0)
        self.export_clm      = tk.BooleanVar(value=True)

        self._build_ui()

    # -----------------------------------------------------------------------
    #  UI construction
    # -----------------------------------------------------------------------
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=COLORS["bg"], pady=8)
        hdr.pack(fill="x", padx=16)
        tk.Label(hdr, text="WAVETABLE ANALYZER",
                 font=("Consolas", 14, "bold"),
                 bg=COLORS["bg"], fg=COLORS["highlight"]).pack(side="left")
        tk.Label(hdr, text="& CONVERTER  v3",
                 font=("Consolas", 14),
                 bg=COLORS["bg"], fg=COLORS["muted"]).pack(side="left", padx=(4, 0))

        # Mode selector
        mode_bar = tk.Frame(self, bg=COLORS["panel"], pady=6)
        mode_bar.pack(fill="x", padx=16, pady=(0, 8))
        modes = [
            ("  Single file  ",       "single"),
            ("  Multiple files  ",    "multi"),
            ("  Batch folder → WT  ", "batch"),
        ]
        for label, val in modes:
            tk.Radiobutton(mode_bar, text=label, variable=self.mode, value=val,
                           command=self._on_mode_change,
                           bg=COLORS["panel"], fg=COLORS["text"],
                           selectcolor=COLORS["accent"],
                           activebackground=COLORS["panel"],
                           font=("Consolas", 10), indicatoron=False,
                           relief="flat", padx=10, pady=4,
                           bd=0, highlightthickness=0).pack(side="left", padx=4)

        # Body
        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        self.left = tk.Frame(body, bg=COLORS["panel"], width=280)
        self.left.pack(side="left", fill="y", padx=(0, 8))
        self.left.pack_propagate(False)
        self._build_left_panel()

        self.right = tk.Frame(body, bg=COLORS["bg"])
        self.right.pack(side="left", fill="both", expand=True)
        self._build_right_panel()

        # Status bar
        self.status_var = tk.StringVar(value="Load a WAV file to get started.")
        tk.Label(self, textvariable=self.status_var,
                 font=("Consolas", 9), bg=COLORS["accent"], fg=COLORS["text"],
                 anchor="w", padx=10).pack(fill="x", side="bottom")

    def _build_left_panel(self):
        # ── FILE ──────────────────────────────
        self._section("FILE")
        self.file_label = tk.Label(
            self.left, text="No file loaded",
            font=("Consolas", 9), bg=COLORS["panel"], fg=COLORS["text"],
            wraplength=255, justify="left")
        self.file_label.pack(anchor="w", padx=12, pady=(0, 6))
        self._btn("Open WAV...", self._open_file).pack(fill="x", padx=12, pady=2)

        # Multi-file navigator (hidden until multi mode)
        self.multi_nav_frame = tk.Frame(self.left, bg=COLORS["panel"])
        self.multi_nav_frame.pack(fill="x", padx=12, pady=(4, 0))
        self._btn("◀", self._prev_file, small=True).pack(side="left", in_=self.multi_nav_frame)
        self.multi_nav_label = tk.Label(
            self.multi_nav_frame, text="", font=("Consolas", 9),
            bg=COLORS["panel"], fg=COLORS["text"], padx=6)
        self.multi_nav_label.pack(side="left")
        self._btn("▶", self._next_file, small=True).pack(side="left", in_=self.multi_nav_frame)
        self.multi_nav_frame.pack_forget()  # hidden by default

        self._sep()

        # ── METADATA ──────────────────────────
        self._section("WAVETABLE METADATA")
        self.meta_label = tk.Label(
            self.left, text="—", font=("Consolas", 9),
            bg=COLORS["panel"], fg=COLORS["wave"],
            wraplength=255, justify="left")
        self.meta_label.pack(anchor="w", padx=12, pady=(0, 4))
        self._sep()

        # ── ANALYSIS CYCLE SIZE ───────────────
        self._section("ANALYSIS CYCLE SIZE")
        self.detect_label = tk.Label(
            self.left, text="—", font=("Consolas", 9),
            bg=COLORS["panel"], fg=COLORS["muted"])
        self.detect_label.pack(anchor="w", padx=12, pady=(0, 4))
        for cs in CYCLE_SIZES:
            tk.Radiobutton(
                self.left, text=f"{cs} samples",
                variable=self.cycle_size, value=cs,
                command=self._on_cycle_size_change,
                bg=COLORS["panel"], fg=COLORS["text"],
                selectcolor=COLORS["accent"],
                activebackground=COLORS["panel"],
                font=("Consolas", 10)).pack(anchor="w", padx=12)
        self._sep()

        # ── FILE INFO ─────────────────────────
        self._section("FILE INFO")
        self.info_label = tk.Label(
            self.left, text="—", font=("Consolas", 9),
            bg=COLORS["panel"], fg=COLORS["text"],
            justify="left", wraplength=255)
        self.info_label.pack(anchor="w", padx=12)
        self._sep()

        # ── EXPORT ────────────────────────────
        self._section("EXPORT")

        # Output cycle size
        tk.Label(self.left, text="Output cycle size:",
                 font=("Consolas", 9), bg=COLORS["panel"],
                 fg=COLORS["text"]).pack(anchor="w", padx=12)
        ttk.Combobox(
            self.left, textvariable=self.export_size,
            values=EXPORT_SIZES, state="readonly", width=10,
            font=("Consolas", 10)).pack(anchor="w", padx=12, pady=(2, 10))

        # Number of cycles — label + spinner on the SAME row
        n_row = tk.Frame(self.left, bg=COLORS["panel"])
        n_row.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(n_row, text="Cycles to export:",
                 font=("Consolas", 9), bg=COLORS["panel"],
                 fg=COLORS["text"]).pack(side="left")
        self._btn("−", self._dec_n, small=True).pack(side="left", padx=(8, 2))
        tk.Label(n_row, textvariable=self.export_n_cycles, width=3,
                 font=("Consolas", 11, "bold"),
                 bg=COLORS["panel"], fg=COLORS["highlight"]).pack(side="left")
        self._btn("+", self._inc_n, small=True).pack(side="left", padx=(2, 0))
        tk.Label(self.left, text="(0 = all cycles)",
                 font=("Consolas", 8), bg=COLORS["panel"],
                 fg=COLORS["muted"]).pack(anchor="w", padx=12, pady=(0, 8))

        self._sep()

        # ── WAVETABLE HEADER ──────────────────
        self._section("WAVETABLE HEADER")
        clm_row = tk.Frame(self.left, bg=COLORS["panel"])
        clm_row.pack(anchor="w", padx=12, pady=(0, 2))
        tk.Checkbutton(
            clm_row, text="Write 'clm' chunk",
            variable=self.export_clm, command=self._on_clm_toggle,
            bg=COLORS["panel"], fg=COLORS["text"],
            selectcolor=COLORS["accent"],
            activebackground=COLORS["panel"],
            font=("Consolas", 10)).pack(side="left")
        self.clm_desc = tk.Label(
            self.left, text=self._clm_text(),
            font=("Consolas", 8), bg=COLORS["panel"], fg=COLORS["muted"],
            wraplength=255, justify="left")
        self.clm_desc.pack(anchor="w", padx=12, pady=(0, 8))
        self._sep()

        # ── EXPORT BUTTONS ────────────────────
        self._btn("Export separate WAVs", self._export_separate).pack(
            fill="x", padx=12, pady=2)
        self._btn("Export unified WAV",   self._export_unified).pack(
            fill="x", padx=12, pady=2)

    def _build_right_panel(self):
        # Cycle navigation
        nav = tk.Frame(self.right, bg=COLORS["bg"])
        nav.pack(fill="x", pady=(0, 6))
        self._btn("◀ Prev", self._prev_cycle, small=True).pack(side="left")
        self.nav_label = tk.Label(
            nav, text="— / —", font=("Consolas", 11, "bold"),
            bg=COLORS["bg"], fg=COLORS["text"], padx=16)
        self.nav_label.pack(side="left")
        self._btn("Next ▶", self._next_cycle, small=True).pack(side="left")
        self.cycle_badge = tk.Label(
            nav, text="", font=("Consolas", 10, "bold"),
            bg=COLORS["bg"], fg=COLORS["highlight"], padx=8)
        self.cycle_badge.pack(side="left")

        # Oscilloscope + FFT
        canvases = tk.Frame(self.right, bg=COLORS["bg"])
        canvases.pack(fill="both", expand=True)
        canvases.columnconfigure(0, weight=3)
        canvases.columnconfigure(1, weight=2)
        canvases.rowconfigure(0, weight=1)

        wf = tk.Frame(canvases, bg=COLORS["panel"])
        wf.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(wf, text="OSCILLOSCOPE", font=("Consolas", 8),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(
                     anchor="w", padx=8, pady=(6, 0))
        self.wave_canvas = tk.Canvas(wf, bg=COLORS["panel"], highlightthickness=0)
        self.wave_canvas.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.wave_canvas.bind("<Configure>", lambda e: self._draw_wave())

        ff = tk.Frame(canvases, bg=COLORS["panel"])
        ff.grid(row=0, column=1, sticky="nsew")
        tk.Label(ff, text="FFT SPECTRUM", font=("Consolas", 8),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(
                     anchor="w", padx=8, pady=(6, 0))
        self.fft_canvas = tk.Canvas(ff, bg=COLORS["panel"], highlightthickness=0)
        self.fft_canvas.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.fft_canvas.bind("<Configure>", lambda e: self._draw_fft())

        # Thumbnail strip
        tk.Label(self.right, text="ALL CYCLES", font=("Consolas", 8),
                 bg=COLORS["bg"], fg=COLORS["muted"]).pack(anchor="w", pady=(6, 2))
        thumb_outer = tk.Frame(self.right, bg=COLORS["bg"], height=72)
        thumb_outer.pack(fill="x")
        thumb_outer.pack_propagate(False)
        self.thumb_scroll = tk.Canvas(
            thumb_outer, bg=COLORS["bg"], highlightthickness=0, height=72)
        sb = ttk.Scrollbar(thumb_outer, orient="horizontal",
                           command=self.thumb_scroll.xview)
        self.thumb_scroll.configure(xscrollcommand=sb.set)
        sb.pack(side="bottom", fill="x")
        self.thumb_scroll.pack(fill="both", expand=True)
        self.thumb_frame = tk.Frame(self.thumb_scroll, bg=COLORS["bg"])
        self.thumb_scroll.create_window((0, 0), window=self.thumb_frame, anchor="nw")
        self.thumb_frame.bind(
            "<Configure>",
            lambda e: self.thumb_scroll.configure(
                scrollregion=self.thumb_scroll.bbox("all")))

    # -----------------------------------------------------------------------
    #  UI helpers
    # -----------------------------------------------------------------------
    def _btn(self, text, cmd, small=False):
        return tk.Button(
            self, text=text, command=cmd,
            font=("Consolas", 9 if small else 10),
            bg=COLORS["accent"], fg=COLORS["text"],
            activebackground=COLORS["highlight"],
            activeforeground="#ffffff",
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2")

    def _section(self, text):
        tk.Label(self.left, text=text, font=("Consolas", 9, "bold"),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(
                     anchor="w", padx=12, pady=(10, 2))

    def _sep(self):
        tk.Frame(self.left, bg=COLORS["grid"], height=1).pack(
            fill="x", padx=12, pady=4)

    def _clm_text(self) -> str:
        if self.export_clm.get():
            return (f"'clm' chunk written\n"
                    f"cycle = {self.export_size.get()} samples\n"
                    f"Deluge · Serum · Vital compatible")
        return "Plain WAV — no wavetable header.\nSynths use their own fallback size."

    # -----------------------------------------------------------------------
    #  Event handlers
    # -----------------------------------------------------------------------
    def _on_mode_change(self):
        mode = self.mode.get()
        if mode == "multi":
            self._open_multi()
        elif mode == "batch":
            self._open_batch()
        else:
            self._open_file()
            #self.multi_nav_frame.pack_forget()

    def _on_cycle_size_change(self):
        if self.audio is not None:
            self._slice_cycles()
            self._refresh_display()

    def _on_clm_toggle(self):
        self.clm_desc.config(text=self._clm_text())

    def _inc_n(self):
        v = self.export_n_cycles.get()
        if self.cycles:
            self.export_n_cycles.set(min(v + 1, len(self.cycles)))

    def _dec_n(self):
        self.export_n_cycles.set(max(0, self.export_n_cycles.get() - 1))

    # -----------------------------------------------------------------------
    #  File loading helpers
    # -----------------------------------------------------------------------
    def _load_file(self, path: str):
        """Load a single WAV into the current working state."""
        audio, sr, chunk_info = read_wav(path)
        self.audio      = audio
        self.sr         = sr
        self.chunk_info = chunk_info
        self.filepath   = path
        self.file_label.config(text=os.path.basename(path))
        self._update_meta_label()
        self._auto_detect(path, chunk_info)
        self._slice_cycles()
        self._refresh_display()
        self.status_var.set(
            f"Loaded: {os.path.basename(path)}  |  "
            f"{len(audio)} samples @ {sr} Hz")

    def _update_meta_label(self):
        """Show detected wavetable chunk info in the left panel."""
        ci = self.chunk_info
        parts = []
        if ci.get("clm "):
            cs = parse_clm_chunk(ci["clm "])
            parts.append(f"clm  → {cs} samples/cycle  (Serum)")
        if ci.get("srge"):
            cs = parse_srge_chunk(ci["srge"])
            parts.append(f"srge → {cs} samples/cycle  (Surge XT)")
        if ci.get("uhWT"):
            parts.append("uhWT → present  (u-he, size unknown)")
        self.meta_label.config(
            text="\n".join(parts) if parts else "No wavetable chunk found")

    # -----------------------------------------------------------------------
    #  File opening modes
    # -----------------------------------------------------------------------
    def _open_file(self):
        mode = self.mode.get()
        if mode == "multi":
            self._open_multi()
        elif mode == "batch":
            self._open_batch()
        else:
            path = filedialog.askopenfilename(
                title="Open a wavetable WAV file",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
            if not path:
                return
            try:
                self._load_file(path)
                self.multi_nav_frame.pack_forget()
            except Exception as e:
                messagebox.showerror("Error", f"Could not read file:\n{e}")

    def _open_multi(self):
        """Let the user pick several WAV files and navigate between them."""
        paths = filedialog.askopenfilenames(
            title="Select wavetable WAV files",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not paths:
            self.mode.set("single")
            return
        # Load all files
        loaded = []
        errors = []
        for p in paths:
            try:
                audio, sr, ci = read_wav(p)
                loaded.append((p, audio, sr, ci))
            except Exception as e:
                errors.append(f"{os.path.basename(p)}: {e}")
        if errors:
            messagebox.showwarning(
                "Some files could not be loaded",
                "\n".join(errors))
        if not loaded:
            self.mode.set("single")
            return
        self.file_list     = loaded
        self.file_list_idx = 0
        self._switch_file(0)
        # Show multi-file navigator
        self.multi_nav_frame.pack(fill="x", padx=12, pady=(4, 0))

    def _switch_file(self, idx: int):
        """Activate a file from file_list by index."""
        self.file_list_idx = idx
        path, audio, sr, ci = self.file_list[idx]
        self.audio      = audio
        self.sr         = sr
        self.chunk_info = ci
        self.filepath   = path
        self.file_label.config(text=os.path.basename(path))
        self.multi_nav_label.config(
            text=f"{idx + 1}/{len(self.file_list)}")
        self._update_meta_label()
        self._auto_detect(path, ci)
        self._slice_cycles()
        self._refresh_display()
        self.status_var.set(
            f"[{idx + 1}/{len(self.file_list)}] "
            f"{os.path.basename(path)}  |  "
            f"{len(audio)} samples @ {sr} Hz")

    def _prev_file(self):
        if self.file_list:
            self._switch_file(
                (self.file_list_idx - 1) % len(self.file_list))

    def _next_file(self):
        if self.file_list:
            self._switch_file(
                (self.file_list_idx + 1) % len(self.file_list))

    def _open_batch(self):
        """Load all WAVs from a folder, take one cycle each, build a wavetable."""
        folder = filedialog.askdirectory(
            title="Select folder containing waveform WAVs")
        if not folder:
            self.mode.set("single")
            return
        wavs = sorted([f for f in os.listdir(folder)
                       if f.lower().endswith(".wav")])
        if not wavs:
            messagebox.showwarning("Batch", "No WAV files found in this folder.")
            self.mode.set("single")
            return
        target       = self.export_size.get()
        cycles_audio = []
        errors       = []
        for fname in wavs:
            try:
                audio, _, _ = read_wav(os.path.join(folder, fname))
                src = audio[:target] if len(audio) >= target else audio
                cycles_audio.append(resample_cycle(src, target))
            except Exception as e:
                errors.append(f"{fname}: {e}")
        if errors:
            messagebox.showwarning(
                "Some files skipped", "\n".join(errors))
        if not cycles_audio:
            messagebox.showerror("Batch", "No valid WAV files could be read.")
            self.mode.set("single")
            return
        self.audio      = np.concatenate(cycles_audio)
        self.sr         = 44100
        self.chunk_info = {}
        self.filepath   = os.path.join(folder, "batch_result.wav")
        self.cycle_size.set(target)
        self.file_label.config(text=f"Batch: {len(wavs)} files")
        self.meta_label.config(text="Assembled from batch — no source chunk")
        self.export_n_cycles.set(0)
        self.multi_nav_frame.pack_forget()
        self._slice_cycles()
        self._refresh_display()
        self.status_var.set(
            f"Batch: {len(wavs)} files  →  "
            f"{len(cycles_audio)} cycles of {target} samples")
        self.mode.set("single")

    # -----------------------------------------------------------------------
    #  Analysis
    # -----------------------------------------------------------------------
    def _auto_detect(self, path: str, chunk_info: dict):
        """
        Determine analysis cycle size:
        1. Use any known wavetable chunk (clm / srge / uhWT).
        2. Fall back to correlation-based detection.
        Pre-fill export size to match.
        """
        cs, source = detect_wt_chunk_cycle_size(chunk_info)
        if cs and cs in CYCLE_SIZES:
            self.cycle_size.set(cs)
            self.export_size.set(cs)
            self.detect_label.config(
                text=f"From '{source}' chunk: {cs} samples")
            self.clm_desc.config(text=self._clm_text())
            return

        best, scores = detect_cycle_size(self.audio)
        self.cycle_size.set(best)
        lines = []
        for sz in CYCLE_SIZES:
            s = scores.get(sz, 0.0)
            n = len(self.audio) // sz
            lines.append(
                f"{sz}: {n} cycles  conf={s:.2f}{'  ◀' if sz == best else ''}")
        self.detect_label.config(text=f"Auto-detected: {best} samples")
        tip = "\n".join(lines)
        self.detect_label.bind(
            "<Enter>", lambda e, t=tip: self.status_var.set(t))
        self.detect_label.bind(
            "<Leave>", lambda e: self.status_var.set(""))

        if source:  # uhWT present but size unknown
            self.detect_label.config(
                text=f"Auto-detected: {best}  ({source})")

    def _slice_cycles(self):
        if self.audio is None:
            return
        cs = self.cycle_size.get()
        n  = len(self.audio) // cs
        self.cycles      = [self.audio[i * cs:(i + 1) * cs] for i in range(n)]
        self.current_idx = 0
        self.export_n_cycles.set(0)

    # -----------------------------------------------------------------------
    #  Display
    # -----------------------------------------------------------------------
    def _refresh_display(self):
        if not self.cycles:
            return
        self.nav_label.config(
            text=f"Cycle  {self.current_idx + 1}  /  {len(self.cycles)}")
        label, _ = classify_cycle(self.cycles[self.current_idx])
        self.cycle_badge.config(
            text=label.upper(), fg=LABEL_COLORS.get(label, COLORS["muted"]))
        cs = self.cycle_size.get()
        self.info_label.config(text=(
            f"Total samples : {len(self.audio)}\n"
            f"Cycles        : {len(self.cycles)}\n"
            f"Cycle size    : {cs} samples\n"
            f"Sample rate   : {self.sr} Hz\n"
            f"Cycle duration: {cs / self.sr * 1000:.1f} ms"
        ))
        self.clm_desc.config(text=self._clm_text())
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
        for yf in [0.25, 0.5, 0.75]:
            c.create_line(0, int(h * yf), w, int(h * yf),
                          fill=COLORS["grid"])
        c.create_line(0, h // 2, w, h // 2,
                      fill=COLORS["muted"], dash=(4, 4))
        samples = self.cycles[self.current_idx]
        pad = 10
        pts = []
        for i, s in enumerate(samples):
            pts.extend([pad + i / max(len(samples) - 1, 1) * (w - 2 * pad),
                        h // 2 - s * (h // 2 - pad)])
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
        n      = min(len(fft), 12)
        pad    = 12
        slot_w = (w - 2 * pad) / max(n, 1)
        bar_w  = max(4, int(slot_w * 0.7))
        labels = ["F","2","3","4","5","6","7","8","9","10","11","12"]
        for i in range(n):
            bh = int(float(fft[i]) * (h - 32))
            x  = int(pad + i * slot_w + (slot_w - bar_w) / 2)
            c.create_rectangle(x, h - 20 - bh, x + bar_w, h - 20,
                               fill=COLORS["highlight"] if i == 0
                               else COLORS["fft"], outline="")
            c.create_text(x + bar_w // 2, h - 8, text=labels[i],
                          font=("Consolas", 8), fill=COLORS["muted"])

    def _build_thumbs(self):
        for child in self.thumb_frame.winfo_children():
            child.destroy()
        for i, cycle in enumerate(self.cycles):
            label, _ = classify_cycle(cycle)
            border    = COLORS["highlight"] if i == self.current_idx \
                        else COLORS["panel"]
            frm = tk.Frame(self.thumb_frame, bg=border, padx=1, pady=1)
            frm.pack(side="left", padx=2)
            th = tk.Canvas(frm, width=48, height=44,
                           bg=COLORS["panel"], highlightthickness=0,
                           cursor="hand2")
            th.pack()
            color = LABEL_COLORS.get(label, COLORS["muted"])
            pts   = []
            for j, s in enumerate(cycle):
                pts.extend([(j / max(len(cycle) - 1, 1)) * 48,
                            22 - float(s) * 18])
            if len(pts) >= 4:
                th.create_line(*pts, fill=color, width=1)
            idx = i
            th.bind("<Button-1>", lambda e, idx=idx: self._goto_cycle(idx))
            tk.Label(frm, text=label[:3].upper(),
                     font=("Consolas", 7), bg=border, fg=color).pack()

    # -----------------------------------------------------------------------
    #  Navigation
    # -----------------------------------------------------------------------
    def _prev_cycle(self):
        if self.cycles:
            self.current_idx = (self.current_idx - 1) % len(self.cycles)
            self._refresh_display()

    def _next_cycle(self):
        if self.cycles:
            self.current_idx = (self.current_idx + 1) % len(self.cycles)
            self._refresh_display()

    def _goto_cycle(self, idx: int):
        self.current_idx = idx
        self._refresh_display()

    # -----------------------------------------------------------------------
    #  Export
    # -----------------------------------------------------------------------
    def _get_export_cycles(self):
        if not self.cycles:
            messagebox.showwarning("Export", "No cycles loaded.")
            return None
        target = self.export_size.get()
        n      = self.export_n_cycles.get()
        src    = self.cycles[:n] if 0 < n <= len(self.cycles) else self.cycles
        return [resample_cycle(c, target) for c in src]

    def _write(self, path: str, audio: np.ndarray, cycle_size: int):
        if self.export_clm.get():
            write_wav_with_clm(path, audio, self.sr, cycle_size)
        else:
            write_wav_plain(path, audio, self.sr)

    def _export_separate(self):
        cycles = self._get_export_cycles()
        if not cycles:
            return
        folder = filedialog.askdirectory(title="Choose export folder")
        if not folder:
            return
        base   = os.path.splitext(os.path.basename(self.filepath or "wt"))[0]
        target = self.export_size.get()
        suffix = "_clm" if self.export_clm.get() else ""
        for i, c in enumerate(cycles):
            label, _ = classify_cycle(self.cycles[i])
            fname    = f"{base}_cycle{i + 1:02d}_{label}_{target}{suffix}.wav"
            self._write(os.path.join(folder, fname), c, target)
        msg = (f"{len(cycles)} files exported to:\n{folder}\n"
               f"CLM: {'yes — Deluge/Serum/Vital' if self.export_clm.get() else 'no'}")
        self.status_var.set(msg.replace("\n", "  |  "))
        messagebox.showinfo("Export complete", msg)

    def _export_unified(self):
        cycles = self._get_export_cycles()
        if not cycles:
            return
        target  = self.export_size.get()
        base    = os.path.splitext(os.path.basename(self.filepath or "wt"))[0]
        suffix  = "_clm" if self.export_clm.get() else ""
        default = f"{base}_{len(cycles)}cycles_{target}{suffix}.wav"
        path    = filedialog.asksaveasfilename(
            title="Save unified wavetable WAV",
            initialfile=default, defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        self._write(path, np.concatenate(cycles), target)
        clm_note = "  |  CLM written" if self.export_clm.get() else ""
        msg = (f"{os.path.basename(path)}\n"
               f"{len(cycles)} cycles × {target} samples{clm_note}")
        self.status_var.set(msg.replace("\n", "  |  "))
        messagebox.showinfo("Export complete", msg)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = WavetableTool()
    app.mainloop()
