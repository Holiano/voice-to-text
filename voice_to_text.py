"""
Voice-to-Text Clipboard Tool
==============================
Press Shift+Alt+D to START recording.
Press Shift+Alt+D again to STOP recording and transcribe.
The transcribed text is automatically copied to your clipboard.

A small dot follows your cursor: red while recording, green when done and ready to paste.

First run: Whisper will download the "base" model (~150 MB). Subsequent runs are instant.
"""

import threading
import datetime
import os
import time
import tkinter as tk

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import pyperclip
import keyboard
from PIL import Image, ImageDraw
from pystray import Icon, Menu, MenuItem


# ── Configuration ────────────────────────────────────────────────────────────
HOTKEY        = "shift+alt+d"
SAMPLE_RATE   = 16000
CHANNELS      = 1
WHISPER_MODEL = "tiny"         # tiny | base | small | medium | large | turbo
DOT_SIZE      = 14             # px diameter
DOT_OFFSET    = 18             # px offset from cursor tip
# ─────────────────────────────────────────────────────────────────────────────

# State
recording     = False
audio_chunks  = []
model         = None
model_loaded  = False
tray_icon_ref = None
lock          = threading.Lock()

# Overlay state
_dot_root     = None           # tkinter root (lives on overlay thread)
_dot_canvas   = None
_dot_visible  = False
_dot_color    = "red"
_dot_lock     = threading.Lock()


# ── Cursor dot overlay (tkinter) ─────────────────────────────────────────────

def _overlay_thread_func():
    """Runs the tkinter event loop for the cursor dot on its own thread."""
    global _dot_root, _dot_canvas

    root = tk.Tk()
    root.overrideredirect(True)           # no title bar / borders
    root.attributes("-topmost", True)
    root.attributes("-transparentcolor", "white")
    root.configure(bg="white")
    root.geometry(f"{DOT_SIZE}x{DOT_SIZE}+0+0")
    root.withdraw()                       # start hidden

    canvas = tk.Canvas(
        root,
        width=DOT_SIZE, height=DOT_SIZE,
        bg="white", highlightthickness=0,
    )
    canvas.pack()

    _dot_root   = root
    _dot_canvas = canvas

    def _tick():
        with _dot_lock:
            visible = _dot_visible
            color   = _dot_color

        if visible and _dot_root:
            try:
                x = root.winfo_pointerx() + DOT_OFFSET
                y = root.winfo_pointery() + DOT_OFFSET
                root.geometry(f"{DOT_SIZE}x{DOT_SIZE}+{x}+{y}")
                root.deiconify()
                canvas.delete("all")
                canvas.create_oval(1, 1, DOT_SIZE - 1, DOT_SIZE - 1, fill=color, outline="")
            except Exception:
                pass
        elif _dot_root:
            try:
                root.withdraw()
            except Exception:
                pass

        root.after(16, _tick)   # ~60 fps

    root.after(16, _tick)
    root.mainloop()


def _show_dot(color: str):
    global _dot_visible, _dot_color
    with _dot_lock:
        _dot_color   = color
        _dot_visible = True


def _hide_dot():
    global _dot_visible
    with _dot_lock:
        _dot_visible = False


# ── Tray icon image helpers ──────────────────────────────────────────────────

def make_icon(state: str) -> Image.Image:
    size = 64
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    colors = {
        "idle":       "#4CAF50",
        "recording":  "#F44336",
        "loading":    "#FF9800",
        "processing": "#2196F3",
    }
    color = colors.get(state, "#9E9E9E")
    draw.ellipse([2, 2, size - 2, size - 2], fill=color, outline="white", width=3)
    cx, cy = size // 2, size // 2
    if state == "recording":
        draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill="white")
    elif state == "processing":
        draw.rectangle([cx - 8, cy - 8, cx + 8, cy + 8], fill="white")
    else:
        draw.rounded_rectangle([cx - 7, cy - 14, cx + 7, cy + 4], radius=6, fill="white")
        draw.arc([cx - 12, cy - 6, cx + 12, cy + 16], start=0, end=180, fill="white", width=3)
        draw.line([cx, cy + 16, cx, cy + 22], fill="white", width=3)
        draw.line([cx - 6, cy + 22, cx + 6, cy + 22], fill="white", width=3)
    return img


def set_icon_state(state: str):
    if tray_icon_ref:
        try:
            tray_icon_ref.icon = make_icon(state)
        except Exception:
            pass


def update_tray_title(text: str):
    if tray_icon_ref:
        try:
            tray_icon_ref.title = text
        except Exception:
            pass


# ── Audio recording ──────────────────────────────────────────────────────────

def audio_callback(indata, _frames, _time_info, _status):
    if recording:
        audio_chunks.append(indata.copy())


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe_and_copy(chunks: list):
    if not chunks:
        _hide_dot()
        set_icon_state("idle")
        update_tray_title(f"Voice-to-Text  |  {HOTKEY.upper()} to record")
        print(f"[{_ts()}] No audio recorded.")
        return

    set_icon_state("processing")
    update_tray_title("Voice-to-Text  |  Transcribing...")

    try:
        audio    = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
        segments, _ = model.transcribe(audio, beam_size=1, language=None)
        text     = " ".join(s.text for s in segments).strip()

        if text:
            pyperclip.copy(text)
            _show_dot("green")
            print(f"[{_ts()}] Transcribed: {text}")
            threading.Timer(3.0, _hide_dot).start()
        else:
            _hide_dot()
            print(f"[{_ts()}] No speech detected.")

    except Exception as exc:
        _hide_dot()
        print(f"[{_ts()}] Transcription error: {exc}")
    finally:
        set_icon_state("idle")
        update_tray_title(f"Voice-to-Text  |  {HOTKEY.upper()} to record")


# ── Hotkey handler ────────────────────────────────────────────────────────────

stream = None


def toggle_recording():
    global recording, audio_chunks

    with lock:
        if not model_loaded:
            print(f"[{_ts()}] Model still loading, please wait...")
            return

        if not recording:
            audio_chunks = []
            recording    = True
            set_icon_state("recording")
            update_tray_title("Voice-to-Text  |  Recording...")
            _show_dot("red")
            print(f"[{_ts()}] Recording started")
        else:
            recording = False
            captured  = list(audio_chunks)
            audio_chunks = []
            print(f"[{_ts()}] Recording stopped - {len(captured)} chunks")
            _show_dot("dodger blue")   # instant feedback: processing
            threading.Thread(target=transcribe_and_copy, args=(captured,), daemon=True).start()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    global model, model_loaded
    print(f"[{_ts()}] Loading Whisper '{WHISPER_MODEL}' model...")
    update_tray_title("Voice-to-Text  |  Loading model...")
    model        = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    model_loaded = True
    set_icon_state("idle")
    update_tray_title(f"Voice-to-Text  |  {HOTKEY.upper()} to record")
    print(f"[{_ts()}] Model loaded OK")


# ── Tray menu ─────────────────────────────────────────────────────────────────

def on_quit(icon, _item):
    keyboard.unhook_all()
    _hide_dot()
    icon.stop()
    os._exit(0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global tray_icon_ref, stream

    print("=" * 55)
    print(f"  Voice-to-Text  |  Press {HOTKEY.upper()} to start/stop")
    print("=" * 55)

    # Start cursor dot overlay on its own thread
    t = threading.Thread(target=_overlay_thread_func, daemon=True)
    t.start()
    time.sleep(0.3)   # let tkinter initialise

    # System tray
    menu = Menu(
        MenuItem("Voice-to-Text", None, enabled=False),
        Menu.SEPARATOR,
        MenuItem(f"Hotkey: {HOTKEY.upper()}", None, enabled=False),
        Menu.SEPARATOR,
        MenuItem("Quit", on_quit),
    )
    icon = Icon(
        "voice_to_text",
        make_icon("loading"),
        title="Voice-to-Text  |  Loading...",
        menu=menu,
    )
    tray_icon_ref = icon

    threading.Thread(target=load_model, daemon=True).start()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
    )
    stream.start()

    keyboard.add_hotkey(HOTKEY, toggle_recording, suppress=False)
    print(f"[{_ts()}] Hotkey '{HOTKEY}' registered")

    icon.run()


if __name__ == "__main__":
    main()
