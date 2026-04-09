# Voice-to-Text

A lightweight Windows tool that transcribes your voice and copies the result straight to your clipboard.

## How it works

1. Press **Shift+Alt+D** to start recording — a red dot appears near your cursor
2. Press **Shift+Alt+D** again to stop — a blue dot shows while transcribing
3. A green dot appears when the text is ready — just paste with **Ctrl+V**

Runs in the system tray. No windows, no popups.

## Setup

```bash
pip install faster-whisper sounddevice numpy pyperclip keyboard pillow pystray
python voice_to_text.py
```

The first run downloads the Whisper `tiny` model (~40 MB). Subsequent runs load instantly.

> **Note:** The `keyboard` library requires running as administrator on Windows for global hotkeys to work.

## Configuration

At the top of `voice_to_text.py`:

| Setting | Default | Description |
|---|---|---|
| `HOTKEY` | `shift+alt+d` | Keyboard shortcut |
| `WHISPER_MODEL` | `tiny` | Model size: `tiny` `base` `small` `medium` `large` |

Larger models are more accurate but slower on CPU.
