# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from pathlib import Path
import mimetypes
import os
import base64
import uuid
import subprocess
import shutil
from IPython.display import display, Image, HTML


def display_audio_at_data(path: str, start_s: float = 0.0, autoplay: bool = False, label: str | None = None):
    p = Path(path).resolve()
    mime, _ = mimetypes.guess_type(p.name)
    if mime is None:
        # typy częste:
        if p.suffix.lower() == ".mp3": mime = "audio/mpeg"
        elif p.suffix.lower() in (".m4a", ".mp4"): mime = "audio/mp4"
        elif p.suffix.lower() == ".wav": mime = "audio/wav"
        else: mime = "audio/mpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    src = f"data:{mime};base64,{b64}"

    el_id = f"a_{uuid.uuid4().hex[:8]}"
    label_html = f"<div style='font:12px monospace;margin:4px 0'>{label or ''}</div>"
    autoplay_attr = "autoplay" if autoplay else ""
    html = f"""
    {label_html}
    <audio id="{el_id}" controls preload="auto" src="{src}" {autoplay_attr} style="width:100%"></audio>
    <script>
      (function(){{
        const a = document.getElementById("{el_id}");
        if(!a) return;
        const setPos = () => {{
          try {{ a.currentTime = {float(start_s):.3f}; }} catch(e) {{}}
        }};
        a.addEventListener('loadedmetadata', setPos, {{once:true}});
      }})();
    </script>
    """
    display(HTML(html))


def _is_image_ext(path: str) -> bool:
    ext = os.path.splitext(path.lower())[1]
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".heic", ".heif"}


def display_image(path: str):
    if path and os.path.exists(path):
        try:
            display(Image(filename=path))
            return
        except Exception as e:
            print(f"[WARN] Nie udało się wyświetlić klatki: {e}")
    
    elif _is_image_ext(path) and os.path.exists(path):
        try:
            display(Image(filename=path))
        except Exception as e:
            print(f"[WARN] Cannot display image: {e}")



_SERVE_DIR = Path("_served"); _SERVE_DIR.mkdir(exist_ok=True)

def _serve_path(p: str | Path) -> str:
    """Zwraca URL do lokalnego pliku tak, by <video> mógł go odczytać w notatniku."""
    p = Path(p).resolve()
    try:
        rel = p.relative_to(Path.cwd().resolve())
        dst = p
    except ValueError:
        dst = _SERVE_DIR / f"{uuid.uuid4().hex}_{p.name}"
        shutil.copy2(p, dst)
        rel = dst.relative_to(Path.cwd())
    # VS Code/Jupyter serwują katalog roboczy jako /files/
    return f"files/{rel.as_posix()}"

def display_video_jump(*, path: str | Path, start_s: float,
                       poster: str | None = None,
                       autoplay: bool = False,
                       muted: bool = False,
                       label: str | None = None,
                       max_height_px: int = 480,
                       playback_rate: float = 1.0) -> None:
    """
    Ładuje CAŁY plik wideo i ustawia kursor na start_s (bez wycinania klipu).
    """
    src_url = _serve_path(path)
    poster_attr = f'poster="{_serve_path(poster)}"' if poster else ""
    autoplay_attr = "autoplay muted" if autoplay else ("muted" if muted else "")
    vid_id = f"vid_{uuid.uuid4().hex}"
    lbl = f"<div style='font:12px monospace;margin:6px 0'>{label or ''}</div>" if label else ""

    # Po loadedmetadata ustawiamy currentTime i (opcjonalnie) autoodtwarzamy.
    # Powtórne ustawienie currentTime, jeśli metadata nie weszły jeszcze.
    js = f"""
    <script>
    (function() {{
      const v = document.getElementById('{vid_id}');
      if (!v) return;
      const desired = {float(start_s):.3f};
      const rate = {float(playback_rate):.3f};
      function seekAndMaybePlay(){{
        try {{
          v.playbackRate = rate;
          v.currentTime = desired;
          {"v.play().catch(()=>{});" if autoplay else ""}
        }} catch (e) {{ /* ignore */ }}
      }}
      if (v.readyState >= 1) {{
        seekAndMaybePlay();
      }} else {{
        v.addEventListener('loadedmetadata', seekAndMaybePlay, {{ once: true }});
      }}
    }})();
    </script>
    """

    html = f"""
    {lbl}
    <video id="{vid_id}" controls {autoplay_attr} {poster_attr} preload="metadata"
           style="width:100%;max-height:{max_height_px}px;">
      <source src="{src_url}" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    {js}
    """
    display(HTML(html))
