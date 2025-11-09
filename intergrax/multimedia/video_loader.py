# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from pathlib import Path
from yt_dlp import YoutubeDL
from typing import Optional
import whisper
import webvtt
from tqdm.auto import tqdm
import cv2
import os
import json


def yt_download_video(youtube_url: str, out_dir:str | Path)->Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outtmpl = str(out_dir/"vid_%(id)s.%(ext)s")
    ydl_opts = {
        # "format": "bv*+ba/b[ext=mp4]/b",
        "format": "bestvideo+bestaudio/best",
        "outtmpl": outtmpl,
        "merge_output_format":"mp4",
        "noplaylist":True,
        "quiet": True,
        "no_warnings": True        
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)

    video_id = info.get("id")
    filepath = out_dir / f"vid_{video_id}.mp4"

    if not filepath.exists():
        ext = info.get("ext", "mp4")
        alt = out_dir / f"vid_{video_id}.{ext}"
        if alt.exists():
            filepath = alt
        else:
            raise FileNotFoundError("Cannot find downloaded video file.")
        
    return filepath



def transcribe_to_vtt(
        input_media_path: str | Path,
        output_vtt_path: Optional[str|Path] = None,
        model_size:str="base",
        language: Optional[str] =None
    )->Path:

    input_media_path = Path(input_media_path)
    output_vtt_path = Path(output_vtt_path) if output_vtt_path else input_media_path.with_suffix('.vtt')
    output_vtt_path.parent.mkdir(parents=True, exist_ok=True)

    def _sec_to_vtt_ts(s: float) -> str:
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:06.3f}"

    if not output_vtt_path.exists():
        model = whisper.load_model(model_size)
        result = model.transcribe(str(input_media_path), language=language)

        vtt = webvtt.WebVTT()
        for seq in tqdm(result.get("segments", []), desc="Transcripting", unit="frame"):
            start = _sec_to_vtt_ts(seq['start'])
            end = _sec_to_vtt_ts(seq['end'])
            text = " ".join(seq.get('text', "").split())
            vtt.captions.append(webvtt.Caption(start,end, text))
        vtt.save(str(output_vtt_path))
    return output_vtt_path


def extract_frames_and_metadata(
    path_to_video: str,
    path_to_transcript: str,
    path_to_save_extracted_frames :str,
    path_to_save_metadatas :str,
    save_metadata:bool=True
):

    def time_str_to_ms(time_str: str) -> int:
        time_str = time_str.strip().replace(',', '.')
        if time_str.count(':') == 3:  
            # HH:MM:SS:MMM → HH:MM:SS.MMM
            parts = time_str.split(':')
            time_str = ':'.join(parts[:-1]) + '.' + parts[-1]

        try:
            h, m, s = time_str.split(':')
            s, ms = (s.split('.') + ['0'])[:2]
            total_ms = (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms.ljust(3, '0')[:3])
            return total_ms
        except Exception:
            raise ValueError(f"Nieprawidłowy format czasu: {time_str}")
        
    def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is not None and height is not None:
            r = width / float(w)
            dim = (int(width), int(round(h * r)))
        elif width is None:
            r = height / float(h)
            dim = (int(round(w * r)), int(height))
        else:
            r = width / float(w)
            dim = (int(width), int(round(h * r)))

        return cv2.resize(image, dim, interpolation=inter)

    metadatas = []        

    video = cv2.VideoCapture(path_to_video)
    trans = webvtt.read(path_to_transcript)
    
    for idx, transcript in enumerate(tqdm(trans, desc="Extracting frames", unit="frame", total=len(trans))):
        start_ms = time_str_to_ms(transcript.start)
        end_ms = time_str_to_ms(transcript.end)
        mid_time_ms = (end_ms+start_ms)/2
        
        text = transcript.text.replace("\n"," ")

        video.set(cv2.CAP_PROP_POS_MSEC, mid_time_ms)

        success, frame = video.read()
        if success:
            image = maintain_aspect_ratio_resize(frame, height=350)

            img_fname = f"frame_{idx}.jpg"
            img_fpath = os.path.join(
                path_to_save_extracted_frames, 
                img_fname
            )
            cv2.imwrite(img_fpath, image)

            metadata = {
                "extracted_frame_path": img_fpath,
                "transcript": text,
                "video_segment_id": idx,
                "video_path": path_to_video,
                "mid_time_ms" : mid_time_ms,
                "start": transcript.start,       # original timestamp string (e.g. "00:00:04.500")
                "end": transcript.end,           # original timestamp string
                "start_ms": start_ms,            # numeric start time in milliseconds
                "end_ms": end_ms,                # numeric end time in milliseconds
                "duration_ms": end_ms - start_ms # useful for analytics/timeline
            }
            metadatas.append(metadata)
        else:
            print(f"ERROR! Cannot extract frame: idx = {idx}")

    if save_metadata:
        fn = os.path.join(path_to_save_metadatas, "metadatas.json")
        with open(fn, "w") as outfile:
            json.dump(metadatas, outfile)

    return metadatas



def extract_frames_from_video(
    path_to_video: str | Path,
    path_to_save_extracted_frames: str | Path,
    every_seconds: float = 1.0,
    target_height: int = 350,
    limit: int | None = None,
):
    
    path_to_video = str(path_to_video)
    out_dir = Path(path_to_save_extracted_frames)
    out_dir.mkdir(parents=True, exist_ok=True)

    def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is not None and height is not None:
            r = width / float(w)
            dim = (int(width), int(round(h * r)))
        elif width is None:
            r = height / float(h)
            dim = (int(round(w * r)), int(height))
        else:
            r = width / float(w)
            dim = (int(width), int(round(h * r)))
        return cv2.resize(image, dim, interpolation=inter)

    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć wideo: {path_to_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        fps = 25.0
    step = max(int(round(fps * every_seconds)), 1)

    saved_paths: list[str] = []
    frame_idx = 0
    saved = 0

    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                image = maintain_aspect_ratio_resize(frame, height=target_height)
                img_fname = f"frame_{len(saved_paths):06d}.jpg"
                img_fpath = out_dir / img_fname
                cv2.imwrite(str(img_fpath), image)
                saved_paths.append(str(img_fpath))
                saved += 1
                if limit is not None and saved >= limit:
                    pbar.update(total_frames - frame_idx - 1)
                    break

            frame_idx += 1
            pbar.update(1)

    cap.release()
    return saved_paths
