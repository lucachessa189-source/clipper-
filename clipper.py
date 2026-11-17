"""
clipper.py - turn a long video into short vertical clips with burned-in subtitles.

Pipeline: download (optional) -> transcribe (faster-whisper) -> Claude picks the best
moments -> ffmpeg cuts, crops to 9:16 and burns in captions. No watermark is added.

Setup:
    pip install faster-whisper anthropic yt-dlp
    Install ffmpeg and make sure it's on PATH
    set ANTHROPIC_API_KEY=your_key        (Windows)  /  export ANTHROPIC_API_KEY=...

Usage:
    python clipper.py "https://youtube.com/watch?v=XXXX" --clips 5
    python clipper.py my_video.mp4 --clips 8 --min 20 --max 50 --no-crop

Only use videos you own or have permission to clip.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import anthropic
from faster_whisper import WhisperModel

MODEL = os.environ.get("CLIPPER_MODEL", "claude-sonnet-5-5")


def download(url: str, workdir: Path) -> Path:
    out = workdir / "source.%(ext)s"
    subprocess.run(
        ["yt-dlp", "-f", "bv*[height<=1080]+ba/b", "--merge-output-format", "mp4",
         "-o", str(out), url],
        check=True,
    )
    return workdir / "source.mp4"


def transcribe(video: Path, size: str):
    print(f"Transcribing with Whisper ({size})...")
    model = WhisperModel(size, device="auto", compute_type="auto")
    segments, _ = model.transcribe(str(video), word_timestamps=True, vad_filter=True)
    segs, words = [], []
    for s in segments:
        segs.append({"start": s.start, "end": s.end, "text": s.text.strip()})
        for w in s.words or []:
            words.append({"start": w.start, "end": w.end, "word": w.word.strip()})
    return segs, words


def pick_clips(segs, n, min_len, max_len):
    transcript = "\n".join(f"[{s['start']:.1f}] {s['text']}" for s in segs)
    prompt = f"""You are an editor who cuts viral short-form clips from long videos.
Below is a transcript; each line starts with its start time in seconds.

Pick the {n} best clips. Rules:
- Each clip is {min_len}-{max_len} seconds long.
- It must stand alone: a strong hook in the first sentence, a complete thought, a payoff.
- Start at the beginning of a sentence and end at the end of one.
- Favor surprising, emotional, funny, controversial or highly useful moments.
- Clips must not overlap.

Return ONLY a JSON array, no other text:
[{{"start": <seconds>, "end": <seconds>, "title": "<short title>", "reason": "<one line>"}}]

TRANSCRIPT:
{transcript}"""
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL, max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        sys.exit(f"Could not parse Claude's response:\n{text}")
    return json.loads(match.group(0))


def snap(clip, words):
    """Snap Claude's rough timestamps to real word boundaries."""
    start = min(words, key=lambda w: abs(w["start"] - clip["start"]))["start"]
    end = min(words, key=lambda w: abs(w["end"] - clip["end"]))["end"]
    return start, end


def ts(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_ass(words, start, end, path: Path, vertical=True):
    w_res, h_res = (1080, 1920) if vertical else (1920, 1080)
    size, margin = (84, 420) if vertical else (60, 80)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w_res}
PlayResY: {h_res}

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,{size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,{margin},1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    ws = [w for w in words if w["start"] >= start - 0.05 and w["end"] <= end + 0.05]
    lines = []
    for i in range(0, len(ws), 3):
        chunk = ws[i:i + 3]
        t0 = chunk[0]["start"] - start
        t1 = chunk[-1]["end"] - start
        if i + 3 < len(ws):  # hold caption until the next one starts if the gap is small
            nxt = ws[i + 3]["start"] - start
            if nxt - t1 < 0.4:
                t1 = nxt
        text = " ".join(w["word"] for w in chunk).upper().replace("{", "").replace("}", "")
        lines.append(f"Dialogue: 0,{ts(max(t0, 0))},{ts(t1)},Default,,0,0,0,,{text}")
    path.write_text(header + "\n".join(lines), encoding="utf-8")


def render(video: Path, start, end, outdir: Path, idx: int, words, vertical):
    ass_name = f"clip_{idx:02d}.ass"
    out_name = f"clip_{idx:02d}.mp4"
    build_ass(words, start, end, outdir / ass_name, vertical)
    vf = ("crop=ih*9/16:ih,scale=1080:1920," if vertical else "") + f"ass={ass_name}"
    # Run inside outdir so the ass filter gets a plain relative path (avoids Windows colon issues)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}",
         "-i", str(video.resolve()), "-vf", vf,
         "-c:v", "libx264", "-preset", "fast", "-crf", "20",
         "-c:a", "aac", "-b:a", "160k", out_name],
        check=True, cwd=outdir,
    )
    return outdir / out_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="YouTube URL or local video file")
    p.add_argument("--clips", type=int, default=5)
    p.add_argument("--min", type=int, default=20, dest="min_len")
    p.add_argument("--max", type=int, default=55, dest="max_len")
    p.add_argument("--out", default="clips")
    p.add_argument("--whisper", default="small", help="tiny/base/small/medium/large-v3")
    p.add_argument("--no-crop", action="store_true", help="keep original aspect ratio")
    a = p.parse_args()

    outdir = Path(a.out)
    outdir.mkdir(exist_ok=True)

    video = Path(a.source) if Path(a.source).exists() else download(a.source, outdir)
    segs, words = transcribe(video, a.whisper)
    if not words:
        sys.exit("No speech found in the video.")

    print("Asking Claude for the best moments...")
    picks = pick_clips(segs, a.clips, a.min_len, a.max_len)

    for i, clip in enumerate(picks, 1):
        start, end = snap(clip, words)
        print(f"[{i}] {clip['title']}  ({start:.0f}s-{end:.0f}s)  {clip['reason']}")
        render(video, start, end, outdir, i, words, vertical=not a.no_crop)

    print(f"\nDone. Clips are in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
