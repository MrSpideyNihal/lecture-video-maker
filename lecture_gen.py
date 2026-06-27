
"""
AI Tutoring Lecture Video Generator
Uses: Ollama (local LLM) + Edge-TTS + Pexels API + FFmpeg + Pillow
Layout: stock video/image on top, subtitle bar on bottom
Subtitles rendered via Pillow drawtext overlay (Windows-safe, no libass needed)
"""

import os, sys, json, time, asyncio, shutil, re, requests, subprocess, math, threading, webbrowser
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# ─────────────────────────────────────────────────────────────
# CONFIG & PERSISTENCE
# ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
PEXELS_API_KEY  = ""                                        # Set via Web UI Settings or PEXELS_API_KEY env var
EDGE_TTS_VOICE  = "en-US-GuyNeural"
OUTPUT_DIR      = Path("lecture_output")

VIDEO_W   = 1280
VIDEO_H   = 720
SUB_H     = 180    # subtitle bar height (bottom)
VIS_H     = VIDEO_H - SUB_H   # visual area height (top)
FPS       = 25

CONFIG_FILE = Path("config.json")

def load_config():
    global OLLAMA_BASE_URL, PEXELS_API_KEY, EDGE_TTS_VOICE, OUTPUT_DIR
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                OLLAMA_BASE_URL = data.get("ollama_base_url", OLLAMA_BASE_URL)
                PEXELS_API_KEY = data.get("pexels_api_key", PEXELS_API_KEY)
                EDGE_TTS_VOICE = data.get("edge_tts_voice", EDGE_TTS_VOICE)
                if "output_dir" in data:
                    OUTPUT_DIR = Path(data["output_dir"])
        except Exception as e:
            print(f"Error loading config.json: {e}")

def save_config_fields(ollama_url, pexels_key, tts_voice, out_dir=None):
    global OLLAMA_BASE_URL, PEXELS_API_KEY, EDGE_TTS_VOICE, OUTPUT_DIR
    OLLAMA_BASE_URL = ollama_url
    PEXELS_API_KEY = pexels_key
    EDGE_TTS_VOICE = tts_voice
    if out_dir:
        OUTPUT_DIR = Path(out_dir)
    data = {
        "ollama_base_url": OLLAMA_BASE_URL,
        "pexels_api_key": PEXELS_API_KEY,
        "edge_tts_voice": EDGE_TTS_VOICE,
        "output_dir": str(OUTPUT_DIR)
    }
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving config.json: {e}")

# Initial load
load_config()
# ─────────────────────────────────────────────────────────────

class C:
    CYAN="\033[96m"; GREEN="\033[92m"; YELLOW="\033[93m"
    RED="\033[91m";  BOLD="\033[1m";   DIM="\033[2m"; RESET="\033[0m"

def banner():
    print(f"""{C.CYAN}{C.BOLD}
 ╔══════════════════════════════════════════════╗
 ║   🎓  AI TUTORING LECTURE VIDEO GENERATOR   ║
 ║   Ollama · Edge-TTS · Pexels · FFmpeg       ║
 ╚══════════════════════════════════════════════╝
{C.RESET}""")

def info(m): print(f"{C.CYAN}  ▸ {m}{C.RESET}")
def ok(m):   print(f"{C.GREEN}  ✔ {m}{C.RESET}")
def warn(m): print(f"{C.YELLOW}  ⚠ {m}{C.RESET}")
def err(m):  print(f"{C.RED}  ✖ {m}{C.RESET}")
def step(n,t): print(f"\n{C.BOLD}{C.CYAN}[{n}] {t}{C.RESET}")


# ── Dependency check ──────────────────────────────────────────
def check_deps():
    step("PRE", "Checking dependencies")
    for tool in ["ffmpeg", "ffprobe"]:
        if not shutil.which(tool):
            err(f"Missing: {tool}  →  https://ffmpeg.org/download.html")
            sys.exit(1)
    ok("ffmpeg found")
    missing = []
    for pkg, imp in [("edge_tts","edge_tts"),("requests","requests"),("PIL","PIL.Image"),("numpy","numpy")]:
        try: __import__(imp)
        except ImportError: missing.append(pkg)
    if missing:
        err(f"Missing packages: {', '.join(missing)}")
        print(f"  pip install edge-tts requests pillow numpy")
        sys.exit(1)
    ok("Python packages found")


# ── Ollama ────────────────────────────────────────────────────
def list_models():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models",[])]
    except: return []

def ollama_generate(model, prompt):
    r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.7, "num_ctx": 4096}
    }, timeout=300)
    if r.status_code != 200:
        err(f"Ollama {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
    return r.json()["response"].strip()

def ollama_chat(model, prompt, system=""):
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json={
            "model": model,
            "messages": [{"role":"system","content":system},
                         {"role":"user","content":prompt}],
            "stream": False, "options": {"temperature": 0.7, "num_ctx": 4096}
        }, timeout=300)
        if r.status_code == 500:
            warn("Chat 500 -- retrying with /api/generate ...")
            return ollama_generate(model, (system + "\n\n" + prompt).strip())
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception:
        warn("Chat failed -- falling back to /api/generate ...")
        return ollama_generate(model, (system + "\n\n" + prompt).strip())


def extract_json_array(raw):
    raw = re.sub(r'```[a-z]*', '', raw).replace('```', '').strip()
    if raw.startswith("["):
        try: return json.loads(raw)
        except: pass
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            for v in obj.values():
                if isinstance(v, list): return v
        except: pass
    m = re.search(r'\[.*\]', raw, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except: pass
    objects = re.findall(r'\{[^{}]+\}', raw, re.DOTALL)
    if objects:
        collected = []
        for o in objects:
            try: collected.append(json.loads(o))
            except: pass
        if collected: return collected
    raise ValueError(f"Cannot extract JSON from model output:\n{raw[:600]}")


# -- Script generation
@dataclass
class Segment:
    title: str
    narration: str
    keywords: list


def generate_script(model, topic, duration_min):
    # 130 wpm average TTS rate; minimum 80 words per segment so narration is never a stub
    MIN_WORDS_PER_SEG = 80
    n_seg = max(3, int(duration_min * 1.5))
    total_words = max(int(duration_min * 130), n_seg * MIN_WORDS_PER_SEG)
    words_per = total_words // n_seg

    # Step 1: Generate segment outline
    struct_prompt = (
        f'You are a professional educational scriptwriter.\n'
        f'Create a high-quality outline for a {duration_min}-minute educational lecture about: "{topic}".\n'
        f'We need exactly {n_seg} logical segments.\n\n'
        'Output a JSON array. Each object has exactly these 2 keys:\n'
        '  "title": section title (max 5 words)\n'
        '  "keywords": array of 3 concrete visual stock media search terms (e.g. ["students math class", "chalkboard write"])\n\n'
        'Output ONLY the raw JSON array starting with [ and ending with ]. No markdown formatting, no extra text.'
    )

    info(f"Generating lecture structure ({model}) with {n_seg} segments...")
    raw_struct = ollama_chat(model, struct_prompt, system="You output only valid raw JSON.")
    info("Parsing outline JSON...")
    data = extract_json_array(raw_struct)

    segs = []
    for item in data:
        kw = item.get("keywords", [topic])
        if isinstance(kw, str): kw = [kw]
        flat = []
        for k in kw:
            if isinstance(k, list): flat.extend(k)
            else: flat.append(str(k))
        segs.append(Segment(
            title=str(item.get("title", "Section")),
            narration="",
            keywords=flat[:3] or [topic]
        ))

    # Ensure we have at least 1 segment
    if not segs:
        segs = [Segment(title=f"Introduction to {topic}", narration="", keywords=[topic])]

    # Step 2: Generate detailed narration for each segment
    info(f"Generating narration for each of the {len(segs)} segments (target: at least {words_per} words per segment)...")
    for idx, seg in enumerate(segs):
        info(f"  Segment {idx+1}/{len(segs)}: {seg.title}")
        
        narration_prompt = (
            f'You are a professional educational lecturer. Write the spoken narration text for a segment titled "{seg.title}" '
            f'as part of a larger lecture on "{topic}".\n\n'
            f'Requirements:\n'
            f'- Write in a natural, engaging, and clear speaking/teaching style.\n'
            f'- Write AT LEAST {words_per} words of detailed explanations.\n'
            f'- Output ONLY the spoken narration text. Do not add any headings, intros, titles, or speaker/audio cues.'
        )
        
        # We can run a quick expansion loop if it's too short
        narration = ""
        for attempt in range(2):
            narration = ollama_chat(model, narration_prompt, system="You write spoken narration text only. Do not include markdown headers, bullet points, brackets, or speaker tags.")
            word_count = len(narration.split())
            if word_count >= words_per * 0.8:
                break
            # If too short, try to prompt for more detail
            narration_prompt += f"\n\nNote: Your previous attempt was only {word_count} words. Please expand the explanation with more examples and detail to be at least {words_per} words."

        seg.narration = narration
        ok(f"    Generated {len(narration.split())} words")

    actual_words = sum(len(s.narration.split()) for s in segs)
    actual_min = actual_words / 130
    ok(f"Script complete: {len(segs)} segments, {actual_words} words (~{actual_min:.1f} minutes of speech)")
    return segs



# ── Edge-TTS ──────────────────────────────────────────────────
async def _tts(text, path, voice):
    import edge_tts
    await edge_tts.Communicate(text, voice).save(str(path))

def synthesise(segs, audio_dir):
    step("2", "Synthesising speech (Edge-TTS)")
    paths = []
    for i, s in enumerate(segs):
        p = audio_dir / f"seg_{i:02d}.mp3"
        info(f"  [{i+1}/{len(segs)}] {s.title}")
        asyncio.run(_tts(s.narration, p, EDGE_TTS_VOICE))
        paths.append(p)
    ok("All audio done")
    return paths

def get_duration(p):
    r = subprocess.run(["ffprobe","-v","quiet","-print_format","json",
                        "-show_format", str(p)], capture_output=True, text=True)
    return float(json.loads(r.stdout)["format"]["duration"])


# ── Pexels fetch (video preferred, image fallback) ────────────
# ── Pexels fetch (video preferred, image fallback) ────────────
def pexels_fetch_multiple(queries, api_key, media_dir, idx, count_needed, media_pref="1"):
    """Fetch up to count_needed unique visuals. Returns list of (Path, 'video'|'photo')."""
    headers = {"Authorization": api_key}
    results = []
    downloaded_urls = set()
    
    # We will search with all queries. Let's collect potential candidates first.
    candidates = []
    
    for query in queries:
        if media_pref == "2":
            stypes = ["video"]
        elif media_pref == "3":
            stypes = ["photo"]
        else:
            stypes = ["video", "photo"]
            
        for stype in stypes:
            if stype == "video":
                try:
                    r = requests.get("https://api.pexels.com/videos/search", headers=headers,
                        params={"query":query,"per_page":15,"orientation":"landscape","size":"medium"},
                        timeout=15)
                    vids = r.json().get("videos",[])
                    for vid in vids:
                        for vf in vid.get("video_files",[]):
                            if vf.get("quality") in ("hd","sd") and vf.get("width",0)>=640:
                                link = vf["link"]
                                if link not in downloaded_urls:
                                    candidates.append({"type": "video", "url": link})
                                    downloaded_urls.add(link)
                                    break
                except Exception as e:
                    warn(f"    video search error for '{query}': {e}")
            elif stype == "photo":
                try:
                    r = requests.get("https://api.pexels.com/v1/search", headers=headers,
                        params={"query":query,"per_page":15,"orientation":"landscape"},
                        timeout=15)
                    photos = r.json().get("photos",[])
                    for photo in photos:
                        link = photo["src"]["large2x"]
                        if link not in downloaded_urls:
                            candidates.append({"type": "photo", "url": link})
                            downloaded_urls.add(link)
                except Exception as e:
                    warn(f"    photo search error for '{query}': {e}")
                    
    if len(candidates) < count_needed:
        fallback_stypes = []
        if media_pref == "2":
            fallback_stypes = ["photo"]
        elif media_pref == "3":
            fallback_stypes = ["video"]
            
        for query in queries:
            for stype in fallback_stypes:
                if stype == "video":
                    try:
                        r = requests.get("https://api.pexels.com/videos/search", headers=headers,
                            params={"query":query,"per_page":15,"orientation":"landscape","size":"medium"},
                            timeout=15)
                        vids = r.json().get("videos",[])
                        for vid in vids:
                            for vf in vid.get("video_files",[]):
                                if vf.get("quality") in ("hd","sd") and vf.get("width",0)>=640:
                                    link = vf["link"]
                                    if link not in downloaded_urls:
                                        candidates.append({"type": "video", "url": link})
                                        downloaded_urls.add(link)
                                        break
                    except Exception as e:
                        pass
                elif stype == "photo":
                    try:
                        r = requests.get("https://api.pexels.com/v1/search", headers=headers,
                            params={"query":query,"per_page":15,"orientation":"landscape"},
                            timeout=15)
                        photos = r.json().get("photos",[])
                        for photo in photos:
                            link = photo["src"]["large2x"]
                            if link not in downloaded_urls:
                                candidates.append({"type": "photo", "url": link})
                                downloaded_urls.add(link)
                    except Exception as e:
                        pass

    if not candidates:
        warn(f"    No Pexels results for segment {idx}, using color background.")
        return []

    final_candidates = []
    if media_pref == "1":
        vids = [c for c in candidates if c["type"] == "video"]
        imgs = [c for c in candidates if c["type"] == "photo"]
        i_vid, i_img = 0, 0
        prefer_video = (idx % 2 == 0)
        while len(final_candidates) < count_needed and (i_vid < len(vids) or i_img < len(imgs)):
            if prefer_video:
                if i_vid < len(vids):
                    final_candidates.append(vids[i_vid])
                    i_vid += 1
                elif i_img < len(imgs):
                    final_candidates.append(imgs[i_img])
                    i_img += 1
            else:
                if i_img < len(imgs):
                    final_candidates.append(imgs[i_img])
                    i_img += 1
                elif i_vid < len(vids):
                    final_candidates.append(vids[i_vid])
                    i_vid += 1
            prefer_video = not prefer_video
    else:
        final_candidates = candidates[:count_needed]

    if not final_candidates:
        warn(f"    No viable visuals after filtering for segment {idx}.")
        return []

    while len(final_candidates) < count_needed:
        final_candidates.append(final_candidates[len(final_candidates) % len(final_candidates)])

    for c_idx, cand in enumerate(final_candidates[:count_needed]):
        ext = ".mp4" if cand["type"] == "video" else ".jpg"
        dest = media_dir / f"visual_{idx:02d}_{c_idx:02d}{ext}"
        try:
            _download(cand["url"], dest)
            results.append((dest, cand["type"]))
            ok(f"    {cand['type']} {c_idx+1}/{count_needed}: fetched from url")
        except Exception as e:
            warn(f"    Failed downloading {cand['url']}: {e}")
            
    return results

def _download(url, dest):
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest,"wb") as f:
        for chunk in r.iter_content(1<<16): f.write(chunk)

def fetch_visuals(segs, media_dir, api_key, media_pref="1", durations=None):
    step("3", "Fetching Pexels visuals")
    results = []
    if durations is None:
        durations = [30.0] * len(segs)
    for i, seg in enumerate(segs):
        dur = durations[i]
        count_needed = max(1, math.ceil(dur / 8.0))
        info(f"  [{i+1}/{len(segs)}] {seg.title} → keywords: {seg.keywords} (needs {count_needed} visuals for {dur:.1f}s)")
        seg_visuals = pexels_fetch_multiple(seg.keywords, api_key, media_dir, i, count_needed, media_pref)
        results.append(seg_visuals)
    return results


# ── Subtitle image renderer (Pillow) ─────────────────────────
def render_subtitle_frames(narration, duration, out_dir, bar_w, bar_h):
    """
    Render subtitle text as PNG frames at FPS into out_dir.
    Returns the directory path (used as ffmpeg image sequence).
    Each frame = dark bar (bar_w x bar_h) with white wrapped text centred.
    """
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    words = narration.split()
    if not words: words = [""]
    # chunk into lines of ~8 words
    chunk_size = 8
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0,len(words),chunk_size)]
    if not chunks: chunks = [""]

    total_frames = int(math.ceil(duration * FPS))
    frames_per_chunk = max(1, total_frames // len(chunks))

    # Try to load a decent font; fall back to default
    font_size = max(28, bar_h // 5)
    font = None
    font_candidates = [
        "C:/Windows/Fonts/Arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for fc in font_candidates:
        if Path(fc).exists():
            try:
                font = ImageFont.truetype(fc, font_size)
                break
            except: pass
    if font is None:
        font = ImageFont.load_default()

    out_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    BG = (15, 15, 40)       # dark navy
    FG = (255, 255, 255)
    ACC = (100, 200, 255)   # accent for title line

    for ci, chunk in enumerate(chunks):
        n_frames = frames_per_chunk if ci < len(chunks)-1 else (total_frames - frame_idx)
        for _ in range(n_frames):
            img = Image.new("RGB", (bar_w, bar_h), BG)
            draw = ImageDraw.Draw(img)
            # centre text
            try:
                bbox = draw.textbbox((0,0), chunk, font=font)
                tw = bbox[2]-bbox[0]; th = bbox[3]-bbox[1]
            except:
                tw, th = len(chunk)*font_size//2, font_size
            x = max(20, (bar_w - tw)//2)
            y = max(10, (bar_h - th)//2)
            # shadow
            draw.text((x+2, y+2), chunk, font=font, fill=(0,0,0))
            draw.text((x, y), chunk, font=font, fill=FG)
            # thin accent line at top of bar
            draw.line([(0,0),(bar_w,0)], fill=ACC, width=3)
            img.save(out_dir / f"frame_{frame_idx:06d}.png")
            frame_idx += 1

    # pad remaining frames with last chunk if any
    while frame_idx < total_frames:
        img = Image.new("RGB", (bar_w, bar_h), BG)
        img.save(out_dir / f"frame_{frame_idx:06d}.png")
        frame_idx += 1

    return out_dir


# ── Build one segment video ───────────────────────────────────
# ── Build one segment video ───────────────────────────────────
def build_segment_video(seg, audio_path, visuals, sub_frames_dir, out_path, duration):
    """
    Layout: VIS_H px visual on top, SUB_H px subtitle bar on bottom.
    All done via ffmpeg complex filter + image sequence — no subtitles filter.
    """
    dur_s = str(duration)

    # ── Build subtitle bar video from PNG frames ──────────────
    sub_video = out_path.with_suffix(".subbar.mp4")
    cmd_sub = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(sub_frames_dir / "frame_%06d.png"),
        "-t", dur_s,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p",
        str(sub_video)
    ]
    r = subprocess.run(cmd_sub, capture_output=True, text=True)
    if r.returncode != 0:
        err("Subtitle bar render failed"); print(r.stderr[-2000:]); raise RuntimeError()

    # ── Build visual video ────────────────────────────────────
    vis_video = out_path.with_suffix(".vis.mp4")

    if not visuals:
        visuals = [(None, None)]

    num_clips = len(visuals)
    clip_dur = duration / num_clips
    clip_files = []

    for idx, (visual_path, visual_kind) in enumerate(visuals):
        clip_out = out_path.with_suffix(f".clip_{idx}.mp4")
        cur_dur = duration - (idx * clip_dur) if idx == num_clips - 1 else clip_dur
        cur_dur_s = str(cur_dur)

        if visual_path and visual_kind == "video":
            cmd_vis = [
                "ffmpeg", "-y",
                "-stream_loop", "-1", "-t", cur_dur_s, "-i", str(visual_path),
                "-vf", f"scale={VIDEO_W}:{VIS_H}:force_original_aspect_ratio=increase,crop={VIDEO_W}:{VIS_H},setsar=1,fps={FPS}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-an", str(clip_out)
            ]
        elif visual_path and visual_kind == "photo":
            n_frames = int(cur_dur * FPS)
            cmd_vis = [
                "ffmpeg", "-y",
                "-loop", "1", "-t", cur_dur_s, "-i", str(visual_path),
                "-vf", (
                    f"scale=2560:-1,"
                    f"zoompan=z='min(zoom+0.0004,1.4)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                    f":d={n_frames}:s={VIDEO_W}x{VIS_H}:fps={FPS},"
                    f"setsar=1"
                ),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-an", "-t", cur_dur_s, str(clip_out)
            ]
        else:
            title_safe = seg.title.replace("'","").replace(":","").replace("\\","")
            cmd_vis = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c=#0d1117:size={VIDEO_W}x{VIS_H}:rate={FPS}",
                "-t", cur_dur_s,
                "-vf", (
                    f"drawtext=text='{title_safe}':fontcolor=white:fontsize=48:"
                    f"x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=black@0.4:boxborderw=20"
                ),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-an", str(clip_out)
            ]

        r = subprocess.run(cmd_vis, capture_output=True, text=True)
        if r.returncode != 0:
            err(f"Visual clip {idx} render failed"); print(r.stderr[-2000:]); raise RuntimeError()
        clip_files.append(clip_out)

    if len(clip_files) == 1:
        if vis_video.exists():
            vis_video.unlink()
        shutil.move(str(clip_files[0]), str(vis_video))
    else:
        concat_list = out_path.with_suffix(".vis_concat.txt")
        with open(concat_list, "w") as f:
            for cf in clip_files:
                f.write(f"file '{str(cf.resolve()).replace('\\','/')}'\n")

        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(vis_video)
        ]
        r = subprocess.run(cmd_concat, capture_output=True, text=True)
        concat_list.unlink(missing_ok=True)
        for cf in clip_files:
            cf.unlink(missing_ok=True)
        if r.returncode != 0:
            err("Visual clip concatenation failed"); print(r.stderr[-2000:]); raise RuntimeError()

    # ── Stack visual + subtitle bar, add audio ────────────────
    cmd_final = [
        "ffmpeg", "-y",
        "-i", str(vis_video),
        "-i", str(sub_video),
        "-i", str(audio_path),
        "-filter_complex",
        "[0:v][1:v]vstack=inputs=2[v]",
        "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-t", dur_s,
        str(out_path)
    ]
    r = subprocess.run(cmd_final, capture_output=True, text=True)
    # cleanup intermediates
    vis_video.unlink(missing_ok=True)
    sub_video.unlink(missing_ok=True)
    if r.returncode != 0:
        err("Final stack failed"); print(r.stderr[-2000:]); raise RuntimeError()


# ── Concat ────────────────────────────────────────────────────
def concatenate(seg_videos, out_path):
    list_file = out_path.parent / "concat_list.txt"
    # Use forward slashes for safety across platforms
    with open(list_file, "w") as f:
        for v in seg_videos:
            p = str(v.resolve()).replace("\\","/")
            f.write(f"file '{p}'\n")
    r = subprocess.run([
        "ffmpeg", "-y", "-f","concat","-safe","0",
        "-i", str(list_file), "-c","copy", str(out_path)
    ], capture_output=True, text=True)
    list_file.unlink(missing_ok=True)
    if r.returncode != 0:
        err("Concatenation failed"); print(r.stderr[-2000:]); raise RuntimeError()


# ── Main ──────────────────────────────────────────────────────
# Shared state for background rendering task
render_status = {
    "status": "idle", # "idle", "running", "done", "error"
    "progress": 0,    # 0 to 100
    "message": "",
    "logs": [],       # list of log strings
    "video_url": "",  # web path to final video
    "video_path": ""  # local file path
}

def log_progress(msg, progress=None):
    global render_status
    timestamp = time.strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    render_status["logs"].append(full_msg)
    render_status["message"] = msg
    if progress is not None:
        render_status["progress"] = progress

def compile_video_job(run_id, topic, voice, media_pref, segments_data, api_key):
    global render_status
    try:
        render_status["status"] = "running"
        render_status["progress"] = 0
        render_status["logs"] = []
        render_status["video_url"] = ""
        render_status["video_path"] = ""

        # 1. Dirs
        log_progress("Creating directories...", 5)
        run_dir = OUTPUT_DIR / run_id
        audio_d = run_dir / "audio"
        media_d = run_dir / "media"
        subs_d  = run_dir / "subs"
        segs_d  = run_dir / "segments"
        for d in [audio_d, media_d, subs_d, segs_d]:
            d.mkdir(parents=True, exist_ok=True)

        # Re-create Segment objects from frontend data
        segs = []
        for s in segments_data:
            segs.append(Segment(
                title=s["title"],
                narration=s["narration"],
                keywords=s["keywords"]
            ))

        # Save script.txt
        script_path = run_dir / "script.txt"
        script_path.write_text("\n\n".join(f"## {s.title}\n{s.narration}" for s in segs), encoding="utf-8")
        log_progress(f"Saved script file to output folder.", 10)

        # 2. TTS
        log_progress("Synthesizing speech via Edge-TTS...", 15)
        audio_files = []
        for idx, s in enumerate(segs):
            p = audio_d / f"seg_{idx:02d}.mp3"
            log_progress(f"  Synthesizing audio for segment {idx+1}/{len(segs)}: {s.title}...", 15 + int(20 * idx / len(segs)))
            asyncio.run(_tts(s.narration, p, voice))
            audio_files.append(p)

        durations = [get_duration(a) for a in audio_files]
        total_dur = sum(durations)
        log_progress(f"Audio synthesis complete. Total duration: {total_dur/60:.1f} mins.", 35)

        # 3. Visuals
        log_progress("Fetching stock visuals from Pexels...", 40)
        visuals = []
        for idx, seg in enumerate(segs):
            dur = durations[idx]
            count_needed = max(1, math.ceil(dur / 8.0))
            log_progress(f"  Fetching {count_needed} visuals for segment {idx+1}: {seg.title}...", 40 + int(20 * idx / len(segs)))
            seg_visuals = pexels_fetch_multiple(seg.keywords, api_key, media_d, idx, count_needed, media_pref)
            visuals.append(seg_visuals)

        # 4. Render Segments
        log_progress("Rendering individual segment videos (FFmpeg + Pillow overlays)...", 60)
        seg_videos = []
        for idx, (seg, audio, seg_visuals, dur) in enumerate(zip(segs, audio_files, visuals, durations)):
            log_progress(f"  Rendering Segment {idx+1}/{len(segs)}: {seg.title}...", 60 + int(25 * idx / len(segs)))
            
            sub_dir = subs_d / f"seg_{idx:02d}"
            render_subtitle_frames(seg.narration, dur, sub_dir, VIDEO_W, SUB_H)
            
            out_path = segs_d / f"seg_{idx:02d}.mp4"
            build_segment_video(seg, audio, seg_visuals, sub_dir, out_path, dur)
            seg_videos.append(out_path)

        # 5. Concat
        log_progress("Assembling final video file...", 90)
        final_name = re.sub(r'[^\w\- ]','', topic)[:40].replace(" ","_")
        final_video = OUTPUT_DIR / f"{final_name}_{run_id}.mp4"
        concatenate(seg_videos, final_video)

        log_progress("Video assembly complete!", 100)
        render_status["status"] = "done"
        render_status["video_url"] = f"/output/{final_video.name}"
        render_status["video_path"] = str(final_video.resolve())

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        err_msg = f"Error during rendering: {e}\n{tb}"
        print(err_msg)
        log_progress(f"ERROR: {e}", 100)
        render_status["status"] = "error"
        render_status["message"] = f"Render failed: {e}"

def run_cli():
    banner()
    check_deps()

    # Pexels key
    api_key = PEXELS_API_KEY or os.environ.get("PEXELS_API_KEY","")
    if not api_key:
        print(f"\n{C.YELLOW}Pexels API key needed — get free key at https://www.pexels.com/api/{C.RESET}")
        api_key = input("  Paste Pexels API key: ").strip()
        if not api_key: err("No key."); sys.exit(1)

    # Ollama model
    step("1a", "Available Ollama models")
    models = list_models()
    if not models:
        err("Ollama not running. Run: ollama serve"); sys.exit(1)
    for i,m in enumerate(models):
        print(f"   {C.DIM}[{i}]{C.RESET} {m}")
    choice = input(f"\n  {C.BOLD}Select model number (or type name): {C.RESET}").strip()
    model = models[int(choice)] if choice.isdigit() else (choice or models[0])
    ok(f"Using: {model}")

    # Topic & duration & media preference
    step("1b", "Lecture parameters")
    topic = input(f"  {C.BOLD}Topic: {C.RESET}").strip()
    if not topic: err("Topic empty."); sys.exit(1)
    try: duration_min = float(input(f"  {C.BOLD}Duration (minutes): {C.RESET}").strip())
    except: err("Invalid duration."); sys.exit(1)

    print(f"\n  {C.BOLD}Select Visual Media Preference:{C.RESET}")
    print(f"   [1] Mix of Videos and Images (default)")
    print(f"   [2] Only Videos")
    print(f"   [3] Only Images")
    media_pref = input("  Choice (1-3): ").strip() or "1"
    if media_pref not in ["1", "2", "3"]:
        media_pref = "1"

    # Dirs
    run_id  = str(int(time.time()))
    run_dir = OUTPUT_DIR / run_id
    audio_d = run_dir / "audio"
    media_d = run_dir / "media"
    subs_d  = run_dir / "subs"
    segs_d  = run_dir / "segments"
    for d in [audio_d, media_d, subs_d, segs_d]:
        d.mkdir(parents=True, exist_ok=True)

    # Generate script
    step("1c", "Generating lecture script")
    segs = generate_script(model, topic, duration_min)
    script_path = run_dir / "script.txt"
    script_path.write_text("\n\n".join(f"## {s.title}\n{s.narration}" for s in segs), encoding="utf-8")
    ok(f"Script → {script_path}")

    # TTS
    audio_files = synthesise(segs, audio_d)
    durations   = [get_duration(a) for a in audio_files]
    total_dur   = sum(durations)
    ok(f"Total audio: {total_dur/60:.1f} min")

    # Visuals
    visuals = fetch_visuals(segs, media_d, api_key, media_pref, durations)

    # Render segments
    step("4", "Rendering segment videos")
    seg_videos = []
    for i,(seg,audio,seg_visuals,dur) in enumerate(
            zip(segs, audio_files, visuals, durations)):
        info(f"  [{i+1}/{len(segs)}] {seg.title}  ({dur:.1f}s)")

        # Render subtitle frames with Pillow
        sub_dir = subs_d / f"seg_{i:02d}"
        render_subtitle_frames(seg.narration, dur, sub_dir, VIDEO_W, SUB_H)

        out_path = segs_d / f"seg_{i:02d}.mp4"
        build_segment_video(seg, audio, seg_visuals, sub_dir, out_path, dur)
        seg_videos.append(out_path)
        ok(f"    ✔ Segment {i+1} done")

    # Concat
    step("5", "Assembling final video")
    final_name  = re.sub(r'[^\w\- ]','', topic)[:40].replace(" ","_")
    final_video = OUTPUT_DIR / f"{final_name}_{run_id}.mp4"
    concatenate(seg_videos, final_video)

    print(f"""
{C.GREEN}{C.BOLD}
 ╔══════════════════════════════════════════════╗
 ║         🎉  LECTURE VIDEO READY!            ║
 ╚══════════════════════════════════════════════╝{C.RESET}
   {C.BOLD}File:    {C.RESET}{final_video}
   {C.BOLD}Duration:{C.RESET} {total_dur/60:.1f} min
   {C.BOLD}Segments:{C.RESET} {len(segs)}
   {C.BOLD}Script:  {C.RESET}{script_path}
""")

HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎓 AI Lecture Generator</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: #111827;
            --card-border: #1f2937;
            --primary: #06b6d4;
            --primary-hover: #0891b2;
            --accent: #8b5cf6;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --success: #10b981;
            --error: #ef4444;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            padding: 2rem 1rem;
            min-height: 100vh;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        header h1 {
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 1.1rem;
        }

        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
            transition: border-color 0.2s;
        }

        .card:hover {
            border-color: #374151;
        }

        .card-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: var(--primary);
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 0.75rem;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 768px) {
            .grid-2 {
                grid-template-columns: 1fr;
            }
        }

        .form-group {
            margin-bottom: 1.25rem;
        }

        label {
            display: block;
            font-size: 0.9rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            color: var(--text-muted);
        }

        input[type="text"], input[type="number"], select, textarea {
            width: 100%;
            padding: 0.75rem 1rem;
            background-color: #1f2937;
            border: 1px solid var(--card-border);
            border-radius: 8px;
            color: var(--text-main);
            font-family: inherit;
            font-size: 1rem;
            transition: outline 0.15s, border-color 0.15s;
        }

        input:focus, select:focus, textarea:focus {
            outline: 2px solid var(--primary);
            border-color: transparent;
        }

        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-family: inherit;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            border: none;
            transition: background-color 0.2s, transform 0.1s;
            gap: 0.5rem;
            text-decoration: none;
        }

        .btn-primary {
            background-color: var(--primary);
            color: #0b0f19;
        }

        .btn-primary:hover {
            background-color: var(--primary-hover);
        }

        .btn-accent {
            background-color: var(--accent);
            color: var(--text-main);
        }

        .btn-accent:hover {
            background-color: #7c3aed;
        }

        .btn-danger {
            background-color: var(--error);
            color: var(--text-main);
        }

        .btn-danger:hover {
            background-color: #dc2626;
        }

        .btn-secondary {
            background-color: #374151;
            color: var(--text-main);
        }

        .btn-secondary:hover {
            background-color: #4b5563;
        }

        .btn-sm {
            padding: 0.4rem 0.8rem;
            font-size: 0.85rem;
            border-radius: 6px;
        }

        .btn:active {
            transform: scale(0.98);
        }

        .segment-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            position: relative;
        }

        .segment-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .segment-num {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--primary);
        }

        .segment-actions {
            display: flex;
            gap: 0.5rem;
        }

        .word-counter {
            text-align: right;
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        .console {
            background-color: #020617;
            border: 1px solid #1e293b;
            font-family: monospace;
            padding: 1rem;
            border-radius: 8px;
            height: 250px;
            overflow-y: auto;
            color: #38bdf8;
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
            white-space: pre-wrap;
        }

        .progress-bar-container {
            width: 100%;
            height: 10px;
            background-color: #1e293b;
            border-radius: 5px;
            overflow: hidden;
            margin-bottom: 1rem;
        }

        .progress-bar {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, var(--primary), var(--accent));
            transition: width 0.3s ease;
        }

        .video-container {
            margin-top: 1.5rem;
            text-align: center;
        }

        video {
            width: 100%;
            max-height: 480px;
            border-radius: 12px;
            border: 1px solid var(--card-border);
            background-color: #000;
            margin-bottom: 1.5rem;
        }

        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .status-running {
            background-color: rgba(14, 165, 233, 0.2);
            color: #38bdf8;
        }

        .status-done {
            background-color: rgba(16, 185, 129, 0.2);
            color: #34d399;
        }

        .status-error {
            background-color: rgba(239, 68, 68, 0.2);
            color: #f87171;
        }

        .flex-between {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .loading-spinner {
            border: 3px solid rgba(255, 255, 255, 0.1);
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border-left-color: var(--primary);
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-right: 0.5rem;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .alert {
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            font-size: 0.95rem;
        }

        .alert-warning {
            background-color: rgba(245, 158, 11, 0.15);
            border: 1px solid rgba(245, 158, 11, 0.3);
            color: #fbbf24;
        }

        .alert-info {
            background-color: rgba(6, 182, 212, 0.1);
            border: 1px solid rgba(6, 182, 212, 0.2);
            color: #67e8f9;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎓 AI Lecture Generator</h1>
            <p>Convert any academic topic into custom tutoring videos with stock clips & subtitles</p>
        </header>

        <!-- Step 1: Configuration -->
        <div class="card" id="config-card">
            <div class="card-title">
                <span>⚙️ Configurations</span>
            </div>
            
            <div class="form-group">
                <label for="config-pexels-key">Pexels API Key</label>
                <input type="text" id="config-pexels-key" placeholder="Enter Pexels API Key">
            </div>

            <div class="grid-2">
                <div class="form-group">
                    <label for="config-ollama-url">Ollama Base URL</label>
                    <input type="text" id="config-ollama-url" value="http://localhost:11434">
                </div>
                <div class="form-group">
                    <label for="config-voice">Default Voiceover</label>
                    <select id="config-voice"></select>
                </div>
            </div>

            <div style="text-align: right;">
                <button class="btn btn-secondary" onclick="saveConfig()">Save Settings</button>
            </div>
        </div>

        <!-- Step 2: Generation Params -->
        <div class="card" id="params-card">
            <div class="card-title">
                <span>📝 1. Topic & Parameters</span>
            </div>

            <div class="form-group">
                <label for="input-topic">Lecture Topic</label>
                <input type="text" id="input-topic" placeholder="e.g., Photosynthesis process, Introduction to Python Loops">
            </div>

            <div class="grid-2">
                <div class="form-group">
                    <label for="input-model">Ollama Model</label>
                    <select id="input-model">
                        <option value="">Select a model</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="input-duration">Target Duration (minutes)</label>
                    <input type="number" id="input-duration" value="1.0" min="0.5" max="15" step="0.5">
                </div>
            </div>

            <div class="form-group">
                <label for="input-pref">Visual Media Preference</label>
                <select id="input-pref">
                    <option value="1">Mix of Videos and Images</option>
                    <option value="2">Only Videos</option>
                    <option value="3">Only Images</option>
                </select>
            </div>

            <div style="text-align: right;">
                <button class="btn btn-primary" id="btn-generate-script" onclick="generateScript()">
                    Generate Lecture Outline & Script
                </button>
            </div>
        </div>

        <!-- Step 3: Script Editor -->
        <div class="card" id="editor-card" style="display: none;">
            <div class="card-title">
                <span>✏️ 2. Edit Lecture Script & Keywords</span>
                <span style="font-size: 0.9rem; font-weight: normal; color: var(--text-muted);">
                    Review or modify narration text and visual terms before rendering.
                </span>
            </div>

            <div id="segments-container"></div>

            <div style="margin-bottom: 2rem; display: flex; justify-content: space-between;">
                <button class="btn btn-secondary btn-sm" onclick="addSegment()">+ Add Segment</button>
                <span id="total-word-count" style="color: var(--text-muted); font-size: 0.95rem;"></span>
            </div>

            <div style="border-top: 1px solid var(--card-border); padding-top: 1.5rem; text-align: right;">
                <button class="btn btn-accent btn-lg" onclick="startRender()">
                    🎬 Render Final Video
                </button>
            </div>
        </div>

        <!-- Step 4: Render Log & Progress -->
        <div class="card" id="render-card" style="display: none;">
            <div class="card-title">
                <span>🔨 3. Render Status</span>
                <span id="render-badge" class="status-badge status-running">Running</span>
            </div>

            <div class="progress-bar-container">
                <div class="progress-bar" id="progress-bar"></div>
            </div>

            <div class="flex-between" style="margin-bottom: 1.5rem;">
                <span id="status-message" style="font-weight: 500;">Starting compilation...</span>
                <span id="progress-percent" style="color: var(--primary); font-weight: 600;">0%</span>
            </div>

            <div class="console" id="console-logs"></div>

            <!-- Step 5: Completed Player -->
            <div id="preview-section" style="display: none;" class="video-container">
                <video id="video-player" controls></video>
                <div style="display: flex; gap: 1rem; justify-content: center;">
                    <button class="btn btn-primary" onclick="openOutputFolder()">📂 Open Outputs Folder</button>
                    <a id="btn-download" class="btn btn-secondary" download>💾 Download MP4</a>
                </div>
            </div>
        </div>
    </div>

    <script>
        let segments = [];
        let pollingInterval = null;

        document.addEventListener('DOMContentLoaded', () => {
            loadConfig();
            loadVoices();
        });

        async function loadConfig() {
            try {
                const res = await fetch('/api/config');
                const data = await res.json();
                document.getElementById('config-pexels-key').value = data.pexels_api_key || '';
                document.getElementById('config-ollama-url').value = data.ollama_base_url || '';
                
                // Fetch models after setting url
                loadModels(data.ollama_base_url);
            } catch (e) {
                console.error("Error fetching config", e);
            }
        }

        async function saveConfig() {
            const pexels_key = document.getElementById('config-pexels-key').value;
            const ollama_url = document.getElementById('config-ollama-url').value;
            const voice = document.getElementById('config-voice').value;
            
            try {
                const res = await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        pexels_api_key: pexels_key,
                        ollama_base_url: ollama_url,
                        edge_tts_voice: voice
                    })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    alert("Settings saved successfully!");
                    loadModels(ollama_url);
                }
            } catch (e) {
                alert("Error saving settings: " + e);
            }
        }

        async function loadModels(url) {
            const select = document.getElementById('input-model');
            select.innerHTML = '<option value="">Loading models...</option>';
            try {
                const res = await fetch('/api/models');
                const data = await res.json();
                select.innerHTML = '';
                if (data.models && data.models.length > 0) {
                    data.models.forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = m;
                        opt.textContent = m;
                        select.appendChild(opt);
                    });
                } else {
                    select.innerHTML = '<option value="">No local models found. Start Ollama!</option>';
                }
            } catch (e) {
                select.innerHTML = '<option value="">Failed to connect to Ollama</option>';
            }
        }

        async function loadVoices() {
            const select = document.getElementById('config-voice');
            try {
                const res = await fetch('/api/voices');
                const voices = await res.json();
                
                // Fetch config to check current voice
                const configRes = await fetch('/api/config');
                const configData = await configRes.json();
                const curVoice = configData.edge_tts_voice || 'en-US-GuyNeural';

                select.innerHTML = '';
                voices.forEach(v => {
                    const opt = document.createElement('option');
                    opt.value = v.id;
                    opt.textContent = v.name;
                    if (v.id === curVoice) opt.selected = true;
                    select.appendChild(opt);
                });
            } catch (e) {
                select.innerHTML = '<option value="en-US-GuyNeural">US Male - Guy</option>';
            }
        }

        async function generateScript() {
            const topic = document.getElementById('input-topic').value.trim();
            const model = document.getElementById('input-model').value;
            const duration = document.getElementById('input-duration').value;
            const btn = document.getElementById('btn-generate-script');

            if (!topic) {
                alert("Please enter a lecture topic.");
                return;
            }
            if (!model) {
                alert("Please select an Ollama model.");
                return;
            }

            btn.disabled = true;
            btn.innerHTML = '<span class="loading-spinner"></span> Generating script segments...';
            
            document.getElementById('editor-card').style.display = 'none';

            try {
                const res = await fetch('/api/generate_script', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ topic, model, duration_min: parseFloat(duration) })
                });
                const data = await res.json();
                if (data.error) {
                    alert("Error generating script: " + data.error);
                } else if (data.segments) {
                    segments = data.segments;
                    renderSegmentsEditor();
                    document.getElementById('editor-card').style.display = 'block';
                    document.getElementById('editor-card').scrollIntoView({ behavior: 'smooth' });
                }
            } catch (e) {
                alert("Failed to connect to script generator server: " + e);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Generate Lecture Outline & Script';
            }
        }

        function renderSegmentsEditor() {
            const container = document.getElementById('segments-container');
            container.innerHTML = '';
            
            let totalWords = 0;

            segments.forEach((seg, idx) => {
                const wordCount = seg.narration.split(/\s+/).filter(Boolean).length;
                totalWords += wordCount;

                const card = document.createElement('div');
                card.className = 'segment-card';
                card.innerHTML = `
                    <div class="segment-header">
                        <span class="segment-num">Segment ${idx + 1}</span>
                        <div class="segment-actions">
                            <button class="btn btn-secondary btn-sm" onclick="moveSeg(${idx}, -1)" \${idx === 0 ? 'disabled' : ''}>▲</button>
                            <button class="btn btn-secondary btn-sm" onclick="moveSeg(${idx}, 1)" \${idx === segments.length - 1 ? 'disabled' : ''}>▼</button>
                            <button class="btn btn-danger btn-sm" onclick="deleteSeg(${idx})">🗑️ Delete</button>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Segment Title</label>
                        <input type="text" class="seg-title" value="\${escapeHtml(seg.title)}" onchange="updateSegData(\${idx}, 'title', this.value)">
                    </div>
                    <div class="form-group">
                        <label>Visual Search Keywords (Comma-separated)</label>
                        <input type="text" class="seg-keywords" value="\${escapeHtml(seg.keywords.join(', '))}" onchange="updateSegKeywords(\${idx}, this.value)">
                    </div>
                    <div class="form-group">
                        <label>Narration / Spoken Text</label>
                        <textarea rows="4" oninput="updateSegNarration(\${idx}, this.value)">\${escapeHtml(seg.narration)}</textarea>
                        <div class="word-counter"><span id="word-count-\${idx}">\${wordCount}</span> words</div>
                    </div>
                `;
                container.appendChild(card);
            });

            document.getElementById('total-word-count').textContent = `Total Word Count: \${totalWords} words (~\${(totalWords/130).toFixed(1)} mins of speech)`;
        }

        function escapeHtml(str) {
            return str
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        function updateSegData(idx, field, value) {
            segments[idx][field] = value;
        }

        function updateSegKeywords(idx, value) {
            segments[idx].keywords = value.split(',').map(s => s.trim()).filter(Boolean);
        }

        function updateSegNarration(idx, value) {
            segments[idx].narration = value;
            const count = value.split(/\s+/).filter(Boolean).length;
            document.getElementById(`word-count-\${idx}`).textContent = count;
            
            // Recalculate total
            let total = 0;
            segments.forEach(s => {
                total += s.narration.split(/\s+/).filter(Boolean).length;
            });
            document.getElementById('total-word-count').textContent = `Total Word Count: \${total} words (~\${(total/130).toFixed(1)} mins of speech)`;
        }

        function moveSeg(idx, dir) {
            const target = idx + dir;
            if (target < 0 || target >= segments.length) return;
            const temp = segments[idx];
            segments[idx] = segments[target];
            segments[target] = temp;
            renderSegmentsEditor();
        }

        function deleteSeg(idx) {
            if (confirm("Are you sure you want to delete this segment?")) {
                segments.splice(idx, 1);
                renderSegmentsEditor();
            }
        }

        function addSegment() {
            segments.push({
                title: "New Section",
                narration: "Type segment speech details here.",
                keywords: ["education"]
            });
            renderSegmentsEditor();
        }

        async function startRender() {
            const topic = document.getElementById('input-topic').value.trim();
            const voice = document.getElementById('config-voice').value;
            const media_pref = document.getElementById('input-pref').value;

            document.getElementById('render-card').style.display = 'block';
            document.getElementById('render-card').scrollIntoView({ behavior: 'smooth' });
            
            // Clear progress values
            document.getElementById('progress-bar').style.width = '0%';
            document.getElementById('progress-percent').textContent = '0%';
            document.getElementById('status-message').textContent = 'Submitting job...';
            document.getElementById('console-logs').innerHTML = '';
            document.getElementById('preview-section').style.display = 'none';

            const badge = document.getElementById('render-badge');
            badge.textContent = 'Submitting';
            badge.className = 'status-badge status-running';

            try {
                const res = await fetch('/api/render', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        topic,
                        voice,
                        media_pref,
                        segments
                    })
                });
                const data = await res.json();
                if (data.error) {
                    alert("Failed to render: " + data.error);
                } else {
                    // Start polling status
                    if (pollingInterval) clearInterval(pollingInterval);
                    pollingInterval = setInterval(pollStatus, 1000);
                }
            } catch (e) {
                alert("Error starting render: " + e);
            }
        }

        async function pollStatus() {
            try {
                const res = await fetch('/api/progress');
                const data = await res.json();
                
                // Update progress metrics
                document.getElementById('progress-bar').style.width = data.progress + '%';
                document.getElementById('progress-percent').textContent = data.progress + '%';
                document.getElementById('status-message').textContent = data.message;
                
                // Update console logs
                const consoleLogs = document.getElementById('console-logs');
                consoleLogs.innerHTML = '';
                data.logs.forEach(log => {
                    const div = document.createElement('div');
                    div.textContent = log;
                    consoleLogs.appendChild(div);
                });
                consoleLogs.scrollTop = consoleLogs.scrollHeight; // Autoscroll

                const badge = document.getElementById('render-badge');

                if (data.status === 'running') {
                    badge.textContent = 'Compiling';
                    badge.className = 'status-badge status-running';
                } else if (data.status === 'done') {
                    badge.textContent = 'Completed';
                    badge.className = 'status-badge status-done';
                    clearInterval(pollingInterval);
                    
                    // Set up video preview
                    const videoPlayer = document.getElementById('video-player');
                    videoPlayer.src = data.video_url;
                    videoPlayer.load();
                    
                    const btnDownload = document.getElementById('btn-download');
                    btnDownload.href = data.video_url;

                    document.getElementById('preview-section').style.display = 'block';
                } else if (data.status === 'error') {
                    badge.textContent = 'Failed';
                    badge.className = 'status-badge status-error';
                    clearInterval(pollingInterval);
                }
            } catch (e) {
                console.error("Error polling progress", e);
            }
        }

        async function openOutputFolder() {
            try {
                await fetch('/api/open_folder', { method: 'POST' });
            } catch (e) {
                console.error("Error opening output folder", e);
            }
        }
    </script>
</body>
</html>"""

class WebAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence default server logging in CLI to keep it clean
        pass

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        
        if url.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
            return
            
        elif url.path == "/api/config":
            data = {
                "ollama_base_url": OLLAMA_BASE_URL,
                "pexels_api_key": PEXELS_API_KEY,
                "edge_tts_voice": EDGE_TTS_VOICE
            }
            self.send_json(data)
            return
            
        elif url.path == "/api/models":
            models = list_models()
            self.send_json({"models": models})
            return
            
        elif url.path == "/api/voices":
            voices = [
                {"id": "en-US-GuyNeural", "name": "US Male - Guy"},
                {"id": "en-US-JennyNeural", "name": "US Female - Jenny"},
                {"id": "en-GB-RyanNeural", "name": "UK Male - Ryan"},
                {"id": "en-GB-SoniaNeural", "name": "UK Female - Sonia"},
                {"id": "en-IN-NeerjaNeural", "name": "Indian Female - Neerja"},
                {"id": "en-IN-PrabhatNeural", "name": "Indian Male - Prabhat"},
                {"id": "en-AU-NatashaNeural", "name": "Australian Female - Natasha"},
                {"id": "en-AU-WilliamNeural", "name": "Australian Male - William"}
            ]
            self.send_json(voices)
            return
            
        elif url.path == "/api/progress":
            global render_status
            self.send_json(render_status)
            return
            
        elif url.path.startswith("/output/"):
            filename = url.path[8:]
            if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
                self.send_error(400, "Bad Request")
                return
            filepath = OUTPUT_DIR / filename
            if filepath.exists() and filepath.is_file():
                ctype = "video/mp4" if filename.endswith(".mp4") else "application/octet-stream"
                try:
                    stat = os.stat(filepath)
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(stat.st_size))
                    self.end_headers()
                    with open(filepath, "rb") as f:
                        shutil.copyfileobj(f, self.wfile)
                except Exception as e:
                    self.send_error(500, f"Error reading file: {e}")
            else:
                self.send_error(404, "File not found")
            return
            
        self.send_error(404, "Not Found")

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            params = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            self.send_error(400, "Invalid JSON body")
            return

        if url.path == "/api/config":
            ollama_url = params.get("ollama_base_url")
            pexels_key = params.get("pexels_api_key", "")
            tts_voice = params.get("edge_tts_voice")
            if ollama_url is not None and tts_voice is not None:
                save_config_fields(ollama_url, pexels_key, tts_voice)
                self.send_json({"status": "success"})
            else:
                self.send_error(400, "Missing required fields: ollama_base_url, edge_tts_voice")
            return
            
        elif url.path == "/api/generate_script":
            model = params.get("model")
            topic = params.get("topic")
            duration_min = params.get("duration_min")
            if not model or not topic or not duration_min:
                self.send_error(400, "Missing parameters")
                return
            try:
                segs = generate_script(model, topic, float(duration_min))
                result = [{"title": s.title, "narration": s.narration, "keywords": s.keywords} for s in segs]
                self.send_json({"segments": result})
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                self.send_json({"error": str(e)}, 500)
            return
            
        elif url.path == "/api/render":
            global render_status
            if render_status["status"] == "running":
                self.send_json({"error": "Render already in progress"}, status_code=400)
                return
                
            topic = params.get("topic")
            voice = params.get("voice", EDGE_TTS_VOICE)
            media_pref = params.get("media_pref", "1")
            segments = params.get("segments")
            
            if not topic or not segments:
                self.send_error(400, "Missing parameters")
                return
                
            run_id = str(int(time.time()))
            t = threading.Thread(
                target=compile_video_job,
                args=(run_id, topic, voice, media_pref, segments, PEXELS_API_KEY)
            )
            t.daemon = True
            t.start()
            self.send_json({"status": "started", "run_id": run_id})
            return
            
        elif url.path == "/api/open_folder":
            try:
                if not OUTPUT_DIR.exists():
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                if sys.platform == "win32":
                    os.startfile(str(OUTPUT_DIR.resolve()))
                else:
                    opener = "open" if sys.platform == "darwin" else "xdg-open"
                    subprocess.run([opener, str(OUTPUT_DIR.resolve())])
                self.send_json({"status": "success"})
            except Exception as e:
                self.send_json({"error": str(e)}, status_code=500)
            return

        self.send_error(404, "Not Found")

    def send_json(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Tutoring Lecture Video Generator")
    parser.add_argument("--cli", action="store_true", help="Run in Command Line Interface mode")
    parser.add_argument("--port", type=int, default=8080, help="Web UI port (default: 8080)")
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        # Check dependencies first (non-fatal for web mode — show warning instead)
        try:
            check_deps()
        except SystemExit:
            print(f"{C.YELLOW}⚠ Some dependencies are missing. The web UI will start but rendering may fail.{C.RESET}")

        server_address = ('127.0.0.1', args.port)
        httpd = HTTPServer(server_address, WebAPIHandler)
        print(f"
{C.GREEN}{C.BOLD}🎓 Server started at http://localhost:{args.port}{C.RESET}")
        print(f"{C.CYAN}Opening web interface in your browser...{C.RESET}")
        print(f"{C.DIM}Press Ctrl+C to stop the server.{C.RESET}
")

        def open_browser():
            time.sleep(1.0)
            webbrowser.open(f"http://localhost:{args.port}")

        t = threading.Thread(target=open_browser)
        t.daemon = True
        t.start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"
{C.YELLOW}Stopping server...{C.RESET}")
            httpd.server_close()
            sys.exit(0)

if __name__ == "__main__":
    main()