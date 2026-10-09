import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

API_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
DEFAULT_MODEL = "elevenlabs/scribe-v2"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
REQUEST_TIMEOUT = 300
REENCODE_BITRATE = "24k"
CHUNK_TARGET_BYTES = 24 * 1024 * 1024

FORMAT_BY_EXTENSION = {
    ".mp3": "mp3",
    ".wav": "wav",
    ".m4a": "m4a",
    ".flac": "flac",
    ".ogg": "ogg",
    ".oga": "ogg",
    ".opus": "opus",
    ".webm": "webm",
    ".aac": "aac",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe a local audio file or public URL with ElevenLabs Scribe v2 "
        "through OpenRouter, with speaker diarization."
    )
    parser.add_argument("input", help="local audio file path or public http(s) URL")
    parser.add_argument(
        "-o",
        "--output",
        help="path for the transcript text file (default: input name with .txt)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="expected number of speakers; improves diarization accuracy",
    )
    parser.add_argument("--language", help="two-letter language hint, e.g. 'en'")
    parser.add_argument(
        "--tag-audio-events",
        action="store_true",
        help="tag non-speech events such as (laughter)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="also save the raw verbose_json response",
    )
    parser.add_argument(
        "--format",
        help="audio format override for URL input (default: inferred from the URL)",
    )
    args = parser.parse_args()
    if args.num_speakers is not None and args.num_speakers < 1:
        parser.error("--num-speakers must be at least 1")
    return args


def is_url(value):
    return urlparse(value).scheme in ("http", "https")


def infer_format(url):
    suffix = Path(urlparse(url).path).suffix.lower()
    return FORMAT_BY_EXTENSION.get(suffix)


def build_provider_options(args):
    options = {"diarize": True}
    if args.num_speakers is not None:
        options["num_speakers"] = args.num_speakers
    if args.tag_audio_events:
        options["tag_audio_events"] = True
    return {"options": {"elevenlabs": options}}


def api_error_message(response):
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or "no response body"
    error = payload.get("error")
    if isinstance(error, dict):
        return error.get("message") or json.dumps(error)
    if error:
        return str(error)
    return response.text.strip()[:500] or "no error message"


def handle_response(response):
    if not response.ok:
        sys.exit(
            f"error: OpenRouter returned HTTP {response.status_code}: "
            f"{api_error_message(response)}"
        )
    try:
        return response.json()
    except ValueError:
        sys.exit(
            "error: could not parse OpenRouter response as JSON: "
            f"{response.text.strip()[:500]}"
        )


def post_transcription(path, args, api_key, provider):
    data = {
        "model": args.model,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "word",
        "provider": json.dumps(provider),
    }
    if args.language:
        data["language"] = args.language
    with path.open("rb") as handle:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            data=data,
            files={"file": (path.name, handle)},
            timeout=REQUEST_TIMEOUT,
        )
    return handle_response(response)


def require_ffmpeg(path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit(
            f"error: {path} is over the 25 MB upload cap and ffmpeg is required "
            "to shrink it; install ffmpeg (https://ffmpeg.org) or host the file "
            "at a public URL and pass that URL"
        )
    return ffmpeg


def run_ffmpeg(command):
    return subprocess.run(command, capture_output=True, text=True)


def reencode_for_upload(ffmpeg, path, workdir):
    target = workdir / f"{path.stem}.shrink.ogg"
    result = run_ffmpeg(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libopus",
            "-b:a",
            REENCODE_BITRATE,
            str(target),
        ]
    )
    if result.returncode != 0 or not target.is_file():
        return None, result.stderr.strip()
    return target, ""


def probe_duration(ffmpeg, path):
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = run_ffmpeg(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ]
        )
        try:
            return float(result.stdout.strip())
        except ValueError:
            pass
    result = run_ffmpeg([ffmpeg, "-hide_banner", "-i", str(path)])
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if match:
        hours, minutes, seconds = match.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return None


def split_for_upload(ffmpeg, path, workdir, target_bytes=None):
    if target_bytes is None:
        target_bytes = CHUNK_TARGET_BYTES
    if target_bytes < 1:
        sys.exit(f"error: could not split {path} into parts under the 25 MB cap")
    duration = probe_duration(ffmpeg, path)
    size = path.stat().st_size
    if duration:
        segment_seconds = max(1, int(duration * target_bytes / size * 0.9))
    else:
        segment_seconds = 600
    outdir = Path(tempfile.mkdtemp(dir=workdir))
    pattern = outdir / f"part_%03d{path.suffix}"
    result = run_ffmpeg(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-c",
            "copy",
            "-reset_timestamps",
            "1",
            str(pattern),
        ]
    )
    parts = sorted(outdir.glob(f"part_*{path.suffix}"))
    if not parts:
        sys.exit(
            f"error: ffmpeg could not split {path}: {result.stderr.strip()[:500]}"
        )
    resolved = []
    for part in parts:
        if part.stat().st_size > MAX_UPLOAD_BYTES:
            resolved.extend(split_for_upload(ffmpeg, part, workdir, target_bytes // 2))
        else:
            resolved.append(part)
    return resolved


def prepare_upload_parts(ffmpeg, path, workdir):
    compact, error = reencode_for_upload(ffmpeg, path, workdir)
    if compact is None:
        print(
            f"warning: could not re-encode {path.name} ({error[:200]}); "
            "splitting the original instead",
            file=sys.stderr,
        )
        candidate = path
    else:
        candidate = compact
        if candidate.stat().st_size <= MAX_UPLOAD_BYTES:
            size_mb = candidate.stat().st_size / 1024 / 1024
            print(
                f"note: re-encoded {path.name} to mono {REENCODE_BITRATE} Opus "
                f"({size_mb:.1f} MB) to fit the 25 MB upload cap",
                file=sys.stderr,
            )
            return [candidate]
    parts = split_for_upload(ffmpeg, candidate, workdir)
    print(
        f"note: {candidate.name} still exceeds the 25 MB cap; "
        f"using {len(parts)} parts",
        file=sys.stderr,
    )
    return parts


def transcribe_local_file(path, args, api_key, provider):
    if not path.is_file():
        sys.exit(f"error: file not found: {path}")
    if path.stat().st_size <= MAX_UPLOAD_BYTES:
        return [post_transcription(path, args, api_key, provider)]
    ffmpeg = require_ffmpeg(path)
    with tempfile.TemporaryDirectory(prefix="openrouter-transcribe-") as tmp:
        parts = prepare_upload_parts(ffmpeg, path, Path(tmp))
        return [post_transcription(part, args, api_key, provider) for part in parts]


def transcribe_url(args, api_key, provider):
    input_audio = {"url": args.input}
    audio_format = args.format or infer_format(args.input)
    if audio_format:
        input_audio["format"] = audio_format
    body = {
        "model": args.model,
        "input_audio": input_audio,
        "response_format": "verbose_json",
        "timestamp_granularities": ["word"],
        "provider": provider,
    }
    if args.language:
        body["language"] = args.language
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=REQUEST_TIMEOUT,
    )
    return handle_response(response)


def join_words(words):
    return " ".join(" ".join(w.get("word", "") for w in words).split())


def speaker_label(entry):
    label = entry.get("speaker_label")
    if label:
        return label
    speaker = entry.get("speaker")
    if speaker is not None:
        return f"speaker_{speaker}"
    return "speaker_unknown"


def format_transcript(result):
    words = result.get("words") or []
    if not words:
        return (result.get("text") or "").strip() + "\n"
    lines = []
    current_label = None
    current_words = []
    for entry in words:
        label = speaker_label(entry)
        if current_words and label != current_label:
            lines.append(f"{current_label}: {join_words(current_words)}")
            current_words = []
        current_label = label
        current_words.append(entry)
    if current_words:
        lines.append(f"{current_label}: {join_words(current_words)}")
    return "\n".join(lines) + "\n"


def format_results(results):
    if len(results) == 1:
        return format_transcript(results[0])
    blocks = []
    for index, result in enumerate(results, start=1):
        blocks.append(f"[part {index}]\n{format_transcript(result).rstrip()}")
    return "\n\n".join(blocks) + "\n"


def default_output_path(input_value):
    if is_url(input_value):
        name = Path(urlparse(input_value).path).name or "transcript"
        return Path(Path(name).stem + ".txt")
    return Path(input_value).with_suffix(".txt")


def usage_summary(results):
    parts = []
    language = next((r.get("language") for r in results if r.get("language")), None)
    if language:
        parts.append(f"detected language: {language}")
    seconds = [
        r["usage"]["seconds"]
        for r in results
        if isinstance(r.get("usage"), dict) and r["usage"].get("seconds") is not None
    ]
    if seconds:
        parts.append(f"audio: {sum(seconds):.1f}s")
    costs = [
        r["usage"]["cost"]
        for r in results
        if isinstance(r.get("usage"), dict) and r["usage"].get("cost") is not None
    ]
    if costs:
        parts.append(f"cost: ${sum(costs):.6f}")
    return ", ".join(parts)


def load_env_file():
    if load_dotenv is None:
        return
    for env_path in (Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"):
        if env_path.is_file():
            load_dotenv(env_path)
            return


def main():
    load_env_file()
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit(
            "error: set the OPENROUTER_API_KEY environment variable "
            "or add it to a .env file"
        )

    provider = build_provider_options(args)
    if is_url(args.input):
        results = [transcribe_url(args, api_key, provider)]
    else:
        results = transcribe_local_file(Path(args.input), args, api_key, provider)

    output_path = Path(args.output) if args.output else default_output_path(args.input)
    output_path.write_text(format_results(results), encoding="utf-8")
    print(f"wrote {output_path}")

    if args.json:
        json_path = output_path.with_suffix(".json")
        payload = results[0] if len(results) == 1 else {"parts": results}
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {json_path}")

    summary = usage_summary(results)
    if summary:
        print(summary)


if __name__ == "__main__":
    main()
