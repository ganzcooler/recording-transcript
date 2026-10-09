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
CHUNK_TARGET_BYTES = 24 * 1024 * 1024
MAX_SEGMENT_SECONDS = 1800


class TranscribeError(Exception):
    pass


class ProviderError(TranscribeError):
    pass

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
        description="Transcribe a local audio file, a directory of audio files, or a "
        "public URL with ElevenLabs Scribe v2 through OpenRouter, with speaker "
        "diarization."
    )
    parser.add_argument(
        "input", help="local audio file, directory of audio files, or public http(s) URL"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="path for the transcript text file (default: input name with .txt); "
        "in batch mode, the output directory",
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
    parser.add_argument(
        "--language", help="language hint, ISO 639-1 or 639-3 (e.g. 'de' or 'deu')"
    )
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
        "--skip-existing",
        action="store_true",
        help="skip inputs whose transcript already exists (batch mode)",
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
        raise TranscribeError(
            f"OpenRouter returned HTTP {response.status_code}: "
            f"{api_error_message(response)}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise TranscribeError(
            "could not parse OpenRouter response as JSON: "
            f"{response.text.strip()[:500]}"
        ) from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise ProviderError(
            f"OpenRouter returned an error in the response body: "
            f"{api_error_message(response)}"
        )
    return payload


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


def run_ffmpeg(command):
    return subprocess.run(command, capture_output=True, text=True)


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
        raise TranscribeError(f"could not split {path} into parts under the 25 MB cap")
    duration = probe_duration(ffmpeg, path)
    size = path.stat().st_size
    if duration:
        segment_seconds = max(1, int(duration * target_bytes / size * 0.9))
    else:
        segment_seconds = 600
    segment_seconds = min(segment_seconds, MAX_SEGMENT_SECONDS)
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
    if len(parts) > 1:
        parts = [part for part in parts if part.stat().st_size >= 1024]
    if not parts:
        raise TranscribeError(
            f"ffmpeg could not split {path}: {result.stderr.strip()[:500]}"
        )
    resolved = []
    for part in parts:
        if part.stat().st_size > MAX_UPLOAD_BYTES:
            resolved.extend(split_for_upload(ffmpeg, part, workdir, target_bytes // 2))
        else:
            resolved.append(part)
    return resolved


def transcribe_parts(parts, args, api_key, provider, ffmpeg, workdir, depth=0):
    results = []
    for part in parts:
        try:
            results.append(post_transcription(part, args, api_key, provider))
            continue
        except ProviderError as exc:
            duration = probe_duration(ffmpeg, part) if ffmpeg else None
            if depth >= 3 or not duration or duration <= 600:
                raise
            print(
                f"note: {part.name} hit a provider error ({exc}); "
                "splitting it into smaller parts and retrying",
                file=sys.stderr,
            )
        smaller = split_for_upload(
            ffmpeg, part, workdir, max(1, part.stat().st_size // 2)
        )
        results.extend(
            transcribe_parts(
                smaller, args, api_key, provider, ffmpeg, workdir, depth + 1
            )
        )
    return results


def transcribe_local_file(path, args, api_key, provider):
    if not path.is_file():
        raise TranscribeError(f"file not found: {path}")
    size = path.stat().st_size
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        if size > MAX_UPLOAD_BYTES:
            raise TranscribeError(
                f"{path} is over the 25 MB upload cap and ffmpeg is required "
                "to split it; install ffmpeg (https://ffmpeg.org) or host the file "
                "at a public URL and pass that URL"
            )
        return [post_transcription(path, args, api_key, provider)]
    duration = probe_duration(ffmpeg, path)
    if size <= MAX_UPLOAD_BYTES and (duration is None or duration <= MAX_SEGMENT_SECONDS):
        return [post_transcription(path, args, api_key, provider)]
    with tempfile.TemporaryDirectory(prefix="openrouter-transcribe-") as tmp:
        workdir = Path(tmp)
        parts = split_for_upload(ffmpeg, path, workdir)
        print(
            f"note: {path.name} split into {len(parts)} parts "
            f"(25 MB / {MAX_SEGMENT_SECONDS // 60}-minute limit per request)",
            file=sys.stderr,
        )
        return transcribe_parts(parts, args, api_key, provider, ffmpeg, workdir)


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


def collect_inputs(value):
    if is_url(value):
        return [value]
    path = Path(value)
    if path.is_dir():
        items = sorted(
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in FORMAT_BY_EXTENSION
        )
        if not items:
            raise TranscribeError(f"no audio files found in {path}")
        return items
    if not path.is_file():
        raise TranscribeError(f"file not found: {path}")
    return [path]


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


def usage_totals(results):
    seconds = 0.0
    cost = 0.0
    for result in results:
        usage = result.get("usage")
        if isinstance(usage, dict):
            seconds += usage.get("seconds") or 0
            cost += usage.get("cost") or 0
    return seconds, cost


def load_env_file():
    if load_dotenv is None:
        return
    for env_path in (Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"):
        if env_path.is_file():
            load_dotenv(env_path)
            return


def transcribe_and_write(input_value, args, api_key, provider, output_path):
    if is_url(str(input_value)):
        results = [transcribe_url(args, api_key, provider)]
    else:
        results = transcribe_local_file(Path(input_value), args, api_key, provider)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    return results


def main():
    load_env_file()
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit(
            "error: set the OPENROUTER_API_KEY environment variable "
            "or add it to a .env file"
        )

    try:
        inputs = collect_inputs(args.input)
    except TranscribeError as exc:
        sys.exit(f"error: {exc}")

    provider = build_provider_options(args)
    batch = not is_url(args.input) and Path(args.input).is_dir()

    if not batch:
        output_path = (
            Path(args.output) if args.output else default_output_path(args.input)
        )
        try:
            transcribe_and_write(inputs[0], args, api_key, provider, output_path)
        except (TranscribeError, requests.RequestException) as exc:
            sys.exit(f"error: {exc}")
        return

    if args.output:
        output_dir = Path(args.output)
        if output_dir.exists() and not output_dir.is_dir():
            sys.exit(f"error: --output must be a directory in batch mode: {output_dir}")
    else:
        output_dir = Path(args.input)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = failed = skipped = 0
    total_seconds = 0.0
    total_cost = 0.0
    for index, item in enumerate(inputs, start=1):
        print(f"[{index}/{len(inputs)}] {item.name}", flush=True)
        output_path = output_dir / f"{item.stem}.txt"
        if args.skip_existing and output_path.exists():
            skipped += 1
            print(f"skipped {item.name} (transcript exists)")
            continue
        try:
            results = transcribe_and_write(item, args, api_key, provider, output_path)
        except (TranscribeError, requests.RequestException) as exc:
            failed += 1
            print(f"error: {item.name}: {exc}", file=sys.stderr)
            continue
        processed += 1
        seconds, cost = usage_totals(results)
        total_seconds += seconds
        total_cost += cost

    summary_parts = [f"{processed} processed"]
    if failed:
        summary_parts.append(f"{failed} failed")
    if skipped:
        summary_parts.append(f"{skipped} skipped")
    summary = ", ".join(summary_parts)
    if total_seconds:
        summary += f", audio: {total_seconds:.1f}s"
    if total_cost:
        summary += f", cost: ${total_cost:.6f}"
    print(f"batch: {summary}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
