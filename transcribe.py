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
# Maximum length of a split part. This is a cap for splitting, not a trigger:
# files under the upload cap are tried as a single request first.
MAX_PART_SECONDS = 1800
# Don't split short files on transient errors; splitting is unlikely to help.
MIN_SPLIT_SECONDS = 600
DESIRED_CODEC = "opus"
DESIRED_CHANNELS = 1
DESIRED_BITRATE = 24_000
DESIRED_SUFFIXES = {".ogg", ".oga", ".opus", ".webm"}
# Allow for container overhead and small encoder differences when deciding
# whether a file is already close enough to the desired 24 kbps Opus format.
BITRATE_TOLERANCE = 1.25
# HTTP errors that may be caused by a request being too large or too slow.
SPLIT_RETRY_STATUS_CODES = {413, 500, 502, 504, 524}


class TranscribeError(Exception):
    pass


class ProviderError(TranscribeError):
    pass


class HttpError(TranscribeError):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


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
        raise HttpError(
            response.status_code,
            f"OpenRouter returned HTTP {response.status_code}: "
            f"{api_error_message(response)}",
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


def parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe_audio_info(ffmpeg, path):
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = run_ffmpeg(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,channels,sample_rate,bit_rate",
                "-show_entries",
                "format=duration,size,bit_rate",
                "-of",
                "json",
                str(path),
            ]
        )
        try:
            payload = json.loads(result.stdout)
        except ValueError:
            return None
        streams = payload.get("streams") or []
        if not streams:
            return None
        stream = streams[0]
        fmt = payload.get("format") or {}
        return {
            "codec_name": stream.get("codec_name"),
            "channels": parse_int(stream.get("channels")),
            "sample_rate": parse_int(stream.get("sample_rate")),
            "bit_rate": parse_int(stream.get("bit_rate"))
            or parse_int(fmt.get("bit_rate")),
            "duration": parse_float(fmt.get("duration")),
            "size": parse_int(fmt.get("size")),
        }

    result = run_ffmpeg([ffmpeg, "-hide_banner", "-i", str(path)])
    text = result.stderr
    audio = re.search(r"Audio:\s*([^,\n]+),\s*(\d+)\s*Hz,\s*([^,\n]+)", text)
    if not audio:
        return None
    codec_name = audio.group(1).strip().split()[0]
    channel_layout = audio.group(3).strip().lower()
    if channel_layout.startswith("mono"):
        channels = 1
    elif channel_layout.startswith("stereo"):
        channels = 2
    else:
        channels = None
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    duration = None
    if duration_match:
        hours, minutes, seconds = duration_match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    bitrate_match = re.search(r"bitrate:\s*(\d+)\s*kb/s", text)
    bit_rate = int(bitrate_match.group(1)) * 1000 if bitrate_match else None
    return {
        "codec_name": codec_name,
        "channels": channels,
        "sample_rate": parse_int(audio.group(2)),
        "bit_rate": bit_rate,
        "duration": duration,
        "size": path.stat().st_size,
    }


def is_desired_format(info, size, suffix):
    if not info:
        return False
    if suffix.lower() not in DESIRED_SUFFIXES:
        return False
    if info.get("codec_name") != DESIRED_CODEC:
        return False
    if info.get("channels") != DESIRED_CHANNELS:
        return False
    # Opus streams always report 48 kHz in ffprobe, so sample_rate is not a
    # reliable indicator of the encoder's input rate.
    if size <= 256 * 1024:
        # Re-encoding a tiny file saves little and adds generation loss.
        return True
    bit_rate = info.get("bit_rate")
    if bit_rate is None:
        duration = info.get("duration")
        if duration:
            bit_rate = size * 8 / duration
    if bit_rate is None:
        return size <= MAX_UPLOAD_BYTES
    return bit_rate <= DESIRED_BITRATE * BITRATE_TOLERANCE


def convert_to_opus(ffmpeg, path, workdir):
    output = workdir / f"{path.stem}.ogg"
    result = run_ffmpeg(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-stats",
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
            "24k",
            str(output),
        ]
    )
    if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        error = result.stderr.strip()
        if error:
            print(
                f"warning: ffmpeg conversion failed for {path.name}: "
                f"{error[:500]}",
                file=sys.stderr,
            )
        return None
    return output


def format_size(num_bytes):
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def is_retryable_error(exc):
    if isinstance(exc, ProviderError):
        return True
    if isinstance(exc, HttpError):
        return exc.status_code in SPLIT_RETRY_STATUS_CODES
    return isinstance(exc, requests.Timeout)


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
    segment_seconds = min(segment_seconds, MAX_PART_SECONDS)
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
        except (ProviderError, HttpError, requests.Timeout) as exc:
            if not is_retryable_error(exc):
                raise
            duration = probe_duration(ffmpeg, part) if ffmpeg else None
            if depth >= 3 or not duration or duration <= MIN_SPLIT_SECONDS:
                raise
            print(
                f"note: {part.name} failed ({exc}); "
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
                "to convert or split it; install ffmpeg (https://ffmpeg.org) "
                "or host the file at a public URL and pass that URL"
            )
        print(
            "note: ffmpeg not found; uploading the original file without "
            "conversion or splitting",
            file=sys.stderr,
        )
        return [post_transcription(path, args, api_key, provider)]
    with tempfile.TemporaryDirectory(prefix="openrouter-transcribe-") as tmp:
        workdir = Path(tmp)
        audio_path = path
        info = probe_audio_info(ffmpeg, path)
        if not is_desired_format(info, size, path.suffix):
            print(
                f"note: converting {path.name} to 24 kbps Opus...",
                file=sys.stderr,
            )
            converted = convert_to_opus(ffmpeg, path, workdir)
            if converted is None:
                print(
                    f"warning: could not convert {path.name} to 24 kbps Opus; "
                    "using the original file",
                    file=sys.stderr,
                )
            else:
                converted_size = converted.stat().st_size
                if converted_size >= size:
                    print(
                        f"note: conversion of {path.name} did not reduce size "
                        f"({format_size(size)} -> {format_size(converted_size)}); "
                        "using the original file",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"note: converted {path.name} to 24 kbps Opus "
                        f"({format_size(size)} -> {format_size(converted_size)})",
                        file=sys.stderr,
                    )
                    audio_path = converted
        size = audio_path.stat().st_size
        duration = probe_duration(ffmpeg, audio_path)
        if size <= MAX_UPLOAD_BYTES:
            try:
                return [post_transcription(audio_path, args, api_key, provider)]
            except (TranscribeError, requests.RequestException) as exc:
                if not is_retryable_error(exc):
                    raise
                if duration is None or duration <= MIN_SPLIT_SECONDS:
                    raise
                print(
                    f"note: single request for {path.name} failed ({exc}); "
                    "splitting it into smaller parts and retrying",
                    file=sys.stderr,
                )
                parts = split_for_upload(
                    ffmpeg, audio_path, workdir, max(1, size // 2)
                )
                print(
                    f"note: split into {len(parts)} parts; "
                    "speaker labels restart per part",
                    file=sys.stderr,
                )
                return transcribe_parts(
                    parts, args, api_key, provider, ffmpeg, workdir
                )
        parts = split_for_upload(ffmpeg, audio_path, workdir)
        print(
            f"note: {path.name} split into {len(parts)} parts "
            f"(25 MB / {MAX_PART_SECONDS // 60}-minute limit per part); "
            "speaker labels restart per part",
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
