# recording-transcript

Transcribe audio with speaker diarization using [ElevenLabs Scribe v2](https://openrouter.ai/elevenlabs/scribe-v2) through [OpenRouter](https://openrouter.ai), without a separate ElevenLabs account.

`transcribe.py` accepts a local audio file or a public URL and writes a plain-text transcript with speaker turns.

## Requirements

- Python 3.8+
- An [OpenRouter API key](https://openrouter.ai/settings/keys)
- `ffmpeg` in `PATH` (only needed for local files over 25 MB)

## Setup

```bash
pip install -r requirements.txt
```

Set your API key:

```powershell
# PowerShell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
```

```bash
# Bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Or put it in a `.env` file in your working directory or next to `transcribe.py`:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

The `.env` file is loaded at startup and never overrides an environment variable that is already set.

## Usage

```bash
python transcribe.py <audio_file_or_url> [options]
```

### Examples

Transcribe a local file:

```bash
python transcribe.py meeting.mp3
```

Writes `meeting.txt` next to the input.

Transcribe a meeting with a known number of speakers and audio-event tags:

```bash
python transcribe.py interview.wav --num-speakers 3 --tag-audio-events
```

Transcribe a public URL (bypasses the 25 MB upload cap):

```bash
python transcribe.py https://example.com/call.wav --language en
```

Transcribe a long recording; local files over the cap are shrunk and split automatically:

```bash
python transcribe.py conference.wav --num-speakers 4
```

Save the raw API response alongside the transcript:

```bash
python transcribe.py recording.m4a --json -o notes/recording.txt
```

### Options

| Option | Description |
| --- | --- |
| `input` | Local audio file path or public `http(s)` URL (required) |
| `-o`, `--output` | Path for the transcript text file (default: input name with `.txt`) |
| `--model` | OpenRouter model ID (default: `elevenlabs/scribe-v2`) |
| `--num-speakers` | Expected number of speakers; improves diarization accuracy |
| `--language` | Two-letter language hint, e.g. `en` |
| `--tag-audio-events` | Tag non-speech events such as `(laughter)` |
| `--json` | Also save the raw `verbose_json` response (split recordings as `{"parts": [...]}`) |
| `--format` | Audio format override for URL input (default: inferred from the URL) |

## Output

The text file contains consecutive words grouped into speaker turns:

```
speaker_0: Hello there.
speaker_1: Hi! (laughter)
```

The console reports the detected language and API usage (audio seconds and cost):

```
wrote meeting.txt
detected language: eng, audio: 12.3s, cost: $0.000678
```

With `--json`, the full response (word timestamps, confidence, speakers, usage) is written to a `.json` file next to the transcript. Split recordings are saved as `{"parts": [...]}`.

## Files over 25 MB

Multipart uploads are capped at 25 MB. For local files above the cap, the script uses `ffmpeg` automatically:

1. Re-encode to mono 16 kHz Opus at 24 kbps. Even 2+ hours of speech fits in 25 MB at that bitrate.
2. If the re-encoded file still exceeds 25 MB (roughly 2.5+ hours of audio), it is split into parts that stay under the cap and each part is transcribed separately, then the transcripts are merged.

If ffmpeg is not installed, the script exits with an error; install it or host the file at a public URL instead. Temporary files are deleted automatically.

Split recordings get part markers in the transcript:

```
[part 1]
speaker_0: Hello there.

[part 2]
speaker_0: ...
```

Note: diarization labels are assigned per request, so `speaker_0` in part 1 is not necessarily the same person as `speaker_0` in part 2. Avoid splitting multi-speaker recordings when speaker identity across parts matters.

## Notes

- Diarization is always enabled via `provider.options.elevenlabs.diarize`.
- `--num-speakers` maps to `num_speakers`; it cannot be combined with `diarization_threshold`, which this script does not expose.
- Uploads are capped at 25 MB (roughly 26 minutes of 128 kbps MP3, 52 minutes at 64 kbps, or 2+ hours of 24 kbps Opus; uncompressed WAV fills the cap in about 13 minutes at 16 kHz mono). Local files over the cap are automatically re-encoded and, if necessary, split (see [Files over 25 MB](#files-over-25-mb)); alternatively, host the file at a public URL and pass that URL, which bypasses the cap.
- Very long files may hit the upstream 60-second request timeout. Split long recordings.
- Detected languages are returned as ISO 639-3 codes (e.g. `eng`), while `--language` accepts a two-letter hint.
- Billing is per second of audio; `usage.cost` shows the cost of each request.
