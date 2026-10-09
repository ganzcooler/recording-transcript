# recording-transcript

Transcribe audio with speaker diarization using [ElevenLabs Scribe v2](https://openrouter.ai/elevenlabs/scribe-v2) through [OpenRouter](https://openrouter.ai), without a separate ElevenLabs account.

`transcribe.py` accepts a local audio file, a directory of audio files, or a public URL and writes a plain-text transcript with speaker turns.

## Requirements

- Python 3.8+
- An [OpenRouter API key](https://openrouter.ai/settings/keys)
- `ffmpeg` in `PATH` (needed for local files over 25 MB or longer than 30 minutes)

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

Transcribe a long recording; local files over 25 MB or 30 minutes are split automatically:

```bash
python transcribe.py conference.wav --num-speakers 4
```

Save the raw API response alongside the transcript:

```bash
python transcribe.py recording.m4a --json -o notes/recording.txt
```

Transcribe every audio file in a folder:

```bash
python transcribe.py input --skip-existing -o transcripts
```

### Language

Without `--language`, Scribe v2 detects the language automatically and can even handle multiple languages in one file. When the recording is known to be one language, a hint can improve accuracy for short, noisy, or accented audio:

```bash
python transcribe.py interview-de.mp3 --language de
```

If your recordings are in German, pass `--language de` unless the audio mixes languages — a known language can improve transcription performance, especially with names, jargon, and accented speech:

```bash
python transcribe.py 261006_woll1.ogg --language de --num-speakers 4
```

`--language` accepts ISO 639-1 (`de`, `en`) or ISO 639-3 (`deu`, `eng`) codes; the detected language in responses is always ISO 639-3 (e.g. `deu`).

### Options

| Option | Description |
| --- | --- |
| `input` | Local audio file, directory of audio files, or public `http(s)` URL (required) |
| `-o`, `--output` | Path for the transcript text file (default: input name with `.txt`); in batch mode, the output directory |
| `--model` | OpenRouter model ID (default: `elevenlabs/scribe-v2`) |
| `--num-speakers` | Expected number of speakers; improves diarization accuracy |
| `--language` | Language hint, ISO 639-1 or 639-3 (`de` or `deu`, `en` or `eng`); can improve accuracy when the language is known |
| `--tag-audio-events` | Tag non-speech events such as `(laughter)` |
| `--json` | Also save the raw `verbose_json` response (split recordings as `{"parts": [...]}`) |
| `--skip-existing` | Skip inputs whose transcript already exists (batch mode) |
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

## Batch mode

When the input is a directory, every supported audio file in it (`mp3`, `wav`, `m4a`, `flac`, `ogg`, `oga`, `opus`, `webm`, `aac`; top level only) is transcribed in alphabetical order:

```bash
python transcribe.py input
```

Transcripts are written next to each input file as `<name>.txt`, or into the directory given with `-o` (created if missing). Add `--skip-existing` to leave inputs that already have a transcript untouched, which is useful to resume an interrupted batch.

A failing file (API error, missing ffmpeg, ...) does not stop the batch: the error is printed, the remaining files are still processed, and a summary is printed at the end. The process exits with code 1 if at least one file failed.

```
[1/3] call-1.mp3
wrote input/call-1.txt
detected language: eng, audio: 1800.0s, cost: $0.048000
[2/3] call-2.mp3
skipped call-2.mp3 (transcript exists)
[3/3] call-3.mp3
error: call-3.mp3: OpenRouter returned HTTP 500: ...
batch: 1 processed, 1 failed, 1 skipped, audio: 1800.0s, cost: $0.048000
```

`--json` saves a `.json` next to each transcript. Files over 25 MB or 30 minutes are still split; see [Long files and the 25 MB cap](#long-files-and-the-25-mb-cap).

## Long files and the 25 MB cap

Multipart uploads are capped at 25 MB, and the upstream provider times out on very long requests, so local files over 25 MB or longer than 30 minutes are split with `ffmpeg` automatically. Splitting uses stream copy (`-c copy`), so there is no re-encoding and no quality loss; each part is uploaded and transcribed separately, then the transcripts are merged.

Pre-compressing large sources keeps uploads small and fast: mono 16 kHz Opus at 24 kbps is a good speech setting (~10.5 MB per hour):

```bash
ffmpeg -i recording.m4a -vn -ac 1 -ar 16000 -c:a libopus -b:a 24k recording.ogg
```

If ffmpeg is not installed, files over 25 MB fail with an error; install ffmpeg or host the file at a public URL instead. Temporary files are deleted automatically.

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
- Uploads are capped at 25 MB (roughly 26 minutes of 128 kbps MP3, 52 minutes at 64 kbps, or 2+ hours of 24 kbps Opus; uncompressed WAV fills the cap in about 13 minutes at 16 kHz mono). Local files over the cap or longer than 30 minutes are split automatically (see [Long files and the 25 MB cap](#long-files-and-the-25-mb-cap)); alternatively, host the file at a public URL and pass that URL, which bypasses the cap.
- Very long requests can hit the upstream 60-second timeout; the script splits long files into parts automatically (30-minute limit per part).
- Detected languages are returned as ISO 639-3 codes (e.g. `deu`, `eng`), and `--language` accepts either ISO 639-1 or ISO 639-3 hints (`de` or `deu`, `en` or `eng`). A hint can improve accuracy when the recording is one known language; for audio that mixes languages, leave it unset and let Scribe detect each one.
- Billing is per second of audio; `usage.cost` shows the cost of each request.
