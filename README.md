# recording-transcript

Transcribe audio with speaker diarization using [ElevenLabs Scribe v2](https://openrouter.ai/elevenlabs/scribe-v2) through [OpenRouter](https://openrouter.ai), without a separate ElevenLabs account.

`transcribe.py` accepts a local audio file or a public URL and writes a plain-text transcript with speaker turns.

## Requirements

- Python 3.8+
- An [OpenRouter API key](https://openrouter.ai/settings/keys)

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
| `--json` | Also save the raw `verbose_json` response |
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

With `--json`, the full response (word timestamps, confidence, speakers, usage) is written to a `.json` file next to the transcript.

## Notes

- Diarization is always enabled via `provider.options.elevenlabs.diarize`.
- `--num-speakers` maps to `num_speakers`; it cannot be combined with `diarization_threshold`, which this script does not expose.
- Uploads are capped at 25 MB (about 27 minutes of 128 kbps MP3). For longer recordings, host the file at a public URL and pass that URL; the script switches to `input_audio.url` automatically.
- Very long files may hit the upstream 180-second request timeout. Split long recordings.
- Detected languages are returned as ISO 639-3 codes (e.g. `eng`), while `--language` accepts a two-letter hint.
- Billing is per second of audio; `usage.cost` shows the cost of each request.
