# assets/audio/

Drop the track you want to animate to in here. The contents are gitignored:
music you supply is licensed to you, not to this repository.

`facade-scan animate` reads whatever you point it at:

```bash
facade-scan animate --audio assets/audio/most-wonderful-time.mp3 ...
```

Any format ffmpeg can decode works. The analysis only ever reads the audio —
it is decoded to mono PCM in a temporary file and never rewritten.
