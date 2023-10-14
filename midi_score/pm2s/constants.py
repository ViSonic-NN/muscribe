# ========== data representation related constants ==========
RESOLUTION = 0.01  # quantization resolution: 0.01s = 10ms

# =========== key signature definitions ==========

# key in sharps in MIDI
# fmt: off
MIDI_KEY_TO_KEY_NAME = {
    0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#m", 8: "G#m", 9: "D#m", 10: "Bbm", 11: "Fm", 12: "Cm",
    -11: "Gm", -10: "Dm", -9: "Am", -8: "Em", -7: "Bm", -6: "F#m", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F",
}
KEYS_BY_NAME = [
    "C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
    "Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm",
]
# fmt: on

# ignore minor keys in key signature prediction!
KEY_VOCAB_SIZE = len(KEYS_BY_NAME) // 2
