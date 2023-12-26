SAMPLE_RATE = 16000
N_NOTES = 88  # Number of notes of piano
BEGIN_NOTE = 21  # MIDI note of A0, the lowest note of a piano.
SEGMENT_LEN = 10.0  # Training segment duration
MAX_VELO = 128

# MIDI key signature information
# fmt: off
MIDI_KEY_ORDER = {
    0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#m", 8: "G#m", 9: "D#m", 10: "Bbm", 11: "Fm", 12: "Cm",
    -11: "Gm", -10: "Dm", -9: "Am", -8: "Em", -7: "Bm", -6: "F#m", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F",
}
NATURAL_KEY_ORDER = [
    "C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
    "Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm",
]
# NAME_TO_NSHARPS is almost the inverse of MIDI_KEY_ORDER ... except that it does not distinguish major and minor keys
NAME_TO_NSHARPS = {
    "C": 0, "Db": -5, "D": 2, "Eb": -3, "E": 4, "F": -1, "F#": 6, "G": 1, "Ab": -4, "A": 3, "Bb": -2, "B": 5,
    "Cm": -3, "C#m": 4, "Dm": 1, "D#m": 6, "Em": 3, "Fm": -2, "F#m": 5, "Gm": 2, "G#m": -3, "Am": 4, "Bbm": 1, "Bm": 6,
}
# fmt: on
_key_to_natural_idx = {name: idx for idx, name in enumerate(NATURAL_KEY_ORDER)}
MIDI_IDX_TO_NATURAL_IDX = {
    midx: _key_to_natural_idx[name] for midx, name in MIDI_KEY_ORDER.items()
}
