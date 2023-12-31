import pretty_midi
import torch


def read_note_sequence(midi_file):
    """
    Load MIDI file into note sequence.

    Parameters
    ----------
    midi_file : str
        Path to MIDI file.

    Returns
    -------
    note_seq: (torch.tensor) in the shape of (n, 4), where n is the number of notes, and 4 is the number of features including (pitch, onset, offset, velocity). The note sequence is sorted by onset time.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = []
    for instrument in midi_data.instruments:
        notes.extend(instrument.notes)
    notes = sorted(notes, key=lambda x: x.start)
    return torch.tensor(
        [
            [note.pitch, note.start, note.end - note.start, note.velocity]
            for note in notes
        ]
    )
