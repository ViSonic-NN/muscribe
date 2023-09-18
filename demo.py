import time

import matplotlib.pyplot as plt
import mir_eval
import numpy as np
import pretty_midi as pm
import torch

from audio_midi import PianoTranscription, load_audio, sample_rate
from midi_score import (
    RNNHandPartProcessor,
    RNNJointBeatProcessor,
    RNNJointQuantisationProcessor,
    RNNKeySignatureProcessor,
    RNNTimeSignatureProcessor,
)


def get_piano_roll(midi_file, start_time, end_time):
    pr = np.zeros((128, int((end_time - start_time) * 100)))
    for instrument in pm.PrettyMIDI(midi_file).instruments:
        for note in instrument.notes:
            if note.start >= end_time or note.end <= start_time:
                continue
            start = int((note.start - start_time) * 100)
            end = int((note.end - start_time) * 100)
            pr[note.pitch, start:end] = 1
    return pr


def plot_beat_tracking(midi_recording, beats_pr, beats_gt):
    start_time, end_time = 0, beats_gt.max() + 5
    beats_pr_seg = beats_pr[
        np.logical_and(beats_pr >= start_time, beats_pr <= end_time)
    ]
    beats_gt_seg = beats_gt[
        np.logical_and(beats_gt >= start_time, beats_gt <= end_time)
    ]
    pr_seg = get_piano_roll(midi_recording, start_time, end_time)

    plt.figure(figsize=(20, 5))
    plt.imshow(pr_seg, aspect="auto", origin="lower", cmap="gray")
    for b in beats_pr_seg:
        plt.axvline(x=(b - start_time) * 100, ymin=0.5, ymax=1, color="g")
    for b in beats_gt_seg:
        plt.axvline(x=(b - start_time) * 100, ymin=0, ymax=0.5, color="r")
    plt.title("Green for predicted beats, Red for ground truth beats.")
    plt.savefig("beat_tracking.png")


def get_midi(audio_path, output_midi_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (audio, _) = load_audio(audio_path, sr=sample_rate, mono=True)
    transcriptor = PianoTranscription(device=device, checkpoint_path=None)
    transcribe_time = time.time()
    transcriptor.transcribe(audio, output_midi_path)
    print("Transcribe time: {:.3f} s".format(time.time() - transcribe_time))


# Get one MIDI recording from the A_MAPS dataset
audio_input = "ハートグレイズ.mp3"
midi_output = "ハートグレイズ.midi"
get_midi(audio_input, midi_output)

# Create processor models
beat_pro = RNNJointBeatProcessor()
quant_pro = RNNJointQuantisationProcessor()
part_pro = RNNHandPartProcessor()
time_sig_pro = RNNTimeSignatureProcessor()
key_sig_pro = RNNKeySignatureProcessor()

beats_pr = beat_pro.process(midi_output)
beats_gt = pm.PrettyMIDI(midi_output).get_beats()

# F-measure for beat tracking
f1 = mir_eval.beat.f_measure(
    mir_eval.beat.trim_beats(beats_gt), mir_eval.beat.trim_beats(beats_pr)
)
print("First 10 predicted beats:")
print(beats_pr[:10])
print("First 10 target beats:")
print(beats_gt[:10])
print("F1 score for beat tracking: {}".format(f1))

plot_beat_tracking(midi_output, beats_pr, beats_gt)

# Process the MIDI recording to the beat predictions
beats, onset_positions, note_values = quant_pro.process(midi_output)
print("onset positions and note values in number of beats:")
print(onset_positions[:20])
print(note_values[:20])

# Predict hand part for each note in the MIDI recording
hand_parts = part_pro.process(midi_output)
print(hand_parts[:20])

time_signature_changes = time_sig_pro.process(midi_output)
key_signature_changes = key_sig_pro.process(midi_output)
print("Time signature changes:")
print(time_signature_changes)
print("Key signature changes:")
print(key_signature_changes)
