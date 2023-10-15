from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Iterator, NamedTuple, Sequence, TypeVar, cast

import numpy as np
import pymusicxml as mxml

from .pm2s.constants import NAME_TO_NSHARPS

T1 = TypeVar("T1")
T2 = TypeVar("T2")

TIME_SIG_D = 4
NOTES_IN_SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTES_IN_FLATS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def num_to_pitch(pitch_num, current_key: str | None = None):
    # 24 is C1, and we start from 21 = A0.
    pitch_num = int(pitch_num)
    if pitch_num < 21:
        raise ValueError(f"Pitch number {pitch_num} is too low")
    pitch_num -= 12
    octave, note = pitch_num // 12, pitch_num % 12
    n_sharps: int = NAME_TO_NSHARPS.get(current_key, 0)  # type: ignore
    note = (NOTES_IN_SHARPS if n_sharps >= 0 else NOTES_IN_FLATS)[note]
    return mxml.Pitch.from_string(f"{note}{octave}")


def splitby(xs: list[T1], is_delim: Callable[[T1], bool]) -> Iterator[list[T1]]:
    group = []
    for x in xs:
        if not is_delim(x):
            group.append(x)
            continue
        if group:
            yield group
        group = [x]
    yield group


def groupby(xs: Iterable[T1], key: Callable[[T1], T2]) -> defaultdict[T2, list[T1]]:
    ret = defaultdict(list)
    for x in xs:
        ret[key(x)].append(x)
    return ret


class Note(NamedTuple):
    onset: Fraction
    duration: Fraction
    pitch_num: int
    velocity: int
    is_right_hand: bool


class KeyChange(NamedTuple):
    onset: Fraction
    key: str


class BPMChange(NamedTuple):
    onset: Fraction
    bpm: float


class NewMeasure(NamedTuple):
    onset: Fraction
    time_sig_n: int


class MusicXMLBuilder:
    def __init__(self, beats: np.ndarray) -> None:
        self.beats = beats
        self._events = []
        self._make_measure_events()

    @property
    def sorted_events(self):
        # Sort NewMeasure before everything else when they occur at the same time
        self._events = sorted(
            self._events, key=lambda e: (e.onset, not isinstance(e, NewMeasure))
        )
        return self._events

    def add_notes(self, midi_notes: np.ndarray, hand_parts: np.ndarray):
        assert len(hand_parts) == len(midi_notes)
        for (pitch, nstart, ndur, velo), hand in zip(midi_notes, hand_parts):
            note_beat, bstart, blength = self._find_in_beats(nstart)
            noffset = self._to_frac((nstart - bstart) / blength)
            nlength = self._to_frac(ndur / blength)
            is_right_hand = hand == 0
            self._events.append(
                Note(note_beat + noffset, nlength, int(pitch), velo, is_right_hand)
            )
        return self

    def add_key_changes(self, key_changes: list[tuple[float, str]]):
        for time, key in key_changes:
            note_beat, _, _ = self._find_in_beats(time)
            # TODO: check `offset = to_frac((time - bstart) / blength)` close to 0
            self._events.append(KeyChange(note_beat, key))

    def infer_bpm_changes(
        self,
        log_bin_size: float = 0.05,
        event_threshold: float = 0.2,
        diff_size: int = 2,
        window_size: int = 10,
    ):
        def bpm_from_mode(n_samples, bins):
            mode_idx = n_samples.argmax()
            if n_samples[mode_idx] < sum(n_samples) / 2:
                return None
            mode_min, mode_max = bins[mode_idx], bins[mode_idx + 1]
            mode_samples = samples[
                np.logical_and(samples >= mode_min, samples <= mode_max)
            ]
            return float(np.mean(mode_samples))

        beat_times = self.beats[:, 0]
        beat_lens = (beat_times[diff_size:] - beat_times[:-diff_size]) / diff_size
        bpms, last_valid_bpm = [], None
        for idx in range(window_size, len(beat_lens)):
            samples = 60 / beat_lens[idx - window_size : idx]
            bin_min, bin_max = np.log2(samples.min() - 0.01), np.log2(
                samples.max() + 0.01
            )
            bin_width = np.log2(1 + log_bin_size)
            bins = 2 ** np.arange(bin_min, bin_max + bin_width, bin_width)
            n_samples, _ = np.histogram(samples, bins)
            if (current_bpm := bpm_from_mode(n_samples, bins)) is None:
                bpms.append(np.nan)
                continue
            bpms.append(current_bpm)
            if last_valid_bpm is None:
                # Advance the first BPM to the first beat
                self._events.append(BPMChange(Fraction(0, 1), current_bpm))
            elif np.abs(1 - current_bpm / last_valid_bpm) > event_threshold:
                self._events.append(
                    BPMChange(Fraction(idx + diff_size, 1), current_bpm)
                )
            last_valid_bpm = current_bpm
        prefix = [np.nan] * (diff_size + window_size)
        return np.array(prefix + bpms)

    def render(self, output_file: str | Path):
        def get_unique(d: dict, ty: type[T1]) -> T1 | None:
            if len(v := d[ty]) > 1:
                raise ValueError(f"Multiple {ty} in one onset: {v}")
            return v[0] if len(v) == 1 else None

        measures = []
        last_key_sig: str | None = None
        last_time_sig: int | None = None
        for mevents in splitby(self.sorted_events, lambda e: isinstance(e, NewMeasure)):
            mevents = groupby(mevents, type)
            new_measure = get_unique(mevents, NewMeasure)
            assert new_measure is not None
            key_change = get_unique(mevents, KeyChange)
            m_kwargs: dict = {}
            if not measures:
                m_kwargs["clef"] = "treble"
            if key_change and last_key_sig != (key := key_change.key):
                last_key_sig = key
                m_kwargs["key"] = key
            if last_time_sig != (time_sig_n := new_measure.time_sig_n):
                last_time_sig = time_sig_n
                m_kwargs["time_signature"] = time_sig_n, TIME_SIG_D
            notes = groupby(cast(list[Note], mevents[Note]), lambda n: n.is_right_hand)
            # Add any BPM change to the right hand
            rhs_content = _make_measure_notes(
                notes[True], 1, new_measure, last_key_sig, mevents[BPMChange]
            )
            lhs_content = _make_measure_notes(
                notes[False], 2, new_measure, last_key_sig
            )
            measures.append(mxml.Measure([rhs_content, lhs_content], **m_kwargs))
        score = mxml.Score([mxml.Part("Piano", measures)])
        score.export_to_file(str(output_file))

    def _make_measure_events(self):
        # Minus 1 -- start from 0
        beat_counts = (self.beats[:, 1].astype(int) - 1).tolist()
        for idx, beat in enumerate(beat_counts):
            if (beat == 0 and idx > 0) or idx == len(beat_counts) - 1:
                prev_beat = beat_counts[idx - 1]
                prev_start = Fraction(idx - 1 - prev_beat, 1)
                self._events.append(
                    NewMeasure(prev_start, prev_beat + 1)  # Add this 1 beat back
                )

    def _find_in_beats(self, time):
        next_beat = np.searchsorted(self.beats[:, 0], time)
        assert 1 <= next_beat <= len(self.beats)
        bstart = self.beats[next_beat - 1, 0]
        if next_beat == len(self.beats):
            # There's no next beat, but we can extrapolate the length
            # from the previous beat...
            assert next_beat > 1
            blength = bstart - self.beats[next_beat - 2, 0]
        else:
            blength = self.beats[next_beat, 0] - bstart
        return next_beat - 1, bstart, blength

    @staticmethod
    def _to_frac(duration: float, denoms=(1, 2, 4, 8), max_n_pieces=2):
        denoms = np.array(denoms).astype(int)
        numers = np.round(duration * denoms).astype(int)
        errors = np.abs(numers / denoms - duration)
        n_pieces = np.array(
            [len(_into_simple_times(Fraction(n, d))) for n, d in zip(numers, denoms)]
        )
        errors[n_pieces > max_n_pieces] = np.inf
        smallest = int(np.argmin(errors))
        return Fraction(int(numers[smallest]), int(denoms[smallest]))


def _into_simple_times(duration: Fraction) -> list[Fraction]:
    def is_pow_of_2(n: int):
        return n & (n - 1) == 0

    def nearest_pow_of_2(n: int):
        return 2 ** int(np.log2(n))

    if duration <= 0:
        return []
    duration = duration * 4 / TIME_SIG_D
    assert isinstance(duration, Fraction) and is_pow_of_2(duration.denominator)
    durations = []
    while duration > 0:
        next_dur = Fraction(nearest_pow_of_2(duration.numerator), duration.denominator)
        durations.append(next_dur)
        duration -= next_dur
    return durations


def _simple_timed_elems(
    duration: Fraction,
    staff_idx: int,
    pitches: Sequence[mxml.Pitch],
    directions: Sequence[mxml.Direction] = tuple(),
):
    def make_dur_obj(duration: mxml.Duration, ties: str | None):
        nonlocal directions
        # only add directions to the first note
        dir_, directions = directions, []
        if len(pitches) == 0:
            rest = mxml.Rest(duration, directions=dir_)
            rest.voice = rest.staff = staff_idx
            return rest
        if len(pitches) == 1:
            note = mxml.Note(pitches[0], duration, ties=ties, directions=dir_)  # type: ignore
            note.voice = note.staff = staff_idx
            return note
        chord = mxml.Chord(pitches, duration, ties=ties, directions=dir_)  # type: ignore
        for n in chord.notes:
            n.voice = n.staff = staff_idx
        return chord

    durations = _into_simple_times(duration)
    if not durations:
        return []
    durations = [mxml.Duration.from_written_length(float(d)) for d in durations]
    if len(durations) == 1:
        tie_status = [None]
    else:
        tie_status = ["start"] + ["continue"] * (len(durations) - 2) + ["stop"]
    return [make_dur_obj(d, t) for d, t in zip(durations, tie_status)]


def _make_measure_notes(
    notes: list[Note],
    staff_idx: int,
    new_measure: NewMeasure,
    keysig: str | None,
    directions: Sequence[BPMChange] = tuple(),
):
    notes_by_onset = sorted(
        groupby(notes, lambda e: e.onset).items(), key=lambda x: x[0]
    )

    # Stick directions to their closest notes
    def closest_note_group_idx(onset):
        onsets = np.array([onset for onset, _ in notes_by_onset])
        return int(np.argmin(np.abs(onsets - onset)))

    directions_ = groupby(directions, lambda d: closest_note_group_idx(d.onset))
    expected_mlen = new_measure.time_sig_n
    ret_objs: list[mxml.Note | mxml.Rest | mxml.Chord] = []
    cur_mlen = Fraction(0, 1)
    for i, (onset, group) in enumerate(notes_by_onset):
        onset -= new_measure.onset
        # Fast forward to current onset by adding rests
        ret_objs.extend(_simple_timed_elems(onset - cur_mlen, staff_idx, []))
        # Decide duration
        if (note_dur := max([n.duration for n in group])) == 0:
            # print("0 duration note detected")
            note_dur = float("inf")
        to_next_onset = float("inf")
        if i < len(notes_by_onset) - 1:
            to_next_onset = notes_by_onset[i + 1][0] - new_measure.onset - onset
        if (to_measure_end := expected_mlen - onset) <= 0:
            raise ValueError(
                f"Current measure ({expected_mlen} / {TIME_SIG_D}) cannot fit more notes"
            )
        duration = cast(Fraction, min(note_dur, to_next_onset, to_measure_end))
        pitches = [num_to_pitch(n.pitch_num, keysig) for n in group]
        # Round to int -- don't give silly numbers like 123.23123176 BPM
        bpm_changes = [mxml.MetronomeMark(1, int(b.bpm)) for b in directions_[i]]
        ret_objs.extend(_simple_timed_elems(duration, staff_idx, pitches, bpm_changes))
        cur_mlen = onset + duration
    # Pad to measure end by adding rests
    ret_objs.extend(_simple_timed_elems(expected_mlen - cur_mlen, staff_idx, []))
    return ret_objs
