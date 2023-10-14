from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterator, NamedTuple, TypeVar, cast

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


def groupby(xs: list[T1], key: Callable[[T1], T2]) -> dict[T2, list[T1]]:
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


class NewMeasure(NamedTuple):
    onset: Fraction
    time_sig_n: int


def _make_events_list(
    beats: np.ndarray,
    midi_notes: np.ndarray,
    hand_parts: np.ndarray,
    key_changes: np.ndarray,
):
    def find_in_beats(time):
        next_beat = np.searchsorted(beats[:, 0], time)
        assert 1 <= next_beat <= len(beats)
        bstart = beats[next_beat - 1, 0]
        if next_beat == len(beats):
            # There's no next beat, but we can extrapolate the length
            # from the previous beat...
            assert next_beat > 1
            blength = bstart - beats[next_beat - 2, 0]
        else:
            blength = beats[next_beat, 0] - bstart
        return next_beat - 1, bstart, blength

    def to_frac(duration: float, denoms=(1, 2, 4, 8), max_n_pieces=2):
        denoms = np.array(denoms).astype(int)
        numers = np.round(duration * denoms).astype(int)
        errors = np.abs(numers / denoms - duration)
        n_pieces = np.array(
            [len(_into_simple_times(Fraction(n, d))) for n, d in zip(numers, denoms)]
        )
        errors[n_pieces > max_n_pieces] = np.inf
        smallest = int(np.argmin(errors))
        return Fraction(int(numers[smallest]), int(denoms[smallest]))

    assert len(hand_parts) == len(midi_notes)
    events: list[Note | KeyChange | NewMeasure] = []
    for (pitch, nstart, ndur, velo), hand in zip(midi_notes, hand_parts):
        note_beat, bstart, blength = find_in_beats(nstart)
        noffset = to_frac((nstart - bstart) / blength)
        nlength = to_frac(ndur / blength)
        is_right_hand = hand == 0
        events.append(
            Note(note_beat + noffset, nlength, int(pitch), velo, is_right_hand)
        )
    for time, key in key_changes:
        note_beat, bstart, blength = find_in_beats(time)
        # TODO: check `offset = to_frac((time - bstart) / blength)` close to 0
        events.append(KeyChange(note_beat, key))
    # Minus 1 -- start from 0
    beat_counts = (beats[:, 1].astype(int) - 1).tolist()
    for idx, beat in enumerate(beat_counts):
        if (beat == 0 and idx > 0) or idx == len(beat_counts) - 1:
            prev_beat = beat_counts[idx - 1]
            prev_start = Fraction(idx - 1 - prev_beat, 1)
            events.append(NewMeasure(prev_start, prev_beat + 1))  # Add this 1 beat back
    # Sort NewMeasure before everything else when they occur at the same time
    return sorted(events, key=lambda e: (e.onset, not isinstance(e, NewMeasure)))


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
    duration: Fraction, pitches: list[mxml.Pitch] | None, staff_idx: int
):
    def make_dur_obj(duration: mxml.Duration, ties: str | None):
        if pitches is None:
            rest = mxml.Rest(duration)
            rest.voice = rest.staff = staff_idx
            return rest
        if len(pitches) == 1:
            note = mxml.Note(pitches[0], duration, ties=ties)  # type: ignore
            note.voice = note.staff = staff_idx
            return note
        chord = mxml.Chord(pitches, duration, ties=ties)  # type: ignore
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
    notes: list[Note], staff_idx: int, new_measure: NewMeasure, keysig: str | None
):
    notes_by_onset = sorted(
        groupby(notes, lambda e: e.onset).items(), key=lambda x: x[0]
    )
    expected_mlen = new_measure.time_sig_n
    ret_objs: list[mxml.Note | mxml.Rest | mxml.Chord] = []
    cur_mlen = Fraction(0, 1)
    for i, (onset, group) in enumerate(notes_by_onset):
        onset -= new_measure.onset
        # Fast forward to current onset by adding rests
        ret_objs.extend(_simple_timed_elems(onset - cur_mlen, None, staff_idx))
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
        ret_objs.extend(_simple_timed_elems(duration, pitches, staff_idx))
        cur_mlen = onset + duration
    # Pad to measure end by adding rests
    ret_objs.extend(_simple_timed_elems(expected_mlen - cur_mlen, None, staff_idx))
    return ret_objs


def write_to_xml(
    beats: np.ndarray,
    midi_notes: np.ndarray,
    hand_parts: np.ndarray,
    key_changes: np.ndarray,
    output_file: str | Path,
):
    def get_unique(d: dict, ty: type[T1]) -> T1 | None:
        if len(v := d[ty]) > 1:
            raise ValueError(f"Multiple {ty} in one onset: {v}")
        return v[0] if len(v) == 1 else None

    events = _make_events_list(beats, midi_notes, hand_parts, key_changes)
    measures = []
    last_key_sig: str | None = None
    last_time_sig: int | None = None
    for mevents in splitby(events, lambda e: isinstance(e, NewMeasure)):
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
        rhs_content = _make_measure_notes(notes[True], 1, new_measure, last_key_sig)
        lhs_content = _make_measure_notes(notes[False], 2, new_measure, last_key_sig)
        measures.append(mxml.Measure([rhs_content, lhs_content], **m_kwargs))
    score = mxml.Score([mxml.Part("Piano", measures)])
    score.export_to_file(str(output_file))
