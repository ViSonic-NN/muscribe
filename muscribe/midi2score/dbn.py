from typing import TypeVar, cast

import numpy as np
import torch
from torch import Tensor, nn

import hmm

T = TypeVar("T")
OLIST = T | list[T]


class DBNDownBeatTrackingProcessor(nn.Module):
    """
    Downbeat tracking with RNNs and a dynamic Bayesian network (DBN)
    approximated by a Hidden Markov Model (HMM).

    Parameters
    ----------
    beats_per_bar: Number of beats per bar to be modeled. Can be either a single number
        or a list or array with bar lengths (in beats).
    fps: Frames per second.
    min_bpm: Minimum tempo used for beat tracking [bpm]. If a list is given, each
        item corresponds to the number of beats per bar at the same position.
    max_bpm: Maximum tempo used for beat tracking [bpm]. If a list is given, each
        item corresponds to the number of beats per bar at the same position.
    num_tempi: Number of tempi to model; if set, limit the number of tempi and use a
        log spacing, otherwise a linear spacing. If a list is given, each
        item corresponds to the number of beats per bar at the same position.
    transition_lambda: Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat to the next one).  If a list is
        given, each item corresponds to the number of beats per bar at the
        same position.
    observation_lambda: Split one (down-)beat period into `observation_lambda` parts, the first
        representing (down-)beat states and the remaining non-beat states.
    threshold: Threshold the RNN (down-)beat activations before Viterbi decoding.
    correct: bool, optional
        Correct the beats (i.e. align them to the nearest peak of the
        (down-)beat activation function).

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Joint Beat and Downbeat Tracking with Recurrent Neural Networks"
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.

    Examples
    --------
    Create a DBNDownBeatTrackingProcessor. The returned array represents the
    positions of the beats and their position inside the bar. The position is
    given in seconds, thus the expected sampling rate is needed. The position
    inside the bar follows the natural counting and starts at 1.

    The number of beats per bar which should be modelled must be given, all
    other parameters (e.g. tempo range) are optional but must have the same
    length as `beats_per_bar`, i.e. must be given for each bar length.

    >>> proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.downbeats.DBNDownBeatTrackingProcessor object at 0x...>

    Call this DBNDownBeatTrackingProcessor with the beat activation function
    returned by RNNDownBeatProcessor to obtain the beat positions.

    >>> act = RNNDownBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([[0.09, 1. ],
           [0.45, 2. ],
           ...,
           [2.14, 3. ],
           [2.49, 4. ]])

    """

    MIN_BPM = 55.0
    MAX_BPM = 215.0
    NUM_TEMPI = 60
    TRANSITION_LAMBDA = 100
    OBSERVATION_LAMBDA = 16
    THRESHOLD = 0.05
    CORRECT = True

    def __init__(
        self,
        beats_per_bar: OLIST[int],
        fps: float,
        min_bpm: OLIST[float] = MIN_BPM,
        max_bpm: OLIST[float] = MAX_BPM,
        num_tempi: OLIST[int] = NUM_TEMPI,
        transition_lambda: OLIST[float] = TRANSITION_LAMBDA,
        observation_lambda: int = OBSERVATION_LAMBDA,
        threshold: float = THRESHOLD,
        correct: bool = CORRECT,
    ):
        super().__init__()
        self.threshold = threshold
        self.correct = correct
        self.fps = fps
        _beats_per_bar = expand_list(beats_per_bar)
        nbeats = len(_beats_per_bar)
        _min_bpm, _max_bpm = expand_list(min_bpm, nbeats), expand_list(max_bpm, nbeats)
        _num_tempi = expand_list(num_tempi, nbeats)
        _transition_lambda = expand_list(transition_lambda, nbeats)
        # convert timing information to construct a beat state space
        min_interval = 60.0 * fps / _max_bpm
        max_interval = 60.0 * fps / _min_bpm
        # model the different bar lengths

        def make_hmm(bi: int):
            # model N beats as a bar
            st = BarStateSpace(
                _beats_per_bar[bi], min_interval[bi], max_interval[bi], _num_tempi[bi]
            )
            tm = BarTransitionModel(st, _transition_lambda[bi])
            om = RNNDownBeatTrackingObservationModel(st, observation_lambda)
            return hmm.HiddenMarkovModel(tm, om)

        self.hmms = [make_hmm(i) for i in range(nbeats)]

    def forward(self, activations: Tensor) -> Tensor:
        """
        Detect the (down-)beats in the given activation function.

        Parameters
        ----------
        activations: torch.Tensor, shape (num_frames, 2)
            Activation function with probabilities corresponding to beats
            and downbeats given in the first and second column, respectively.

        Returns
        -------
        beats: torch.Tensor, shape (num_beats, 2)
            Detected (down-)beat positions [seconds] and beat numbers.

        """
        import itertools as it

        # use only the activations > threshold (init offset to be added later)
        first = 0
        if self.threshold:
            activations, first = threshold_activations(activations, self.threshold)
        # return no beats if no activations given / remain after thresholding
        if not activations.any():
            return torch.zeros((0, 2))
        # decoding of the activations with HMM
        results = [a.viterbi(b) for a, b in zip(self.hmms, it.repeat(activations))]
        # choose the best HMM (highest log probability)
        best = torch.argmax(torch.stack([r[1] for r in results]))
        # the best path through the state space
        path, _ = results[best]
        # the state space and observation model of the best HMM
        st = cast(BarTransitionModel, self.hmms[best].transition_model).state_space
        om = self.hmms[best].observation_model
        # the positions inside the pattern (0..num_beats)
        positions = st.state_positions[path]
        # corresponding beats (add 1 for natural counting)
        beat_numbers = positions.long() + 1
        if self.correct:
            beats = torch.empty(0, dtype=torch.long)
            # for each detection determine the "beat range", i.e. states where
            # the pointers of the observation model are >= 1
            beat_range = om.pointers[path] >= 1
            # if there aren't any in the beat range, there are no beats
            if not beat_range.any():
                return torch.zeros((0, 2))
            # get all change points between True and False (cast to int before)
            idx = torch.nonzero(torch.diff(beat_range.long()), as_tuple=True)[0] + 1
            # if the first frame is in the beat range, add a change at frame 0
            if beat_range[0]:
                idx = torch.cat([torch.tensor(0), idx])
            # if the last frame is in the beat range, append the length of the
            # array
            if beat_range[-1]:
                idx = torch.cat([idx, torch.tensor(beat_range.size())])
            # iterate over all regions
            for left, right in idx.reshape((-1, 2)):
                # pick the frame with the highest activations value
                # Note: we look for both beats and down-beat activations;
                #       since np.argmax works on the flattened array, we
                #       need to divide by 2
                peak = torch.argmax(activations[left:right], keepdim=True) // 2 + left
                beats = torch.cat([beats, peak])
        else:
            # transitions are the points where the beat numbers change
            # FIXME: we might miss the first or last beat!
            #        we could calculate the interval towards the beginning/end
            #        to decide whether to include these points
            beats = torch.nonzero(torch.diff(beat_numbers), as_tuple=True)[0] + 1
        # return the beat positions (converted to seconds) and beat numbers
        return torch.stack(
            [(beats + first) / float(self.fps), beat_numbers[beats]], dim=1
        )


class BeatStateSpace:
    """
    State space for beat tracking with a HMM.

    Parameters
    ----------
    min_interval: Minimum interval to model.
    max_interval: Maximum interval to model.
    num_intervals: Number of intervals to model; if set, limit the number of intervals
        and use a log spacing instead of the default linear spacing.

    Attributes
    ----------
    num_states: int
        Number of states.
    num_intervals: int
        Number of intervals.
    intervals: Tensor, shape (num_intervals,)
        Modeled intervals.
    first_states: Tensor, shape (num_intervals,)
        First state of each interval.
    last_states: Tensor, shape (num_intervals,)
        Last state of each interval.
    state_positions: Tensor, shape (num_states,)
        Positions of the states (i.e. 0...1).
    state_intervals: Tensor, shape (num_states,)
        Intervals of the states (i.e. 1 / tempo).

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    """

    def __init__(
        self, min_interval: float, max_interval: float, num_intervals: int | None = None
    ):
        from math import log2

        # per default, use a linear spacing of the tempi
        intervals = torch.arange(round(min_interval), round(max_interval) + 1).long()
        # if num_intervals is given (and smaller than the length of the linear
        # spacing of the intervals) use a log spacing and limit the number of
        # intervals to the given value
        if num_intervals is not None and num_intervals < len(intervals):
            # we must approach the number of intervals iteratively
            num_log_intervals = num_intervals
            intervals = torch.zeros(0)
            while len(intervals) < num_intervals:
                intervals = torch.logspace(
                    log2(min_interval),
                    log2(max_interval),
                    num_log_intervals,
                    base=2,
                )
                # quantize to integer intervals
                intervals = torch.unique(torch.round(intervals)).long()
                num_log_intervals += 1
        # save the intervals
        self.intervals = intervals
        # number of states and intervals
        self.num_states = int(torch.sum(intervals))
        self.num_intervals = len(intervals)
        # define first and last states
        first_states = torch.cumsum(torch.cat([torch.zeros(1), intervals[:-1]]), dim=0)
        self.first_states = first_states.long()
        self.last_states = torch.cumsum(intervals, dim=0) - 1
        # define the positions and intervals of the states
        self.state_positions = torch.empty(self.num_states)
        self.state_intervals = torch.empty(self.num_states, dtype=torch.long)
        # Note: having an index counter is faster than ndenumerate
        idx = 0
        for i in intervals.tolist():
            self.state_positions[idx : idx + i] = torch.linspace(0, 1, i + 1)[:i]
            self.state_intervals[idx : idx + i] = i
            idx += i


class BarStateSpace:
    """
    State space for bar tracking with a HMM.

    Model `num_beat` identical beats with the given arguments in a single state
    space.

    Parameters
    ----------
    num_beats: Number of beats to form a bar.
    min_interval: Minimum beat interval to model.
    max_interval: Maximum beat interval to model.
    num_intervals: int, optional
        Number of beat intervals to model; if set, limit the number of
        intervals and use a log spacing instead of the default linear spacing.

    Attributes
    ----------
    num_beats
    num_states: int
        Number of states.
    num_intervals: int
        Number of intervals.
    state_positions: Tensor, shape (num_states,)
        Positions of the states.
    state_intervals: Tensor, shape (num_states,)
        Intervals of the states.
    first_states: list[Tensor], each of shape (num_intervals,)
        First states of each beat.
    last_states: list[Tensor], each of shape (num_intervals,)
        Last states of each beat.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.
    """

    def __init__(
        self,
        num_beats: int,
        min_interval: int,
        max_interval: int,
        num_intervals: int | None = None,
    ):
        # model N beats as a bar
        self.num_beats = num_beats
        self.num_states = 0
        # create a BeatStateSpace and stack it `num_beats` times
        bss = BeatStateSpace(min_interval, max_interval, num_intervals)
        # define position (add beat counter) and interval states
        self.state_positions = torch.cat(
            [bss.state_positions + b for b in range(self.num_beats)]
        )
        self.state_intervals = torch.cat(
            [bss.state_intervals for _ in range(self.num_beats)]
        )
        # save the first and last states of the individual beats in a list
        self.first_states, self.last_states = [], []
        for _ in range(self.num_beats):
            # add the current number of states as offset
            self.first_states.append(bss.first_states + self.num_states)
            self.last_states.append(bss.last_states + self.num_states)
            # finally increase the number of states
            self.num_states += bss.num_states


class BarTransitionModel(hmm.SparseTransitionModel):
    """
    Transition model for bar tracking with a HMM.

    Within the beats of the bar the tempo stays the same; at beat boundaries
    transitions from one tempo (i.e. interval) to another following an
    exponential distribution are allowed.

    Parameters
    ----------
    state_space: :class:`BarStateSpace` instance
    transition_lambda: float or list
        Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat to the next one).
        None can be used to set the tempo change probability to 0.
        If a list is given, the individual values represent the lambdas for
        each transition into the beat at this index position.

    Attributes
    ----------
    state_space: :class:`BarStateSpace` instance
    tlambda: list
    states: Tensor, shape (num_transitions,)
    pointers: Tensor, shape (num_transitions,)
    probabilities: Tensor, shape (num_transitions,)

    Notes
    -----
    Bars performing tempo changes only at bar boundaries (and not at the beat
    boundaries) must have set all but the first `transition_lambda` values to
    None, e.g. [100, None, None] for a bar with 3 beats.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.
    """

    def __init__(self, state_space: BarStateSpace, transition_lambda: OLIST[float]):
        # expand transition_lambda to a list if a single value is given
        if isinstance(transition_lambda, list):
            tlambda = transition_lambda
        else:
            tlambda = [transition_lambda] * state_space.num_beats
        if state_space.num_beats != len(tlambda):
            raise ValueError(
                "length of `transition_lambda` must be equal to "
                "`num_beats` of `state_space`."
            )
        # save attributes
        self.state_space = state_space
        self.tlambda = tlambda
        # TODO: this could be unified with the BeatTransitionModel
        # same tempo transitions probabilities within the state space is 1
        # Note: use all states, but remove all first states of the individual
        #       beats, because there are no same tempo transitions into them
        states = torch.arange(state_space.num_states, dtype=torch.long)
        diff_rhs = set(torch.cat(state_space.first_states).tolist())
        states = torch.tensor(list(set(states.tolist()) - diff_rhs))
        prev_states = states - 1
        probabilities = torch.ones_like(states).float()
        # tempo transitions occur at the boundary between beats (unless the
        # corresponding transition_lambda is set to None)
        for beat in range(state_space.num_beats):
            # connect to the first states of the actual beat
            to_states = state_space.first_states[beat]
            # connect from the last states of the previous beat
            from_states = state_space.last_states[beat - 1]
            # transition follow an exponential tempo distribution
            from_int = state_space.state_intervals[from_states]
            to_int = state_space.state_intervals[to_states]
            prob = exponential_transition(from_int, to_int, tlambda[beat])
            # use only the states with transitions to/from != 0
            from_prob, to_prob = torch.nonzero(prob).T
            states = torch.cat((states, to_states[to_prob]))
            prev_states = torch.cat((prev_states, from_states[from_prob]))
            probabilities = torch.cat((probabilities, prob[prob != 0]))
        # make the transitions sparse
        states, pointers, probabilities = self.dense_to_sparse(
            states, prev_states, probabilities
        )
        super().__init__(states, pointers, probabilities)


class RNNDownBeatTrackingObservationModel(hmm.ObservationModel):
    """
    Observation model for downbeat tracking with a HMM.

    Parameters
    ----------
    state_space: BarStateSpace instance.
    observation_lambda: Split each (down-)beat period into `observation_lambda` parts, the
        first representing (down-)beat states and the remaining non-beat
        states.

    Attributes
    ----------
    pointers: Tensor, shape (num_states,)

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Joint Beat and Downbeat Tracking with Recurrent Neural Networks"
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.
    """

    def __init__(self, state_space: BarStateSpace, observation_lambda: int):
        self.observation_lambda = observation_lambda
        # compute observation pointers
        # always point to the non-beat densities
        pointers = torch.zeros(state_space.num_states, dtype=torch.long)
        # unless they are in the beat range of the state space
        border = 1.0 / observation_lambda
        pointers[state_space.state_positions % 1 < border] = 1
        # the downbeat (i.e. the first beat range) points to density column 2
        pointers[state_space.state_positions < border] = 2
        super().__init__(pointers)

    def log_densities(self, observations: Tensor):
        """
        Compute the log densities of the observations.

        Parameters
        ----------
        observations: Tensor, shape (N, 2)
            Observations (i.e. 2D activations of a RNN, the columns represent
            'beat' and 'downbeat' probabilities)

        Returns
        -------
        Tensor, shape (N, 3)
            Log densities of the observations, the columns represent the
            observation log probability densities for no-beats, beats and
            downbeats.
        """

        no_beats = (1.0 - torch.sum(observations, dim=1)) / (
            self.observation_lambda - 1
        )
        densities = torch.cat((no_beats.unsqueeze(1), observations), dim=1)
        return densities.log()


def threshold_activations(activations: Tensor, threshold: float):
    """
    Threshold activations to include only the main segment exceeding the given
    threshold (i.e. first to last time/index exceeding the threshold).

    Parameters
    ----------
    activations: Activations to be thresholded.
    threshold: Threshold value.

    Returns
    -------
    activations: Thresholded activations
    start: Index of the first activation exceeding the threshold.

    Notes
    -----

    This function can be used to extract the main segment of beat activations
    to track only the beats where the activations exceed the threshold.
    """

    first = last = 0
    # use only the activations > threshold
    idx = torch.nonzero(activations >= threshold, as_tuple=True)[0]
    if idx.any():
        first = max(first, int(idx.min()))
        last = min(len(activations), int(idx.max()) + 1)
    # return thresholded activations segment and first index
    return activations[first:last], first


# transition distributions
def exponential_transition(
    from_intervals: Tensor,
    to_intervals: Tensor,
    transition_lambda: float,
    threshold: float | None = None,
    norm: bool = True,
):
    """
    Exponential tempo transition.

    Parameters
    ----------
    from_intervals: Intervals where the transitions originate from.
    to_intervals: Intervals where the transitions terminate.
    transition_lambda: Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat/bar to the next one). If None,
        allow only transitions from/to the same interval.
    threshold: Set transition probabilities below this threshold to zero.
    norm: Normalize the emission probabilities to sum 1.

    Returns
    -------
    probabilities: Tensor, shape (num_from_intervals, num_to_intervals)
        Probability of each transition from an interval to another.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.
    """

    # no transition lambda
    if transition_lambda is None:
        # return a diagonal matrix
        return torch.diag(
            torch.diag(torch.ones((len(from_intervals), len(to_intervals))))
        )
    # compute the transition probabilities
    ratio = to_intervals.float() / from_intervals.float().unsqueeze(1)
    prob = torch.exp(-transition_lambda * torch.abs(ratio - 1.0))
    # set values below threshold to 0
    if threshold is None:
        threshold = np.spacing(1)
    prob[prob <= threshold] = 0
    # normalize the emission probabilities
    if norm:
        prob /= torch.sum(prob, dim=1).unsqueeze(1)
    return prob


def expand_list(x: OLIST[T], n: int | None = None) -> np.ndarray:
    """Expand arguments into list of n elements if they are not already."""

    ret = np.array(x, ndmin=1)
    if n is None or (n_ := ret.shape[0]) == n:
        return ret
    if n_ == 1:
        return np.repeat(ret, n)
    raise ValueError(f"Invalid number of items in {x}: {n_} (expected 1 or {n})")
