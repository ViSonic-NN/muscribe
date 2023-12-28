import warnings
from abc import ABC, abstractmethod

import torch
import torch_scatter
from torch import Tensor


class SparseTransitionModel:
    """
    Transition model class for a HMM, storing the state transitions in a
    sparse representation.

    The transition model is defined similar to a scipy compressed sparse row
    matrix and holds all transition probabilities from one state to an other.
    This allows an efficient Viterbi decoding of the HMM.

    Parameters
    ----------
    states : Tensor, shape (num_transitions,)
        All states transitioning to state s are stored in:
        states[pointers[s]:pointers[s+1]]
    pointers : Tensor, shape (num_transitions,)
        Pointers for the `states` array for state s.
    probabilities : Tensor, shape (num_transitions,)
        The corresponding transition are stored in:
        probabilities[pointers[s]:pointers[s+1]].

    Notes
    -----
    This class should be either used for loading saved transition models or
    being sub-classed to define a specific transition model.

    See Also
    --------
    SparseTransitionModel.dense_to_sparse
    scipy.sparse.csr_matrix
    """

    def __init__(self, states: Tensor, pointers: Tensor, probabilities: Tensor):
        self.states = states
        self.pointers = pointers
        self.probs = torch.nn.Parameter(probabilities, requires_grad=True)

    @property
    def num_states(self):
        """Number of states."""
        return len(self.pointers) - 1

    @property
    def num_transitions(self):
        """Number of transitions."""
        return len(self.probs)

    @property
    def log_probabilities(self):
        """Transition log probabilities."""
        return torch.log(self.probs)

    # TODO: add make_dense() method
    # TODO: add from_dense() method

    @staticmethod
    def dense_to_sparse(states: Tensor, prev_states: Tensor, probabilities: Tensor):
        """
        Return a sparse representation of dense transitions.

        This method removes all duplicate states and thus allows an efficient
        Viterbi decoding of the HMM.

        Parameters
        ----------
        states : Tensor, shape (num_transitions,)
            Array with states (i.e. destination states).
        prev_states : Tensor, shape (num_transitions,)
            Array with previous states (i.e. origination states).
        probabilities : Tensor, shape (num_transitions,)
            Transition probabilities.

        Returns
        -------
        states : Tensor, shape (num_transitions,)
            All states transitioning to state s are returned in:
            states[pointers[s]:pointers[s+1]]
        pointers : Tensor, shape (num_transitions,)
            Pointers for the `states` array for state s.
        probabilities : Tensor, shape (num_transitions,)
            The corresponding transition are returned in:
            probabilities[pointers[s]:pointers[s+1]].

        See Also
        --------
        :class:`SparseTransitionModel`

        Notes
        -----
        Three 1D numpy arrays of same length must be given. The indices
        correspond to each other, i.e. the first entry of all three arrays
        define the transition from the state defined prev_states[0] to that
        defined in states[0] with the probability defined in probabilities[0].
        """
        from scipy.sparse import csr_matrix

        # check for a proper probability distribution, i.e. the emission
        # probabilities of each prev_state must sum to 1
        prev_states = prev_states.long()
        if not torch.allclose(
            torch.bincount(prev_states, weights=probabilities), torch.ones(1)
        ):
            raise ValueError("Not a probability distribution.")
        # convert everything into a sparse CSR matrix, make sure it is square.
        # looking through prev_states is enough, because there *must* be a
        # transition *from* every state
        num_states = prev_states.max() + 1
        transitions = csr_matrix(
            (probabilities, (states, prev_states)), shape=(num_states, num_states)
        )
        # convert to correct types
        states = torch.from_numpy(transitions.indices).long()
        pointers = torch.from_numpy(transitions.indptr).long()
        probabilities = torch.from_numpy(transitions.data)
        return states, pointers, probabilities


class ObservationModel(ABC):
    """
    Observation model class for a HMM.

    The observation model is defined as a plain 1D numpy arrays `pointers` and
    the methods `log_densities()` and `densities()` which return 2D numpy
    arrays with the (log) densities of the observations.

    Parameters
    ----------
    pointers : Tensor, shape (num_states,)
        Pointers from HMM states to the correct densities. The length of the
        array must be equal to the number of states of the HMM and pointing
        from each state to the corresponding column of the array returned
        by one of the `log_densities()` or `densities()` methods. The
        `pointers` type must be torch.long.

    See Also
    --------
    ObservationModel.log_densities
    ObservationModel.densities
    """

    def __init__(self, pointers: Tensor):
        self.pointers = pointers

    @abstractmethod
    def log_densities(self, observations: Tensor) -> Tensor:
        """
        Log densities (or probabilities) of the observations for each state.

        Parameters
        ----------
        observations : Tensor
            Observations.

        Returns
        -------
        Tensor
            Log densities as a 2D Tensor with the number of rows being
            equal to the number of observations and the columns representing
            the different observation log probability densities. The type must
            be np.float.

        """
        pass

    def densities(self, observations: Tensor):
        """
        Densities (or probabilities) of the observations for each state.

        This defaults to computing the exp of the `log_densities`.
        You can provide a special implementation to speed-up everything.

        Parameters
        ----------
        observations

        Returns
        -------
            Densities as a 2D Tensor with the number of rows being equal
            to the number of observations and the columns representing the
            different observation log probability densities. The type must be
            np.float.

        """
        return torch.exp(self.log_densities(observations))


class HiddenMarkovModel(object):
    """
    Hidden Markov Model

    To search for the best path through the state space with the Viterbi
    algorithm, the following parameters must be defined.

    Parameters
    ----------
    transition_model
    observation_model
    initial_distribution : Optional[Tensor], shape (num_states,)
        Initial state distribution; if 'None' a uniform distribution is
        assumed.
    """

    def __init__(
        self,
        transition_model: SparseTransitionModel,
        observation_model: ObservationModel,
        initial_distribution: Tensor | None = None,
    ):
        self.transition_model = tm = transition_model
        self.observation_model = om = observation_model
        self.n_hstates = n_hstates = transition_model.num_states
        if initial_distribution is None:
            initial_distribution = torch.ones(n_hstates) / n_hstates
        if not torch.allclose(initial_distribution.sum(), torch.ones(1)):
            raise ValueError(
                "Initial distribution is not a probability " "distribution."
            )
        self.init_dist = torch.nn.Parameter(initial_distribution)

        # This hidden Markov model is a sparse model where each state only look at a few states.
        # `range_sizes` holds the number of states (a small number) that each state looks at.
        hid_states = torch.arange(self.n_hstates)
        range_sizes = tm.pointers[hid_states + 1] - tm.pointers[hid_states]
        # `hid_states` is an "expanded" list of hidden states. From 0 to self.n_hstates - 1,
        # each number `i` is repeated `range_sizes[i]` times.
        self.hid_states = torch.repeat_interleave(hid_states, range_sizes)
        # `obs_states` is a list of the observed states corresponding to each hidden state.
        self.obs_states = om.pointers[self.hid_states]
        # `prev_states` is a flattened list of the previous states of each hidden state.
        # It would be a ragged 2D list, but is then flattened.
        self.prev_states = tm.states
        self.pointers = torch.arange(int(tm.pointers.max()))

    def viterbi(self, obs: Tensor):
        """
        Determine the best path with the Viterbi algorithm.

        Parameters
        ----------
        obs : Tensor, shape (n_observations, n_obs_states)
            Observations to decode the optimal path for.

        Returns
        -------
        path : Tensor, shape (n_observations,)
            Best state-space path sequence.
        log_prob : float
            Corresponding log probability.
        """

        # Viterbi variables, init with the initial state distribution
        curr_viterbi = torch.log(self.init_dist)
        # Back-tracking pointers
        n_obs = obs.shape[0]
        bt_pointers = torch.empty(
            [n_obs, self.n_hstates], dtype=torch.long, device=obs.device
        )

        obs_probs = self.observation_model.log_densities(obs)
        trans_probs = self.transition_model.log_probabilities[self.pointers]
        trans_and_obs = trans_probs + obs_probs[:, self.obs_states]
        # Iterate over all observations
        for frame in range(n_obs):
            # The following few lines are equivalent to a log-domain sparse matmul
            # (which we don't have an operator for, so this is a workaround).
            # Weight the previous state with the transition probability
            # and the current observation probability density.
            transition_prob = curr_viterbi[self.prev_states] + trans_and_obs[frame]
            # FIXME: the default value of scatter_max is 0, which is incorrect.
            # Should be -inf.
            curr_viterbi, argmax = torch_scatter.scatter_max(
                transition_prob, self.hid_states
            )
            # Update the backtracking pointers
            bt_pointers[frame] = self.prev_states[argmax]
        # fetch the final best state
        state = curr_viterbi.argmax()
        # set the path's probability to that of the best state
        log_prob = curr_viterbi[state]

        # raise warning if the sequence has -inf probability
        if torch.isinf(log_prob):
            warnings.warn(
                "-inf log probability during Viterbi decoding "
                "cannot find a valid path",
                RuntimeWarning,
                stacklevel=1,
            )
            # return empty path sequence
            return torch.empty(0, dtype=torch.long, device=obs.device), log_prob
        # back tracked path, a.k.a. path sequence
        path = torch.empty(n_obs, dtype=torch.long, device=obs.device)
        # track the path backwards, start with the last frame and do not
        # include the pointer for frame 0, since it includes the transitions
        # to the prior distribution states
        for frame in range(n_obs - 1, -1, -1):
            # save the state in the path
            path[frame] = state
            # fetch the next previous one
            state = bt_pointers[frame, state]
        # return the tracked path and its probability
        return path, log_prob
