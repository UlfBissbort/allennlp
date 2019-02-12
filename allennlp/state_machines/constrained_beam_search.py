from collections import defaultdict
from typing import Dict, List, Optional

import torch

from allennlp.state_machines import util
from allennlp.state_machines.beam_search import BeamSearch, StateType
from allennlp.state_machines.transition_functions import TransitionFunction


class ConstrainedBeamSearch(BeamSearch[StateType]):
    """
    This class implements beam search over transition sequences given an initial ``State``, a
    ``TransitionFunction``, and a list of allowed transition sequences.  We will do a beam search
    `over the list of allowed sequences` and return the highest scoring states found by the beam.
    This is only actually a `beam search` if your beam size is smaller than the list of allowed
    transition sequences; otherwise, we are just scoring and sorting the sequences using a prefix
    tree.

    The initial ``State`` is assumed to be `batched`.  The value we return from the search is a
    dictionary from batch indices to ranked finished states.

    IMPORTANT: We assume that the ``TransitionFunction`` that you are using returns possible next
    states in sorted order, so we do not do an additional sort inside of
    ``ConstrainedBeamSearch.search()``.  If you're implementing your own ``TransitionFunction``,
    you must ensure that you've sorted the states that you return.

    Parameters
    ----------
    beam_size : ``Optional[int]``
        The beam size to use.  Because this is a `constrained` beam search, we allow for the case
        where you just want to evaluate all options in the constrained set.  In that case, you
        don't need a beam, and you can pass a beam size of ``None``, and we will just evaluate
        everything.  This lets us be more efficient in :func:`TransitionFunction.take_step` and
        skip the sorting that is typically done there.
    allowed_sequences : ``torch.Tensor``
        A ``(batch_size, num_sequences, sequence_length)`` tensor containing the transition
        sequences that we will search in.  The values in this tensor must match whatever the
        ``State`` keeps in its ``action_history`` variable (typically this is action indices).
    allowed_sequence_mask : ``torch.Tensor``
        A ``(batch_size, num_sequences, sequence_length)`` tensor indicating whether each entry in
        the ``allowed_sequences`` tensor is padding.  The allowed sequences could be padded both on
        the ``num_sequences`` dimension and the ``sequence_length`` dimension.
    per_node_beam_size : ``int``, optional (default = beam_size)
        The maximum number of candidates to consider per node, at each step in the search.
        If not given, this just defaults to `beam_size`. Setting this parameter
        to a number smaller than `beam_size` may give better results, as it can introduce
        more diversity into the search. See Freitag and Al-Onaizan 2017,
        "Beam Search Strategies for Neural Machine Translation".
    allow_partial_constraints: ``bool``, optional (default = False)
        If True, then sequences that "go beyond" the constraints continue with
        unconstrained beam search. In particular, this allows the constraint that
        sequences must _start_ a certain way.
    """
    def __init__(self,
                 beam_size: Optional[int],
                 allowed_sequences: torch.Tensor,
                 allowed_sequence_mask: torch.Tensor = None,
                 per_node_beam_size: int = None,
                 allow_partial_constraints: bool = False) -> None:
        super().__init__(beam_size, per_node_beam_size)
        if allowed_sequence_mask is None:
            allowed_sequence_mask = torch.ones_like(allowed_sequences)
        self._allowed_transitions = util.construct_prefix_tree(allowed_sequences, allowed_sequence_mask)
        self._allow_partial_constraints = allow_partial_constraints

    def search(self,
               num_steps: int,
               initial_state: StateType,
               transition_function: TransitionFunction,
               keep_final_unfinished_states: bool = True) -> Dict[int, List[StateType]]:
        """
        Parameters
        ----------
        num_steps : int
            If self._allow_partial_constraints is False, this argument is ignored, as it's implicit
            in the constraints themselves. Otherwise it's the maximum number of steps to take in our search.
        initial_state : ``State``
            The starting state of our search.  This is assumed to be `batched`, and our beam search
            is batch-aware - we'll keep ``beam_size`` states around for each instance in the batch.
        transition_function : ``TransitionFunction``
            The ``TransitionFunction`` object that defines and scores transitions from one state to the
            next.
        keep_final_unfinished_states : ``bool``, optional (default: True)
            If we run out of time steps should we return unfinished states?
            Again, this is ignored unless self._allow_partial_constraints is True.

        Returns
        -------
        best_states : ``Dict[int, List[StateType]]``
            This is a mapping from batch index to the top states for that instance.
        """
        finished_states: Dict[int, List[StateType]] = defaultdict(list)
        states = [initial_state]
        step_num = 0
        while states and (num_steps is None or step_num < num_steps):
            step_num += 1
            print(step_num, states, finished_states)
            keep_unfinished_states_this_step = keep_final_unfinished_states and step_num == num_steps
            next_states: Dict[int, List[StateType]] = defaultdict(list)
            grouped_state = states[0].combine_states(states)
            allowed_actions = []
            for batch_index, action_history in zip(grouped_state.batch_indices,
                                                   grouped_state.action_history):
                history = tuple(action_history)
                allowed_transitions = self._allowed_transitions[batch_index]
                if history in allowed_transitions:
                    allowed_actions.append(allowed_transitions[history])
                elif self._allow_partial_constraints:
                    allowed_actions.append(None)
                else:
                    raise RuntimeError("no valid transitions")
            for next_state in transition_function.take_step(grouped_state,
                                                            max_actions=self._per_node_beam_size,
                                                            allowed_actions=allowed_actions):
                # NOTE: we're doing state.batch_indices[0] here (and similar things below),
                # hard-coding a group size of 1.  But, our use of `next_state.is_finished()`
                # already checks for that, as it crashes if the group size is not 1.
                batch_index = next_state.batch_indices[0]
                is_finished = next_state.is_finished()

                if is_finished or keep_unfinished_states_this_step:
                    finished_states[batch_index].append(next_state)
                if not is_finished:
                    next_states[batch_index].append(next_state)
            states = []
            for batch_index, batch_states in next_states.items():
                # The states from the generator are already sorted, so we can just take the first
                # ones here, without an additional sort.
                if self._beam_size:
                    batch_states = batch_states[:self._beam_size]
                states.extend(batch_states)

        print(finished_states)
        best_states: Dict[int, List[StateType]] = {}
        for batch_index, batch_states in finished_states.items():
            # The time this sort takes is pretty negligible, no particular need to optimize this
            # yet.  Maybe with a larger beam size...
            finished_to_sort = [(-state.score[0].item(), state) for state in batch_states]
            finished_to_sort.sort(key=lambda x: x[0])
            best_states[batch_index] = [state[1] for state in finished_to_sort[:self._beam_size]]
        return best_states
