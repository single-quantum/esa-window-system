from copy import copy, deepcopy

import numpy.typing as npt
import numpy as np

from esawindowsystem.core.encoder_functions import map_PPM_symbols


class Edge:

    __slots__ = ('from_state', 'to_state', 'edge_input', 'edge_output', 'gamma',
                 'lmbda', 'hamming_distance', 'edge_input_label', 'edge_output_label')

    def __init__(self):

        self.from_state: int | None = None
        self.to_state: int | None = None
        self.edge_input = None
        self.edge_output = None
        # Used for the BCJR algorithm
        self.gamma = None
        self.lmbda = None
        # Used for the Viterbi algorithm
        self.hamming_distance = None

    def set_edge(
            self,
            from_state: int,
            to_state: int,
            edge_input: int | list[int] | tuple[int, ...] | npt.NDArray[np.int8],
            edge_output: npt.NDArray[np.int_] | list[int] | tuple[int, ...],
            gamma: float | None):

        self.from_state: int | None = from_state
        self.to_state: int | None = to_state
        self.edge_input = edge_input
        self.edge_output = edge_output
        if isinstance(edge_input, int):
            self.edge_input_label = edge_input
        else:
            self.edge_input_label = map_PPM_symbols(edge_input, len(edge_input))[0]

        self.set_edge_output_label()
        self.gamma = gamma

    def set_edge_output_label(self):
        """Set edge output label, based on edge output. Converts tuple of bits to number, based on bit value. """
        if self.edge_output is None:
            return

        if any(i < 0 for i in self.edge_output):
            self.edge_output_label = map_PPM_symbols(
                tuple(0 if i < 0 else 1 for i in self.edge_output), len(self.edge_output))[0]
        else:
            self.edge_output_label = map_PPM_symbols(self.edge_output, len(self.edge_output))[0]

    # Not sure if this is necessary, but this way the edges can be sorted and
    # the 0 input is always the first item in the list of edges
    def __lt__(self, other):
        if not self.edge_input:
            return
        return self.edge_input < other.edge_input


class State:
    __slots__ = ('label', 'edges', 'alpha', 'beta')

    def __init__(self, label: int, num_edges: int):
        self.label: int = label
        self.edges: list[Edge] = []
        self.alpha: None | float = None
        self.beta: None | float = None

    def __str__(self):
        return self.label


class Stage:
    __slots__ = ('time_step', 'states')

    def __init__(self, num_states: int, num_input_bits: int, time_step: int):
        self.time_step = time_step

        # Initiate states (States here do not have edges yet)
        self.states: tuple[State, ...] = tuple(State(i, 2**num_input_bits) for i in range(num_states))

    def __lt__(self, other):
        if not self.time_step:
            return
        return self.time_step < other.time_step


class Trellis:
    __slots__ = ('memory_size', 'num_states', 'stages', 'num_input_bits', 'num_output_bits', 'edge_model', 'edges')

    def __init__(
            self,
            memory_size: int,
            num_output_bits: int,
            time_steps: int,
            edges: list[list[Edge]],
            num_input_bits: int = 1):

        self.memory_size = memory_size
        self.num_states = 2**memory_size

        # There are two stages for 1 timestep/transition, one for the initial state and one for the final state.
        # The `time_steps + 1`-th stage has all states, but no edges. This is needed to properly calculate beta
        self.stages: tuple[Stage, ...] = tuple(
            Stage(self.num_states, num_input_bits, time_step) for time_step in range(time_steps + 1)
        )
        # self.stages = sorted(self.stages)
        self.num_input_bits = num_input_bits
        self.num_output_bits = num_output_bits
        self.edge_model = edges

    def set_edges(self,
                  edges: list[list[Edge]],
                  zero_initiated: bool = True,
                  zero_terminated: bool = True
                  ) -> None:
        """Add edges to each state, as specified by the edges tuple. """
        ending_state_labels: set[int]
        if zero_initiated:
            starting_state_labels: set[int] = {0}

            # Loop over the number of stages equal to the memory size, as it takes that many stages
            # to fully develop the Trellis.
            for i in range(self.memory_size):
                state: State = self.stages[0].states[0]
                stage: Stage = self.stages[i]
                for state_label in starting_state_labels:
                    state = stage.states[state_label]
                    state.edges = deepcopy(edges[state.label])

                ending_state_labels = {e.to_state for e in state.edges if e.to_state is not None}
                starting_state_labels = ending_state_labels

        start: int = self.memory_size if zero_initiated else 0
        end: int = len(self.stages) - 1 - self.memory_size if zero_terminated else -1

        for stage in self.stages[start:end]:
            for label, state in enumerate(stage.states):
                state.edges = [copy(e) for e in edges[label]]

                # See Edge __lt__ method comment
                # state.edges.sort()

        if not zero_terminated:
            return

        starting_state_labels = set({s.label for s in self.stages[end].states})
        ending_state_labels = set()
        # Since there is one more stage than time steps, the last stage has no
        # edges. There is only one state with beta value.
        for stage in self.stages[end:-1]:
            for state_label in starting_state_labels:
                state_edges: list[Edge] = list(filter(lambda e: e.edge_input == 0, edges[state_label]))
                stage.states[state_label].edges = deepcopy(state_edges)
                ending_state_labels = ending_state_labels.union(
                    {e.to_state for e in state_edges if e.to_state is not None})

            starting_state_labels = ending_state_labels
