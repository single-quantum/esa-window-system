from copy import copy, deepcopy


class Edge:

    __slots__ = ('from_state', 'to_state', 'edge_input', 'edge_output', 'gamma', 'lmbda', 'hamming_distance')

    def __init__(self):

        self.from_state = None
        self.to_state = None
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
            edge_input: int | list[int],
            edge_output: tuple,
            gamma: float | None):

        self.from_state = from_state
        self.to_state = to_state
        self.edge_input = edge_input
        self.edge_output = edge_output
        self.gamma = gamma

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
    def __init__(self, num_states, num_input_bits, time_step):
        self.time_step = time_step

        # Initiate states (States here do not have edges yet)
        self.states = tuple(State(i, 2**num_input_bits) for i in range(num_states))

    def __lt__(self, other):
        if not self.time_step:
            return
        return self.time_step < other.time_step


class Trellis:
    def __init__(
            self,
            memory_size: int,
            num_output_bits: int,
            time_steps: int,
            edges,
            num_input_bits: int = 1):

        self.memory_size = memory_size
        self.num_states = 2**memory_size

        # The time_steps + 1-th stage has all states, but no edges. This is needed to properly calculate beta
        self.stages = tuple(
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
        if zero_initiated:
            starting_state_labels: set[int] = {0}

            state: State
            for i in range(self.memory_size):
                stage: Stage = self.stages[i]
                for state_label in starting_state_labels:
                    state = stage.states[state_label]
                    state.edges = deepcopy(edges[state.label])

                ending_state_labels = {e.to_state for e in state.edges}
                starting_state_labels = ending_state_labels

        start = self.memory_size if zero_initiated else 0
        end = len(self.stages) - 1 - self.memory_size if zero_terminated else None

        for stage in self.stages[start:end]:
            for label, state in enumerate(stage.states):
                state.edges = [copy(e) for e in edges[label]]

                # See Edge __lt__ method comment
                # state.edges.sort()

        if not zero_terminated:
            return

        starting_state_labels = set({s.label for s in self.stages[end].states})
        ending_state_labels: set[int] = set()
        # Since there is one more stage than time steps, the last stage has no
        # edges. There is only one state with beta value.
        for stage in self.stages[end:-1]:
            for state_label in starting_state_labels:
                state_edges: list[Edge] = list(filter(lambda e: e.edge_input == 0, edges[state_label]))
                stage.states[state_label].edges = deepcopy(state_edges)
                ending_state_labels = ending_state_labels.union({e.to_state for e in state_edges})

            starting_state_labels = ending_state_labels
