from esawindowsystem.core.trellis import Trellis
from esawindowsystem.core.utils import generate_outer_code_edges


def test_initialize_trellis():
    memory_size = 2
    num_output_bits = 3
    edge_template = generate_outer_code_edges(memory_size, False)
    time_steps = 5
    tr = Trellis(memory_size, num_output_bits, time_steps, edge_template, 1)
    assert len(tr.stages) == 6
    assert len(tr.stages[0].states) == 2**memory_size


def test_set_edges_trellis_zero_initiated():
    memory_size = 2
    num_output_bits = 3
    edge_template = generate_outer_code_edges(memory_size, False)
    time_steps = 5
    tr = Trellis(memory_size, num_output_bits, time_steps, edge_template, 1)
    tr.set_edges(edge_template, zero_initiated=True)

    # First state in first stage should be only state with edges
    stage = tr.stages[0]
    assert len(stage.states[0].edges) == 2
    for i in range(1, 2**memory_size):
        assert len(stage.states[i].edges) == 0
    # First and third state in second stage should have two edges
    assert len(tr.stages[1].states[0].edges) == 2
    assert len(tr.stages[1].states[2].edges) == 2
    # Second and fourth state in second stage should have no edges
    assert len(tr.stages[1].states[1].edges) == 0
    assert len(tr.stages[1].states[3].edges) == 0


def test_set_edges_trellis_zero_terminated():
    memory_size = 2
    num_output_bits = 3
    edge_template = generate_outer_code_edges(memory_size, False)
    time_steps = 5
    tr = Trellis(memory_size, num_output_bits, time_steps, edge_template, 1)
    tr.set_edges(edge_template, zero_terminated=True)

    # Whether zero terminated or not, the last states should have no edges
    for i in range(4):
        assert len(tr.stages[-1].states[i].edges) == 0
    # The first and third state in the second to last stage should have only one possible transition to be zero terminated
    assert len(tr.stages[-2].states[0].edges) == 1
    assert len(tr.stages[-2].states[2].edges) == 1
