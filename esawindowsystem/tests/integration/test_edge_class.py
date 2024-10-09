from esawindowsystem.core.trellis import Edge


def test_set_edge_output_label_3_output_bits_value_0_no_bpsk():
    e = Edge()
    e.set_edge(0, 0, (1, 1, 1), (0, 0, 0), None)
    assert e.edge_input_label == 7
    assert e.edge_output_label == 0


def test_set_edge_output_label_3_output_bits_value_4_no_bpsk():
    e = Edge()
    e.set_edge(0, 0, 0, (1, 0, 0), None)
    assert e.edge_output_label == 4


def test_set_edge_output_label_3_output_bits_value_0_with_bpsk():
    e = Edge()
    e.set_edge(0, 0, 1, (-1, -1, -1), None)
    assert e.edge_input_label == 1
    assert e.edge_output_label == 0


def test_set_edge_output_label_3_output_bits_value_4_with_bpsk():
    e = Edge()
    e.set_edge(0, 0, 0, (1, -1, -1), None)
    assert e.edge_output_label == 4


def test_set_egde_output_label_2_output_bits_value_2():
    e = Edge()
    e.set_edge(0, 0, 0, (1, 0), None)
    assert e.edge_output_label == 2
