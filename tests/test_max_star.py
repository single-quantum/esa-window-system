from pytest import approx


from core.BCJR_decoder_functions import max_star


def test_max_star():
    assert max_star(1, 1) == approx(1.693, rel=1E-3)
