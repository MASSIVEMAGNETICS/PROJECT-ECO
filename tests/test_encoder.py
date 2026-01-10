from victor_cognitive_river import FractalEncoder


def test_encoder_dimension_and_range():
    enc = FractalEncoder(dim=16)
    v = enc.encode("hello world")
    assert len(v) == 16
    assert all(-1.0 <= x <= 1.0 for x in v)


def test_encoder_deterministic():
    enc = FractalEncoder(dim=8)
    a = enc.encode("same string")
    b = enc.encode("same string")
    assert a == b
