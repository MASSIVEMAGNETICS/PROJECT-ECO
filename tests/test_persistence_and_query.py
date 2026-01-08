import os
import tempfile
from victor_cognitive_river import CognitiveRiver, FractalEncoder


def test_absorb_persist_and_reload():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        enc = FractalEncoder()
        river = CognitiveRiver(persistence_path=path, encoder=enc)
        # Absorb two items with sufficient similarity to pass threshold
        r1 = river.absorb("first item about cats and their behavior")
        assert r1["status"] == "ABSORBED"
        r2 = river.absorb("first item about cats and their behavior patterns")
        assert r2["status"] == "ABSORBED"
        # Query for cats
        results = river.query("cats")
        assert isinstance(results, list)
        # Reload a fresh river from the same file
        river2 = CognitiveRiver(persistence_path=path, encoder=enc)
        assert river2.status()["river_depth"] >= 2
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
