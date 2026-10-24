from vision import coolbench


def _run(label, step_ms, clip_s):
    return {"label": label, "opencv": "5.0.0", "machine": "aarch64", "cpus": 4, "kleidicv": True, "frames": 60,
            "steps": {"motion": step_ms}, "timeline": {"ring_a": {"s": clip_s, "events": 2}}}


def test_compare_reports_speedup_of_second_build_over_first():
    md = coolbench.compare(_run("pip", 200.0, 6.0), _run("cool", 100.0, 4.0))
    assert "| motion | 200.0 | 100.0 | 2.00x |" in md
    assert "| ring_a | 6.0 | 4.0 | 1.50x | 2 / 2 |" in md
    assert "**1.50x**" in md
