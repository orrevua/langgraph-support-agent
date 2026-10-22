import sys

from supportagent import demo


def test_demo_runs_offline(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["demo"])
    monkeypatch.setenv("STORAGE", "memory")
    demo.main()
    out = capsys.readouterr().out
    assert "PAUSED - human approval required" in out
    assert "('RF-002', 1150.0, 'maria.ops')" in out  # edited amount was applied
    assert "rejected" in out
