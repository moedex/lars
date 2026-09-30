import argparse

import pytest

from lars import presets
from lars.cli import _apply_preset, build_parser


def test_preset_fills_unset_flags_and_explicit_flags_win():
    args = build_parser().parse_args(["serve", "--preset", "30b"])
    _apply_preset(args)
    preset = presets.PRESETS["30b"]
    assert (args.backend, args.model, args.adapter) == (preset.backend, preset.model, list(preset.adapters))
    assert args.calibration == [f"{a}/{presets.CALIBRATION_FILE}" for a in preset.adapters]
    args = build_parser().parse_args(["serve", "--preset", "30b", "--adapter", "checkpoints/mine"])
    _apply_preset(args)
    assert args.adapter == ["checkpoints/mine"] and args.calibration is None


def test_unknown_preset_is_refused():
    with pytest.raises(SystemExit, match="unknown preset"):
        _apply_preset(argparse.Namespace(preset="nope"))


def test_hub_ids_download_and_local_paths_do_not(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id: calls.append(repo_id) or "/cache/a")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo_id, filename: calls.append((repo_id, filename)) or "/cache/c.json")
    assert presets.resolve_adapter(str(tmp_path)) == str(tmp_path)
    assert presets.resolve_adapter("org/adapter") == "/cache/a"
    local = tmp_path / "cal.json"
    local.write_text("{}")
    assert presets.resolve_calibration(str(local)) == str(local)
    assert presets.resolve_calibration("org/adapter/lars-calibration.json") == "/cache/c.json"
    assert calls == ["org/adapter", ("org/adapter", "lars-calibration.json")]
