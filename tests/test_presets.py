import argparse

import pytest

from lars import presets
from lars.cli import _apply_preset, build_parser


def test_preset_fills_unset_flags_and_explicit_flags_win():
    args = build_parser().parse_args(["serve", "--preset", "30b"])
    _apply_preset(args)
    preset = presets.PRESETS["30b"]
    assert (args.backend, args.model, args.adapter) == (preset.backend, preset.model, list(preset.adapters))
    assert args.calibration == list(preset.calibrations)
    args = build_parser().parse_args(["serve", "--preset", "30b", "--adapter", "checkpoints/mine"])
    _apply_preset(args)
    assert args.adapter == ["checkpoints/mine"] and args.calibration is None


def test_unknown_preset_is_refused():
    with pytest.raises(SystemExit, match="unknown preset"):
        _apply_preset(argparse.Namespace(preset="nope"))


def test_hub_ids_download_and_local_paths_do_not(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    calls = []

    def snapshot(repo_id, allow_patterns=None, local_files_only=False):
        calls.append((repo_id, allow_patterns))
        return "/cache/repo"

    def download(repo_id, filename, local_files_only=False):
        calls.append((repo_id, filename))
        return "/cache/c.json"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert presets.resolve_adapter(str(tmp_path)) == str(tmp_path)
    assert presets.resolve_adapter("org/adapter") == "/cache/repo"
    assert presets.resolve_adapter("org/repo/30b-a3b") == "/cache/repo/30b-a3b"
    local = tmp_path / "cal.json"
    local.write_text("{}")
    assert presets.resolve_calibration(str(local)) == str(local)
    assert presets.resolve_calibration("org/repo/4b/lars-calibration.json") == "/cache/c.json"
    assert calls == [("org/adapter", None), ("org/repo", ["30b-a3b/*"]), ("org/repo", "4b/lars-calibration.json")]


def test_a_model_in_a_repo_folder_is_downloaded_and_a_plain_repo_is_left_to_the_backend(monkeypatch):
    monkeypatch.setattr(presets, "resolve_hub", lambda value, cached_only=False: f"/cache/{value}")
    assert presets.resolve_model("moedex/lars/4b") == "/cache/moedex/lars/4b"
    assert presets.resolve_model(presets.M4) == presets.M4


def test_the_4b_preset_is_a_fused_model_with_its_calibrator():
    args = build_parser().parse_args(["serve", "--preset", "4b"])
    _apply_preset(args)
    assert args.model == "moedex/lars/4b" and args.adapter == []
    assert args.calibration == [f"moedex/lars/4b/{presets.CALIBRATION_FILE}"]
