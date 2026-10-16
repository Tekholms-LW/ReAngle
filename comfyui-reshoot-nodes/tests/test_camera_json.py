import json
import numpy as np
import pytest
from reshoot_nodes.camera import camera_path, PRESETS
from reshoot_nodes.camera_json import (look_at_transform, transform_to_keyframe, path_from_json, preset_to_json,
                                       parse_path, MAX_TRAVEL)


def rot_angle(A, B):
    R = A[:3, :3].T @ B[:3, :3]
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def centre(T):
    return -T[:3, :3].T @ T[:3, 3]


def test_identity_keyframe():
    assert np.allclose(look_at_transform([0, 0, 0], [0, 0, 3]), np.eye(4))


@pytest.mark.parametrize("p", PRESETS)
def test_roundtrip_keyframe(p):
    for T in camera_path(p, 25, 5, 3.0):
        pos, look = transform_to_keyframe(T, 3.0)
        T2 = look_at_transform(pos, look)
        assert rot_angle(T, T2) < 1e-6 and np.allclose(centre(T), centre(T2), atol=1e-9)


@pytest.mark.parametrize("p", PRESETS)
def test_json_matches_preset(p):
    n = 41
    ref = camera_path(p, 20, n, 3.0, ease="smooth")
    got = path_from_json(json.dumps(preset_to_json(p, 20, 3.0)), n)["poses"]
    for a, b in zip(ref, got):
        assert rot_angle(a, b) < 0.5                                   # degrees
        assert np.linalg.norm(centre(a) - centre(b)) < 0.01 * 3.0     # 1% of pivot


def test_interpolation_hits_keyframes_and_is_smooth():
    spec = {"pivot_depth": 3, "ease": "linear", "keyframes": [
        {"t": 0, "pos": [0, 0, 0], "look": [0, 0, 3]},
        {"t": 0.5, "pos": [0.4, 0, 0], "look": [0, 0, 3]},
        {"t": 1, "pos": [0.4, -0.4, 0.2], "look": [0, 0, 3]}]}
    poses = path_from_json(spec, 21)["poses"]
    assert np.allclose(poses[0], np.eye(4))
    assert np.allclose(centre(poses[10]), [0.4, 0, 0], atol=1e-6)
    assert np.allclose(centre(poses[20]), [0.4, -0.4, 0.2], atol=1e-6)
    steps = np.linalg.norm(np.diff([centre(T) for T in poses], axis=0), axis=1)
    assert steps.max() < 3 * steps.mean()                              # no jumps


def test_clamps_angle_and_travel():
    spec = {"pivot_depth": 3, "keyframes": [{"t": 0, "pos": [0, 0, 0], "look": [0, 0, 3]},
                                            {"t": 1, "pos": [5, 0, 0], "look": [-20, 0, 3]}]}
    out = path_from_json(spec, 9)
    T = out["poses"][-1]
    assert out["clamped_frames"] > 0
    assert rot_angle(np.eye(4), T) <= 30.0 + 1e-6
    assert np.linalg.norm(centre(T)) <= MAX_TRAVEL * 3 + 1e-9


@pytest.mark.parametrize("bad", [
    {"keyframes": []},
    {"keyframes": [{"t": 0, "pos": [1, 0, 0], "look": [0, 0, 3]}]},
    {"keyframes": [{"t": 0, "pos": [0, 0, 0], "look": [0, 0, 3]}, {"t": 0, "pos": [0, 0, 0], "look": [0, 0, 3]}]},
])
def test_rejects_invalid(bad):
    with pytest.raises(ValueError):
        parse_path(bad)


def test_freeze_flag_and_node():
    import importlib.util
    spec = preset_to_json("orbit_left", 15); spec["freeze_time"] = True
    assert path_from_json(spec, 5)["freeze_time"]
    if importlib.util.find_spec("torch"):
        import torch
        from reshoot_nodes.nodes import CameraPathFromJSON, DepthReproject
        path, nclamp = CameraPathFromJSON().build(json.dumps(spec), 5)
        v, u = np.mgrid[0:24, 0:32]
        frames = np.stack([np.full((24, 32, 3), i / 5, np.float32) for i in range(5)])
        w, m, _ = DepthReproject().warp(torch.from_numpy(frames), torch.full((5, 24, 32, 3), 0.5), path,
                                        60.0, 1.0, 10.0, True)
        valid = m.numpy()[-1] == 0
        assert np.allclose(w.numpy()[-1][valid], 0.0)                  # last frame shows source frame 0
