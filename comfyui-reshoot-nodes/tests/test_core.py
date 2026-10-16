import importlib.util
import numpy as np
import pytest
from reshoot_nodes.camera import camera_path, intrinsics, PRESETS
from reshoot_nodes.reproject import warp_frame, warp_video, depth_from_relative

H, W = 48, 64


def checker():
    v, u = np.mgrid[0:H, 0:W]
    img = np.stack([u / W, v / H, ((u // 8 + v // 8) % 2).astype(float)], -1).astype(np.float32)
    return img


def test_identity_is_lossless():
    img, d = checker(), np.full((H, W), 4.0)
    w, m, wd = warp_frame(img, d, intrinsics(W, H), np.eye(4))
    assert np.allclose(w, img) and m.sum() == 0 and np.allclose(wd, 4.0)


@pytest.mark.parametrize("p", PRESETS)
def test_paths_start_at_identity_and_are_rigid(p):
    path = camera_path(p, 30, 9)
    assert path.shape == (9, 4, 4) and np.allclose(path[0], np.eye(4))
    for T in path:
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1)


def test_orbit_angle_capped_at_30():
    T = camera_path("orbit_left", 90, 5)[-1]
    ang = np.degrees(np.arccos((np.trace(T[:3, :3]) - 1) / 2))
    assert np.isclose(ang, 30.0)


def test_orbit_keeps_pivot_fixed():
    T = camera_path("orbit_right", 20, 3, pivot_depth=5.0)[-1]
    assert np.allclose(T @ np.array([0, 0, 5.0, 1]), [0, 0, 5.0, 1])


def test_truck_shifts_plane_and_opens_hole_on_edge():
    img, z = checker(), 4.0
    K = intrinsics(W, H)
    T = np.eye(4); T[0, 3] = -0.25     # scene moves left => camera trucks right
    w, m, _ = warp_frame(img, np.full((H, W), z), K, T)
    shift = int(round(K[0, 0] * 0.25 / z))
    assert shift > 0
    assert np.allclose(w[:, : W - shift], img[:, shift:])
    assert m[:, W - shift:].all() and m[:, : W - shift].sum() == 0


def test_occlusion_near_wins():
    img = np.zeros((H, W, 3), np.float32); img[:, W // 2 - 4: W // 2 + 4] = 1.0
    d = np.full((H, W), 8.0); d[:, W // 2 - 4: W // 2 + 4] = 2.0   # near bright bar on far bg
    T = np.eye(4); T[0, 3] = 0.3
    w, m, wd = warp_frame(img, d, intrinsics(W, H), T)
    bar = (wd > 0) & (wd < 4.0)
    assert bar.any() and np.allclose(w[bar], 1.0)
    assert m.sum() > 0                                     # disocclusion holes appear


def test_push_magnifies():
    img = checker(); K = intrinsics(W, H)
    T = camera_path("push", 30, 2, pivot_depth=4.0)[-1]
    w, m, wd = warp_frame(img, np.full((H, W), 4.0), K, T)
    assert wd[H // 2, W // 2] < 4.0 and m.sum() > 0       # closer + edges uncovered by splat gaps


def test_warp_video_resamples_path():
    frames = np.stack([checker()] * 5); depths = np.full((5, H, W), 3.0)
    w, m, wd = warp_video(frames, depths, intrinsics(W, H), camera_path("truck", 10, 9))
    assert w.shape == frames.shape and m.shape == (5, H, W) and m[0].sum() == 0


def test_depth_from_relative_inverse():
    d = depth_from_relative(np.array([0.0, 1.0]), near=1, far=10)
    assert np.allclose(d, [10, 1])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_comfy_node_wrapper():
    import torch
    from reshoot_nodes.nodes import CameraPathFromPreset, DepthReproject
    (path,) = CameraPathFromPreset().build("orbit_left", 15, 3, 3.0, False)
    imgs = torch.from_numpy(np.stack([checker()] * 3))
    depth = torch.full((3, H, W, 3), 0.5)
    w, m, wd = DepthReproject().warp(imgs, depth, path, 60.0, 1.0, 10.0, True)
    assert w.shape == imgs.shape and m.shape == (3, H, W) and wd.shape == imgs.shape


def test_fill_cracks_closes_thin_gaps_keeps_big_holes():
    from reshoot_nodes.reproject import fill_cracks
    img = checker(); hole = np.zeros((H, W), np.float32); d = np.full((H, W), 3.0)
    hole[10, 10] = 1; img[10, 10] = 0          # single-pixel crack
    hole[20:40, 30:50] = 1                     # big disocclusion
    out, m, dd = fill_cracks(img, hole, d, passes=2)
    assert m[10, 10] == 0 and np.all(out[10, 10, :2] > 0)
    assert m[25:35, 35:45].all()               # interior of big hole untouched


def test_push_cracks_filled_by_warp_video():
    frames = np.stack([checker()] * 2); depths = np.full((2, H, W), 4.0)
    path = camera_path("push", 30, 2, pivot_depth=4.0)
    _, m0, _ = warp_video(frames, depths, intrinsics(W, H), path, crack_passes=0)
    _, m2, _ = warp_video(frames, depths, intrinsics(W, H), path, crack_passes=2)
    assert m2[1].sum() < m0[1].sum()
