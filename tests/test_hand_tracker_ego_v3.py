"""vision/hand_tracker_ego_v3.py, depth_filter_v2.py and rgbd_source.ClipRGBD on synthetic input.

No camera and no MediaPipe model: the detector is a stand-in that hands back a
synthetic hand, projected through known intrinsics, and the depth map is that hand's
palm plane.
"""
import json

import numpy as np
import pytest

HAS_VISION = False
if int(np.__version__.split(".")[0]) < 2:
    try:
        import cv2
        from scipy.spatial.transform import Rotation

        from vision.camera import CameraCalibration
        from vision.depth_filter import DEFAULT_SIGMA_RAY, RayKalman
        from vision.depth_filter_v2 import RayKalmanV2
        from vision.depth_wrist import PALM, DepthFusion, FusedWrist, palm_depth
        from vision.hand_tracker import RawHand, _PoseFilter
        from vision.hand_tracker_ego_v3 import SIGMA_RAY_SENSOR, DepthEgoHandTracker, DepthFusionV1
        from vision.rgbd_source import ClipRGBD
        from vision.wrist_pose import solve_wrist_pose

        HAS_VISION = True
    except ImportError:
        HAS_VISION = False

pytestmark = pytest.mark.skipif(not HAS_VISION, reason="needs OpenCV and MediaPipe (.venv-vision)")

W, H = 640, 480
K = np.array([[383.76, 0.0, 321.35], [0.0, 383.36, 245.01], [0.0, 0.0, 1.0]])

# Canonical right hand in metres: x out of palm, y towards thumb, z along middle finger
# (as in tests/test_hand_scale.py).
RIGHT_HAND = np.array([
    [0.000,  0.000, 0.000],
    [0.010,  0.020, 0.025], [0.020,  0.040, 0.045], [0.030,  0.055, 0.065], [0.040,  0.065, 0.080],
    [0.000,  0.022, 0.090], [0.010,  0.025, 0.130], [0.020,  0.026, 0.152], [0.030,  0.027, 0.170],
    [0.000,  0.000, 0.092], [0.010,  0.000, 0.135], [0.010,  0.000, 0.160], [0.010,  0.000, 0.180],
    [0.000, -0.020, 0.090], [0.010, -0.022, 0.128], [0.020, -0.023, 0.150], [0.030, -0.024, 0.168],
    [0.000, -0.038, 0.082], [0.010, -0.042, 0.112], [0.018, -0.044, 0.128], [0.025, -0.045, 0.142],
])
# Back of the hand towards the camera, fingers up the image and away, as seen from above and behind.
# Columns: palm normal away from the camera, y, fingers up the image; then tilted away.
# (Built only where scipy imported: the arm venv collects this module too, then skips it.)
R_HAND = Rotation.from_euler("xyz", [-40.0, 10.0, 0.0], degrees=True).as_matrix() @ np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]) if HAS_VISION else None
# MediaPipe's world landmarks are this much smaller than the real hand, so PnP places it
# this much too close.
MP_SHRINK = 1.2


def calibration():
    return CameraCalibration((W, H), K, np.zeros(5))


def hand(wrist):
    """(true hand in the camera frame (21, 3), its RawHand as MediaPipe would report it)."""
    world = RIGHT_HAND - RIGHT_HAND.mean(axis=0)
    true = wrist + (world - world[0]) @ R_HAND.T
    uvw = true @ K.T
    image = np.column_stack([uvw[:, 0] / uvw[:, 2] / W, uvw[:, 1] / uvw[:, 2] / H, np.zeros(21)])
    return true, RawHand(image, world / MP_SHRINK, "Right", 0.9)


def plane_depth(true):
    """(H, W) depth of the true palm plane on every pixel's ray, metres."""
    palm = true[list(PALM)]
    centre = palm.mean(axis=0)
    normal = np.linalg.svd(palm - centre)[2][-1]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], axis=-1)
    return ((centre @ normal) / (rays @ normal)).astype(np.float32)


class Detector:
    """Stand-in for MediaPipe: one list of RawHand per call."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.closed = False

    def __call__(self, rgb, t):
        return self.frames.pop(0)

    def close(self):
        self.closed = True


class ScriptedFusion:
    """Fusion that returns PnP times a fixed scale with scripted sources, and counts resets."""

    made = []

    def __init__(self, sources=("depth",), rotation=None):
        self.sources = list(sources)
        self.rotation = rotation
        self.resets = 0
        self.calls = 0
        ScriptedFusion.made.append(self)

    def reset(self):
        self.resets += 1

    def __call__(self, pose, image_landmarks, world_landmarks, depth_m, camera_matrix, side):
        source = self.sources[self.calls % len(self.sources)]
        self.calls += 1
        f = FusedWrist(pose.position * 1.1, 1.1, source, 1.1)
        if self.rotation is not None:
            object.__setattr__(f, "rotation", self.rotation)
        return f


FRAME = np.zeros((H, W, 3), np.uint8)
WRIST = np.array([0.08, 0.02, 0.36])


def wobbling(n, seed=0):
    """n frames of one hand whose wrist wobbles a few mm, as (true hands, detector frames)."""
    rng = np.random.default_rng(seed)
    hands = [hand(WRIST + rng.normal(0, 0.004, 3)) for _ in range(n)]
    return [h[0] for h in hands], [[h[1]] for h in hands]


def test_sensor_depth_puts_the_wrist_where_the_hand_is():
    true, raw = hand(WRIST)
    depth = plane_depth(true)
    tracker = DepthEgoHandTracker(calibration(), detector=Detector([[raw], [raw]]))

    out = tracker.process(FRAME, 0.0, depth)["right"]
    assert out.fused.source == "depth"
    assert out.sigma_ray == SIGMA_RAY_SENSOR
    assert out.fused.scale == pytest.approx(MP_SHRINK, rel=1e-3)
    assert out.fused.pixels >= 30
    assert np.allclose(out.wrist.position, true[0], atol=5e-4)

    # Without depth: PnP's distance, too close by MP_SHRINK, at the M4 ray noise.
    plain = DepthEgoHandTracker(calibration(), detector=Detector([[raw]])).process(FRAME, 0.0)["right"]
    assert plain.fused.source == "pnp" and plain.sigma_ray == DEFAULT_SIGMA_RAY
    assert np.allclose(plain.wrist.position, true[0] / MP_SHRINK, atol=5e-4)


def test_without_depth_it_is_the_v2_ray_filter():
    """No depth: RayKalman with the M4 defaults on raw PnP and One Euro rotation, as RayEgoHandTracker."""
    _, frames = wobbling(20)
    tracker = DepthEgoHandTracker(calibration(), detector=Detector(frames))
    kf, rot = RayKalman(), _PoseFilter()
    for i, (raw,) in enumerate(frames):
        t = i / 30.0
        out = tracker.process(FRAME, t)["right"]
        pose = solve_wrist_pose(raw.world_landmarks, raw.image_landmarks, (W, H), K, "right")
        assert np.allclose(out.wrist.position, kf(pose.position, t), atol=1e-12)
        assert np.allclose(out.wrist.rotation, rot(pose, t).rotation, atol=1e-12)


def test_ray_noise_follows_the_source_frame_by_frame():
    """As tools/eval_depth_wrist.run_kalman: sigma_ray set on every frame from that frame's source."""
    sources = ["depth", "depth", "hold", "depth", "pnp", "hold", "depth"] * 3
    _, frames = wobbling(len(sources), seed=1)
    ScriptedFusion.made.clear()
    tracker = DepthEgoHandTracker(calibration(), detector=Detector(frames),
                                  fusion_factory=lambda: ScriptedFusion(sources))
    kf = RayKalman()
    for i, ((raw,), source) in enumerate(zip(frames, sources)):
        t = i / 30.0
        out = tracker.process(FRAME, t)["right"]
        expected_sigma = SIGMA_RAY_SENSOR if source == "depth" else DEFAULT_SIGMA_RAY
        assert out.fused.source == source and out.sigma_ray == expected_sigma
        pose = solve_wrist_pose(raw.world_landmarks, raw.image_landmarks, (W, H), K, "right")
        kf.sigma_ray = expected_sigma
        assert np.allclose(out.wrist.position, kf(pose.position * 1.1, t), atol=1e-12)


def test_small_sensor_ray_noise_follows_a_depth_step_faster():
    """The point of the per-frame sigma: a real move along the ray is followed within a few frames."""
    def run(source):
        frames = [[hand(WRIST)[1]]] * 10 + [[hand(WRIST * 1.15)[1]]] * 12
        tracker = DepthEgoHandTracker(calibration(), detector=Detector(frames),
                                      fusion_factory=lambda: ScriptedFusion([source]))
        return [np.linalg.norm(tracker.process(FRAME, i / 30.0)["right"].wrist.position) for i in range(len(frames))]

    target = np.linalg.norm(WRIST * 1.15 / MP_SHRINK * 1.1)  # PnP places it MP_SHRINK too close
    depth, hold = abs(run("depth")[-1] - target), abs(run("hold")[-1] - target)
    assert depth < 0.002 and hold > 3 * depth


def test_fusion_rotation_replaces_pnp_rotation():
    alt = Rotation.from_euler("z", 30, degrees=True).as_matrix() @ R_HAND
    _, raw = hand(WRIST)
    tracker = DepthEgoHandTracker(calibration(), detector=Detector([[raw]]),
                                  fusion_factory=lambda: ScriptedFusion(rotation=alt))
    out = tracker.process(FRAME, 0.0)["right"]
    assert np.allclose(out.wrist.rotation, alt, atol=1e-9)


def test_fusion_is_made_once_per_side_and_reset_after_a_gap():
    _, frames = wobbling(5)
    ScriptedFusion.made.clear()
    tracker = DepthEgoHandTracker(calibration(), detector=Detector(frames), fusion_factory=ScriptedFusion)
    assert len(ScriptedFusion.made) == 2
    for t in (0.0, 0.033, 0.066, 0.5, 0.533):  # one 0.43 s gap, longer than RESET_GAP_S
        tracker.process(FRAME, t)
    right = [f for f in ScriptedFusion.made if f.calls]
    assert len(right) == 1 and right[0].calls == 5 and right[0].resets == 1
    tracker.close()
    assert tracker._detect.closed


def test_depth_must_be_aligned_to_the_frame():
    _, raw = hand(WRIST)
    tracker = DepthEgoHandTracker(calibration(), detector=Detector([[raw]]))
    with pytest.raises(ValueError, match="align"):
        tracker.process(FRAME, 0.0, np.zeros((240, 424), np.float32))


def test_v1_fusion_is_m5_depth_fusion():
    true, raw = hand(WRIST)
    depth = plane_depth(true) * np.float32(1.03)  # the sensor and PnP disagree a little
    pose = solve_wrist_pose(raw.world_landmarks, raw.image_landmarks, (W, H), K, "right")
    ours, theirs = DepthFusionV1(), DepthFusion()
    for _ in range(3):
        a = ours(pose, raw.image_landmarks, raw.world_landmarks, depth, K, "right")
        sensor = palm_depth(depth, raw.image_landmarks)
        b = theirs(pose.position, pose.rotation, raw.world_landmarks, sensor, K, "right")
        assert np.array_equal(a.position, b.position) and (a.scale, a.source, a.measured) == (b.scale, b.source, b.measured)
        assert a.pixels == sensor.n
    none = ours(pose, raw.image_landmarks, raw.world_landmarks, None, K, "right")
    assert none.source == "hold" and none.pixels is None
    ours.reset()
    assert ours(pose, raw.image_landmarks, raw.world_landmarks, None, K, "right").source == "pnp"


def test_ray_kalman_v2_sigma_is_per_measurement():
    rng = np.random.default_rng(2)
    zs = WRIST + rng.normal(0, 0.01, (30, 3))
    sigmas = rng.choice([0.005, 0.045], 30)
    ours, ref = RayKalmanV2(), RayKalman()
    for i, (z, s) in enumerate(zip(zs, sigmas)):
        ref.sigma_ray = s
        assert np.array_equal(ours(z, i / 30.0, s), ref(z, i / 30.0))
        assert ours.sigma_ray == DEFAULT_SIGMA_RAY  # its own value is left as it was
    plain, ref = RayKalmanV2(), RayKalman()
    for i, z in enumerate(zs):
        assert np.array_equal(plain(z, i / 30.0), ref(z, i / 30.0))


def write_clip(path, n=3, rows=None):
    path.mkdir()
    video = cv2.VideoWriter(str(path / "color.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (W, H))
    for i in range(n):
        video.write(np.full((H, W, 3), 40 * i, np.uint8))
    video.release()
    depth = np.stack([np.full((H, W), 300 + i, np.uint16) for i in range(rows or n)])
    np.save(path / "depth.npy", depth)
    with (path / "frames.jsonl").open("w") as f:
        for i in range(n):
            f.write(json.dumps({"i": i, "t": 0.05 + i / 30.0, "phase": "still"}) + "\n")
    meta = {"color": {"width": W, "height": H, "fx": 383.0, "fy": 384.0, "cx": 320.0, "cy": 240.0},
            "depth_scale": 0.001}
    (path / "meta.json").write_text(json.dumps(meta))


def test_clip_source_gives_frames_metric_depth_and_recorded_times(tmp_path):
    write_clip(tmp_path / "clip")
    src = ClipRGBD(tmp_path / "clip")
    assert src.calibration.image_size == (W, H)
    assert np.allclose(src.calibration.camera_matrix, [[383.0, 0, 320.0], [0, 384.0, 240.0], [0, 0, 1]])
    for i in range(3):
        ok, frame, depth = src.read()
        assert ok and frame.shape == (H, W, 3)
        assert depth.dtype == np.float32 and np.allclose(depth, (300 + i) * 0.001)
        assert src.t == pytest.approx(0.05 + i / 30.0)
    assert src.read() == (False, None, None)
    src.rewind()
    assert src.read()[0] and src.index == 0
    src.release()

    ok, _, depth = ClipRGBD(tmp_path / "clip", depth=False).read()
    assert ok and depth is None


def test_clip_source_refuses_mismatched_depth(tmp_path):
    write_clip(tmp_path / "clip", n=3, rows=2)
    with pytest.raises(ValueError, match="depth.npy"):
        ClipRGBD(tmp_path / "clip")
