#!/usr/bin/env python3
"""
analyze_run.py — summary of a tools/record_run_v2.py CSV: wrist distance and its source, arm command
offsets and speed, hand command speed and limits. tools/run_teleop_real.sh prints it after a run.

Usage (with .venv-arm):  tools/analyze_run.py data/runs/<tag>/record.csv
"""
import sys, collections, json, re
import numpy as np
from pathlib import Path
import os
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, os.path.expanduser(os.environ.get("ALLEGRO_TELEOP_DIR", "~/allegro_vision_teleop")))
rows = collections.defaultdict(list)
for line in open(sys.argv[1]):
    k, t, *v = line.rstrip("\n").split(",", 2)
    rows[k].append((float(t), v[0] if v else ""))
def arr(k):
    r = rows.get(k, [])
    if not r: return np.zeros(0), np.zeros((0, 1))
    return np.array([t for t, _ in r]), np.array([[float(x) for x in v.split(",")] for _, v in r])
from safety_utils import get_joint_limits, CONTROLLER_JOINT_ORDER
for side in ("left", "right"):
    print(f"== {side}")
    t, w = arr(f"{side}_wrist")
    if len(t):
        print(f"  wrist: {len(t)} samples, {len(t)/(t[-1]-t[0]):.1f} Hz, z p5/p50/p95 = {np.percentile(w[:,2],[5,50,95]).round(3).tolist()} m, gaps>0.3s: {int((np.diff(t)>0.3).sum())}")
    src = collections.Counter()
    for _, d in rows.get(f"{side}_diag", []):
        m = re.search(r'"source":\s*"(\w+)"', d) or re.search(r"source[=:]\s*(\w+)", d)
        src[m.group(1) if m else "?"] += 1
    n = sum(src.values()) or 1
    print("  wrist distance source:", {k: f"{100*v/n:.0f}%" for k, v in src.most_common()})
    t, q = arr(f"{side}_cmd")
    if len(t):
        d = np.degrees(np.abs(q - q[0]).max(0)); sp = np.abs(np.diff(q, axis=0)).max(1) / np.maximum(np.diff(t), 1e-3)
        print(f"  arm cmd: max offset from start deg {d.round(1).tolist()} (box 17.2), speed p95/max {np.percentile(sp,95):.2f}/{np.percentile(sp,99.9):.2f} rad/s")
        from arm.ik_v2 import DiffIK
        ik = DiffIK(); p = np.array([ik.fk(x)[0] for x in q[::10]]); 
        print(f"  arm EEF: moved up to {1000*np.linalg.norm(p-p[0],axis=1).max():.0f} mm from start (cap 70); range mm xyz {(1000*(p.max(0)-p.min(0))).round(0).tolist()}")
    t, h = arr(f"{side}_hand_cmd")
    if len(t):
        lim = get_joint_limits(side); lo = np.array([lim[j][0] for j in CONTROLLER_JOINT_ORDER]); hi = np.array([lim[j][1] for j in CONTROLLER_JOINT_ORDER])
        out = int(((h < lo - 1e-6) | (h > hi + 1e-6)).any(1).sum())
        sp = np.abs(np.diff(h, axis=0)) / np.maximum(np.diff(t), 5e-3)[:, None]
        print(f"  hand cmd: {len(t)/(t[-1]-t[0]):.0f} Hz, outside limits {out}, joint speed p95/p99/max {np.percentile(sp.max(1),95):.1f}/{np.percentile(sp.max(1),99):.1f}/{sp.max():.1f} rad/s, range of motion (max over joints) {np.degrees((h.max(0)-h.min(0)).max()):.0f} deg")
        tm, hm = arr(f"{side}_hand_meas")
        if len(tm):
            i = np.searchsorted(t, tm).clip(0, len(t)-1); print(f"  hand meas-cmd: p50/p95 {np.degrees(np.percentile(np.abs(hm-h[i]).max(1),[50,95])).round(1).tolist()} deg")
