#!/usr/bin/env python3
"""
check_xarm.py — read an xArm7's state without moving it.

Connects, prints firmware version, state, mode, error and warning codes, joint angles,
TCP pose and the TCP offset, then disconnects. Nothing here enables motors, clears
errors, changes mode or sends a motion: those are the calls that make an arm move or
let it move, and they belong to a later step that asks first.

The IP is an argument and never stored in the repository (it is public).

Usage (run with .venv-arm):
    .venv-arm/bin/python tools/check_xarm.py <ip>
"""
from __future__ import annotations

import argparse

import numpy as np
from xarm.wrapper import XArmAPI

# xArm's state codes (SDK docs, "state").
STATES = {1: "in motion", 2: "sleeping", 3: "suspended", 4: "stopping"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ip")
    args = p.parse_args()

    arm = XArmAPI(args.ip, is_radian=True)
    try:
        if not arm.connected:
            raise SystemExit(f"could not connect to {args.ip}")
        print(f"version     : {arm.version}")
        print(f"axis        : {arm.axis}")
        print(f"state       : {arm.state} ({STATES.get(arm.state, '?')})")
        print(f"mode        : {arm.mode}")
        print(f"error/warn  : {arm.error_code} / {arm.warn_code}")
        print(f"angles (deg): {np.round(np.degrees(arm.angles[:arm.axis]), 2).tolist()}")
        print(f"tcp (mm,rad): {np.round(arm.position, 3).tolist()}")
        print(f"tcp offset  : {np.round(arm.tcp_offset, 3).tolist()}")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
