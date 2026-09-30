#!/usr/bin/env python
"""Derive cfg/task/base/hdmi-base-h1_2.yaml from hdmi-base.yaml.

H1-2 has a single `torso_joint` where G1 has waist_yaw/roll/pitch. Four places in
hdmi-base.yaml hard-code the G1 pattern, and one of them (`action.action_scaling`)
is a **dict** -- Hydra merges dicts, so a task-level override cannot remove the
stale waist keys, and `JointPosition` rejects any regex that matches nothing.
The only clean fix is a base the H1-2 tasks inherit instead.

Re-run this whenever hdmi-base.yaml changes.
"""
import pathlib

BASE = pathlib.Path("/home/mchang344/mj_ws/simbench/HDMI/cfg/task/base")

def main():
    s = (BASE / "hdmi-base.yaml").read_text()
    s = s.replace("""    waist_roll_joint: 0.44
    waist_pitch_joint: 0.44
    waist_yaw_joint: 0.55
""", """    torso_joint: 0.55
""")
    s = s.replace('      "waist_.*_joint", \n', '      "torso_joint", \n')
    s = s.replace('["waist_.*_joint", ', '["torso_joint", ')
    leftover = [l for l in s.split("\n") if "waist" in l]
    if leftover:
        raise SystemExit(f"unhandled waist references, update this script:\n" +
                         "\n".join(leftover))
    header = ("# @package task\n"
              "# Auto-derived from hdmi-base.yaml for Unitree H1-2, which has a single\n"
              "# torso_joint where G1 has waist_yaw/roll/pitch. Regenerate with\n"
              "# pipeline/scripts/make_h1_2_base.py if hdmi-base.yaml changes.\n")
    s = s.replace("# @package task\n", "", 1)
    out = BASE / "hdmi-base-h1_2.yaml"
    out.write_text(header + s)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
