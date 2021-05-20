""" # noqa 
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa 
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa 
Author : Benjamin Blundell - k1803390@kcl.ac.uk

objs_to_json.py - Given a list of OBJ files representing our points, combine 
them into JSON animation for our analysis.

This file is used in the generate stats script. The resulting json can be used
in the jupyter notebook provided, or with the blender_vis.py script.
"""

import math
import os
import json

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OBJs to JSON")
    parser.add_argument("--path", default=".", help="Path to the obj files")
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        metavar="N",
        help="number of OBJ files to \
                        consider(default: all)",
    )
    args = parser.parse_args()

    obj_files = []
    for dirname, dirnames, filenames in os.walk(args.path):
        for filename in filenames:
            obj_extentions = ["obj", "OBJ"]
            if any(x in filename for x in obj_extentions) and "shape" in filename:
                obj_files.append(os.path.join(dirname, filename))

    final_objs = []
    # Go through and select 'limit' number of objects equally spaced
    if args.limit != -1:
        num_objs = len(obj_files)
        step_size = math.floor(num_objs / args.limit)
        for i in range(args.limit):
            final_objs.append(obj_files[i * step_size])
    else:
        final_objs = obj_files

    print("Exporting OBJ files to animation json")
    final_objs.sort()
    frameidx = 0

    animation = {}
    animation["frames"] = []

    for objpath in final_objs:
        # scene = pywavefront.Wavefront(objpath, parse=True)
        scene = {}
        scene["vertices"] = []
        with open(objpath, "r") as f:
            for line in f.readlines()[2:]:
                if "v" in line:
                    tokens = line.replace("v", "").replace("\n", "").split(" ")
                    vertex = {}
                    vertex["x"] = float(tokens[1])
                    vertex["y"] = float(tokens[2])
                    vertex["z"] = float(tokens[3])
                    scene["vertices"].append(vertex)
                frameidx += 1
        animation["frames"].append(scene)
        # if frameidx > args.limit:
        #  break

    final_json = json.dumps(animation, indent=2)
    with open(args.path + "/animation.json", "w") as f:
        f.write(final_json)
