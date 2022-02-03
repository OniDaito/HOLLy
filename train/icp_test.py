"""
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/          # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/          # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

icp_test.py - Using icp along with RMSD across the animation.json
file in order to detect when the network has come up with a good
final structure and is merely rotating around what it has.

"""

from typing import List
import numpy as np
import json
import argparse
import math
from stats.simpleicp import (
    PointCloud,
    matching,
    reject,
    estimate_rigid_body_transformation,
    check_convergence_criteria,
    create_homogeneous_transformation_matrix,
)
from numba import jit
import matplotlib.pyplot as plt


def icp(
    X_fix,
    X_mov,
    correspondences=500,
    neighbors=5,
    min_planarity=0.3,
    min_change=0.01,
    max_iterations=20,
):
    pcfix = PointCloud(X_fix[:, 0], X_fix[:, 1], X_fix[:, 2])
    pcmov = PointCloud(X_mov[:, 0], X_mov[:, 1], X_mov[:, 2])
    pcfix.select_n_points(correspondences)
    sel_orig = pcfix.sel
    pcfix.estimate_normals(neighbors)
    H = np.eye(4)
    residual_distances = []

    for i in range(0, max_iterations):
        initial_distances = matching(pcfix, pcmov)
        # Todo Change initial_distances without return argument
        initial_distances = reject(pcfix, pcmov, min_planarity, initial_distances)
        dH, residuals = estimate_rigid_body_transformation(
            pcfix.x[pcfix.sel],
            pcfix.y[pcfix.sel],
            pcfix.z[pcfix.sel],
            pcfix.nx[pcfix.sel],
            pcfix.ny[pcfix.sel],
            pcfix.nz[pcfix.sel],
            pcmov.x[pcmov.sel],
            pcmov.y[pcmov.sel],
            pcmov.z[pcmov.sel],
        )

        residual_distances.append(residuals)
        pcmov.transform(dH)
        H = dH @ H
        pcfix.sel = sel_orig

        if i > 0:
            if check_convergence_criteria(
                residual_distances[i], residual_distances[i - 1], min_change
            ):

                break

        return (pcfix, pcmov, H)


@jit(nopython=True)
def distsquare(v, w):
    return (v[0] - w[0])**2 + (v[1] - w[1])**2 + (v[2] - w[2])**2


@jit(nopython=True)
def rmsd_score(fixed: List, moved: List):
    total_dist = 0

    for v in fixed:
        min_d = 1000000
        min_i = 0

        for widx in range(len(moved)):
            w = moved[widx]
            dd = distsquare(v, w)
            if dd < min_d:
                min_d = dd
                min_i = widx

        if len(moved) == 0:
            break

        del moved[min_i]
        total_dist += math.sqrt(min_d)  # removed the **2 as the range is a bit better this way

    total_dist /= len(fixed)
    return total_dist


def perform_icp(path):
    models = []
    scores = []

    with open(path + "/objs/animation.json") as json_file:
        animation = json.load(json_file)
        obj = animation["frames"][0]
        dirs = []

        for idx, vertex in enumerate(obj["vertices"]):

            for v in animation["frames"]:
                x = v["vertices"][idx]["x"]
                y = v["vertices"][idx]["y"]
                z = v["vertices"][idx]["z"]
                dirs.append((x, y, z))

            models.append(np.array(dirs))

    dist = 10

    for midx in range(dist, len(models), 1):
        # Order seems to matter. No idea why?
        pcfix, pcmov, _ = icp(models[midx], models[midx - dist])

        # Convert the clouds to something more handy
        c0 = []
        for i in range(len(pcfix.x)):
            c0.append((pcfix.x[i], pcfix.y[i], pcfix.z[i]))

        c1 = []
        for i in range(len(pcmov.x)):
            c1.append((pcmov.x[i], pcmov.y[i], pcmov.z[i]))

        score = rmsd_score(c0, c1)
        del c0
        del c1
        del pcfix
        del pcmov
        scores.append((midx, score))
        print(midx, score)

    return scores


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch ICP test")

    parser.add_argument(
        "--savedir", default="./save", help="The name for checkpoint save directory."
    )

    # Initial setup of PyTorch
    args = parser.parse_args()
    scores = perform_icp(args.savedir)
    p = list(zip(*scores))
    plt.plot(p[0], p[1])
    plt.xlabel('Save Interval through training.')
    plt.ylabel('Mean Distance (L1)')
    #plt.show()
    plt.savefig("icp_test.png")
