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


def icp(
    X_fix,
    X_mov,
    correspondences=1000,
    neighbors=10,
    min_planarity=0.3,
    min_change=1,
    max_iterations=100,
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


def dist(v, w):
    return (v[0] - w[0])**2 + (v[1] - w[1])**2 + (v[2] - w[2])**2


def rmsd_score(fixed: PointCloud, moved: PointCloud):
    # Convert the clouds to something more handy
    c0 = []
    for i in range(len(fixed.x)):
        c0.append((fixed.x[i], fixed.y[i], fixed.z[i]))

    c1 = []
    for i in range(len(moved.x)):
        c1.append((moved.x[i], moved.y[i], moved.z[i]))

    total_dist = 0

    for v in c0:
        min_d = 1000000
        min_i = 0

        for widx in range(len(c1)):
            w = c1[widx]
            dd = dist(v, w)
            if dd < min_d:
                min_d = dd
                min_i = widx

        if len(c1) == 0:
            break

        del c1[min_i]
        total_dist += math.sqrt(min_d)**2

    total_dist /= len(c0)
    return total_dist


def perform_icp(path):
    models = []

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

    dist = 20
    for i in range(dist, len(models), 10):
        # Order seems to matter. No idea why?
        pcfix, pcmov, _ = icp(models[i], models[i - dist])
        score = rmsd_score(pcfix, pcmov)
        print(i, score)


if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch ICP test")

    parser.add_argument(
        "--savedir", default="./save", help="The name for checkpoint save directory."
    )

    # Initial setup of PyTorch
    args = parser.parse_args()
    perform_icp(args.savedir)
