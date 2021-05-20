"""  # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/       # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/       # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

util_points.py - functions to do with our points tensor
such as loading it from disk or initialising it with 
random data.
"""

import random
from util.math import Points, Point, PointsTen


def load_points(filename) -> Points:
    """
    Load the points from a text file.

    Parameters
    ----------
    filename : str
        A path and filename for the points file

    Returns
    -------
    Points
        Our Points instance

    """
    points = Points(size=0)
    i = 0

    with open(filename, "r") as f:
        for line in f.readlines():
            tokens = line.replace("\n", "").split(",")
            x = float(tokens[0])
            y = float(tokens[1])
            z = float(tokens[2])
            points.append(Point(x, y, z))
            i = i + 1

    return points


def init_points(num_points=500, device="cpu", deterministic=False) -> PointsTen:
    """
    Rather than load a torus or fixed shape, create a
    tensor that contains a random number of points.

    Parameters
    ----------
    num_points : int
        The number of points to make (default 500).
    device : str
        The device that holds the points (cuda / cpu).
        Default - cpu.
    deterministic : bool
        Are we going for a deterministic run?
        Default - False.

    Returns
    -------
    PointsTen
        Our Points in PointsTen form.
    """
    points = Points()
    if deterministic:
        # TODO - can we guarantee this gives the same numbers?
        random.seed(a=9001)

    # Everything is roughly centred in the images so spawn
    # the points close to the centre
    for i in range(0, num_points):
        p = Point(
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
            1.0,
        )
        points.append(p)

    fpoints = PointsTen(device=device)
    fpoints.from_points(points)
    return fpoints


def save_points(filename, points: Points):
    """
    Save the points to a text file.

    Parameters
    ----------
    filename : str
        The file path to save to.
    points : Points
        The points to save.

    Returns
    -------
    None
    """
    tt = points.data.cpu().detach().numpy()
    with open(filename, "w") as f:
        for i in range(0, len(points.data)):
            x = tt[i][0][0]
            y = tt[i][1][0]
            z = tt[i][2][0]
            f.write(str(x) + "," + str(y) + "," + str(z) + "\n")
