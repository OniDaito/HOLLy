"""  # noqa 
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/       # noqa 
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/       # noqa 
Author : Benjamin Blundell - benjamin.blundell@kcl.ac.uk

util_plyobj.py - utils for loading and saving models

TODO - merge with util_points as there is a bit of 
overlap here and there.

"""
from util.math import Points, Point


def save_ply(path, vertices):
    """
    Save a basic ascii ply file that just has vertices.

    Parameters
    ----------
    path : str
        A path and filename for the save file
    vertices : list
        A list of items, each of which has at least 3 float elements.

    Returns
    -------
    None

    """

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment VCGLIB generated\n")
        f.write("element vertex " + str(len(vertices)) + "\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("element face 0\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v in vertices:
            f.write(str(round(v[0], 4)) + " ")
            f.write(str(round(v[1], 4)) + " ")
            f.write(str(round(v[2], 4)) + "\n")


def load_ply(path) -> Points:
    """
    Load a ply file, returning a Points instance.

    Parameters
    ----------
    path : str
        A path and filename for the ply file

    Returns
    -------
    Points
        Our Points instance

    """
    points = Points()
    verts_now = False

    with open(path, "r") as f:
        for line in f.readlines():
            if verts_now:
                tokens = line.replace("\n", "").split(" ")
                if not (len(tokens) >= 4 and tokens[0] == "3"):
                    x = float(tokens[0])
                    y = float(tokens[1])
                    z = float(tokens[2])
                    points.append(Point(x, y, z, 1.0))

            if "end_header" in line:
                verts_now = True

    return points


def save_obj(path, vertices: list):
    """
    Save a basic ascii obj file that just has vertices. '''

    Parameters
    ----------
    path : str
        A path and filename for the save file
    vertices : list
        A list of items, each of which has at least 3 float elements.

    Returns
    -------
    None

    """
    with open(path, "w") as f:
        f.write("# shaper output from our neural net\n")
        f.write("o shape\n")
        for point in vertices:
            vertex = "v {:.4f} {:.4f} {:.4f}\n".format(
                float(point[0]), float(point[1]), float(point[2])
            )
            f.write(vertex)
        f.write("\n")


def load_obj(objpath) -> Points:
    """
    Load the points from the OBJ file and generate a Points of
    x,y,z,w vertices.

    Parameters
    ----------
    objpath : str
        A path and filename for the obj file.

    Returns
    -------
    Points
    """
    import pywavefront

    scene = pywavefront.Wavefront(objpath, parse=True)
    points = Points()

    for i in range(0, len(scene.vertices)):
        x = scene.vertices[i][0]
        y = scene.vertices[i][1]
        z = scene.vertices[i][2]
        points.append(Point(x, y, z, 1.0))

    return points
