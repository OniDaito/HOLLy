"""  # noqa 
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/       # noqa 
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/       # noqa 
Author : Benjamin Blundell - k1803390@kcl.ac.uk

util_math.py - Useful math functions mostly found in the splatting
pipeline and a few other places.

"""

import math
import torch
import random
from pyquaternion import Quaternion



class Point:
    """ A point for rendering."""

    def __init__(self, x: float, y: float, z: float, w=1.0):
        """
        Create our point with the homogenous co-ordinate.

        Parameters
        ----------
        x : float
        y : float
        z : float
        w : float
            Default - 1.0.

        Returns
        -------
        self
        """
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def as_list(self):
        """
        Return the point as a list.

        Parameters
        ----------
        None

        Returns
        -------
        list
            x, y, z then w.
        """
        return [self.x, self.y, self.z, self.w]

    def __str__(self):
        return str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ", " + str(self.w)


class Points:
    """A collection of points in a list. Mostly used
    to convert into a particular format of Tensor
    for use in the network proper - PointsTen - and to
    enforce types to allow type checking."""

    def __init__(self, size=0):
        """
        Create our points class.

        Parameters
        ----------
        size : int
            How many points should we we fill initially?
            Default - 0.

        Returns
        -------
        self
        """
        self.data = []
        self.size = size
        for _ in range(size):
            self.data.append(Point(0.0, 0.0, 0.0, 1.0))
        self.counter = 0

    def from_iterable(self, data):
        """
        Create our points but from something in a list of
        lists format.

        Parameters
        ----------
        data : list
            a list of lists of size 4.

        Returns
        -------
        self
        """
        for d in data:
            assert len(d) == 4
            p = Point(d[0], d[1], d[2], d[3])
            self.append(p)
        return self

    def from_chunk(self, data):
        """
        Create our points from a single dimension iterable.

        Parameters
        ----------
        data : list

        Returns
        -------
        self
        """
        for i in range(0, len(data), 4):
            p = Point(data[i], data[i + 1], data[i + 2], data[i + 3])
            self.append(p)
        return self

    def get_iterable(self):
        """
        Get a list of tuples f

        Parameters
        ----------
        none

        Returns
        -------
        List
            A list of size 4 tuples
        """
        vertices = []
        for p in self.data:
            vertices.append((p.x, p.y, p.z, p.w))
        return vertices

    def get_chunk(self):
        """
        Return all the points as a flat list

        Parameters
        ----------
        none

        Returns
        -------
        List
            A 1D list of all the points
        """
        vertices = []
        for p in self.data:
            vertices.append(p.x)
            vertices.append(p.y)
            vertices.append(p.z)
            vertices.append(p.w)
        return vertices

    def append(self, point: Point):
        """
        Add a point to the container.

        Parameters
        ----------
        point : Point

        Returns
        -------
        self
        """
        self.data.append(point)
        self.size = len(self.data)
        return self

    def __next__(self) -> Point:
        if self.counter >= self.size:
            self.counter = 0
            raise StopIteration
        else:
            rval = self.__getitem__(self.counter)
            self.counter += 1
            return rval

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return self

    def __getitem__(self, idx) -> Point:
        return self.data[idx]
    
    def __str__(self):
        s = ""
        for p in self.data:
            s += p.__str__() + "\n"
        return s


class VecRot:
    """
    A Rotation represented as a vector, similar to the Angle Axis
    representation. The length of the vector gives the rotation in
    radians.
    """

    def __init__(self, x: float, y: float, z: float):
        """
        Create our rotation using x, y and z co-ordinates

        Parameters
        ----------
        x : float
        y : float
        z : float

        Returns
        -------
        self

        """
        self.x = x
        self.y = y
        self.z = z

    def get_length(self) -> float:
        """
        Return the length of this vector.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The length of the vector, and therefore, the angle of rotation
            in radians.

        """
        return math.sqrt(
            math.pow(self.x, 2) + math.pow(self.y, 2) + math.pow(self.z, 2)
        )

    def get_angle(self):
        return self.get_length()

    def get_normalised(self) -> list:
        """
        Return a normalised version of the vector.

        Parameters
        ----------
        None

        Returns
        -------
        list
            The normalised x, y and z values in a list.

        """
        length = self.get_length()
        return [self.x / length, self.y / length, self.z / length]

    def as_list(self):
        """
        Return the rotation as an list

        Parameters
        ----------
        None

        Returns
        -------
        list
            x, y and z in a list.
        """
        return [self.x, self.y, self.z]

    def as_nested(self):
        """
        Return a list of lists, each 1 element long in the
        order x, y and z. Useful, when converting to a
        Tensor for use with pytorch.

        Parameters
        ----------
        None

        Returns
        -------
        list
            a list of lists of x, y and z

        """
        return [[self.x], [self.y], [self.z]]

    def get_mat(self):
        """
        Return a 4x4 rotation matrix

        Parameters
        ----------
        None

        Returns
        -------
        list
            a list of lists of float - our matrix

        """
        angle = self.get_angle()
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1 - c
        x, y, z = self.get_normalised()

        m = [[t * x * x + c, t * x * y - z * s, t * x * z + y * s, 0],
             [t * x * y + z * s, t * y * y + c, t * y * z - x * s, 0],
             [t * x * z - y * s, t * y * z + x * s, t * z * z + c, 0],
             [0, 0, 0, 1]]

        return m

    def rotate_points(self, points: Points) -> Points:
        """
        Given a list of points, rotate them

        Parameters
        ----------
        points : Points
            The points we want to rotate.

        Returns
        -------
        Points
            a new list of rotated points

        """
        new_points = Points()
        m = self.get_mat()
        for p in points:
            np = Point(
                m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3] * p.w,
                m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3] * p.w,
                m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3] * p.w,
                p.w
            )
            new_points.append(np)
        return new_points

    def to_ten(self, device="cpu"):
        """
        Convert to a VecRotTen class - a collection of three tensors.
        Useful in our pytorch neural net.

        Parameters
        ----------
        device : str
            What device will the result be bound to? CUDA / cpu?
            Default - cpu.

        Returns
        -------
        VecRotTen

        """
        return VecRotTen(
            torch.tensor([self.x], dtype=torch.float32, device=device),
            torch.tensor([self.y], dtype=torch.float32, device=device),
            torch.tensor([self.z], dtype=torch.float32, device=device),
        )

    def __str__(self):
        return str(self.x) + ", " + str(self.y) + ", " + str(self.z)

    def random(self):
        """
        Generate a random rotation, sampled uniformly from SO(3). This is
        an 'in-place' operation.

        https://demonstrations.wolfram.com/SamplingAUniformlyRandomRotation
        or even easier? http://planning.cs.uiuc.edu/node198.html
        Graphics Gems 3 apparently

        Parameters
        ----------
        None

        Returns
        -------
        self

        """

        u1 = random.random()
        u2 = random.random()
        u3 = random.random()
        rr = Quaternion(
            math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2),
            math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2),
            math.sqrt(u1) * math.sin(2.0 * math.pi * u3),
            math.sqrt(u1) * math.cos(2.0 * math.pi * u3),
        )
        self.x = rr.axis[0] * rr.radians
        self.y = rr.axis[1] * rr.radians
        self.z = rr.axis[2] * rr.radians
        return self


class VecRotTen:
    """ The pytorch Tensor version of the VecRot class."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        """
        Create our class.

        Parameters
        ----------
        x : torch.Tensor
            of shape [1]
        y : torch.Tensor
            of shape [1]
        z : torch.Tensor
            of shape [1]

        Returns
        -------
        self
        """
        self.x = x
        self.y = y
        self.z = z

    def get_length(self) -> torch.Tensor:
        """
        Return the length and therefore the angle.

        Parameters
        ----------
        None

        Returns
        -------
        torch.Tensor
            The length inside a tensor.
        """
        return torch.sqrt(
            torch.pow(self.x, 2) + torch.pow(self.y, 2) + torch.pow(self.z, 2)
        )

    def get_angle(self):
        return self.get_length()

    def get_normalised(self):
        """
        Return the normalised version as a list of tensors

        Parameters
        ----------
        None

        Returns
        -------
        list
            x, y and z tensors, normalised, in a list.
        """
        length = self.get_length()
        return [self.x / length, self.y / length, self.z / length]

    def as_list(self):
        """
        Return the rotation as an list

        Parameters
        ----------
        None

        Returns
        -------
        list
            x, y and z in a list.
        """
        return [self.x, self.y, self.z]

    def random(self):
        """
        Generate a random rotation, sampled uniformly from SO(3). This is
        an 'in-place' operation.

        https://demonstrations.wolfram.com/SamplingAUniformlyRandomRotation
        or even easier? http://planning.cs.uiuc.edu/node198.html
        Graphics Gems 3 apparently

        Parameters
        ----------
        None

        Returns
        -------
        self

        """
        u1 = random.random()
        u2 = random.random()
        u3 = random.random()
        rr = Quaternion(
            math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2),
            math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2),
            math.sqrt(u1) * math.sin(2.0 * math.pi * u3),
            math.sqrt(u1) * math.cos(2.0 * math.pi * u3),
        )
        self.x = torch.Tensor([rr.axis[0] * rr.radians], device=self.x.device)
        self.y = torch.Tensor([rr.axis[1] * rr.radians], device=self.x.device)
        self.z = torch.Tensor([rr.axis[2] * rr.radians], device=self.x.device)
        return self


class Trans:
    """ Translation as two floats."""

    def __init__(self, x: float, y: float):
        """
        Create our translation in the x, y plane.

        Parameters
        ----------
        x : float
        y : float

        Returns
        -------
        self
        """
        self.x = x
        self.y = y

    def to_ten(self, device="cpu"):
        """
        Convert to a TransTen class

        Parameters
        ----------
        device : str
            The device to bind the TransTen to. CUDA / cpu.
            Default - cpu.

        Returns
        -------
        TransTen
        """
        return TransTen(
            torch.tensor([self.x], dtype=torch.float32, device=device),
            torch.tensor([self.y], dtype=torch.float32, device=device),
        )


class TransTen:
    """ Translation as tensors."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        """
        Create our translation using Tensors

        Parameters
        ----------
        x : torch.Tensor
        y : torch.Tensor

        Returns
        -------
        self
        """
        self.x = x
        self.y = y


class Mask:
    """ Our mask for points."""

    def __init__(self, m: list):
        """
        Create our mask - a list of 1.0s and 0.0s

        Parameters
        ----------
        m : list
            A list of 1.0s and 0.0s.

        Returns
        -------
        self
        """
        self.mask = m
        for i in self.mask:
            assert i == 0.0 or i == 1.0

    def __len__(self) -> int:
        return len(self.mask)

    def to_ten(self, device="cpu"):
        """
        Return this mask as a tensor.
        Not an 'in-place' operation.

        Parameters
        ----------
        m : list
            A list of 1.0s and 0.0s.

        Returns
        -------
        torch.Tensor
            The mask as a tensor.
        """
        tm = []
        for m in self.mask:
            tm.append([m])
        return torch.tensor(tm, dtype=torch.float32, device=device)


class PointsTen:
    """ Points, but in their tensor form for pytorch."""

    # TODO - maybe just extend torch.Tensor?

    def __init__(self, device="cpu"):
        """
        Create our Points on a particular device.

        Parameters
        ----------
        x : float
        y : float
        z : float
        w : float
            Default - 1.0.

        Returns
        -------
        self
        """
        self.device = device

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def from_points(self, points):
        """
        Create our PointsTen from a Points instance

        Parameters
        ----------
        points : Points

        Returns
        -------
        self

        """
        tp = []
        for p in points:
            ttp = []
            ttp.append([p.x])
            ttp.append([p.y])
            ttp.append([p.z])
            ttp.append([p.w])
            tp.append(ttp)
        self.data = torch.tensor(tp, dtype=torch.float32, device=self.device)
        return self

    def from_tensor(self, t: torch.Tensor):
        """
        Create our PointsTen from a tensor.

        This take a tensor and assumes it's in the correct
        shape (N, 4, 1). Not ideal and will need to be
        improved.

        Parameters
        ----------
        t : torch.Tensor

        Returns
        -------
        self

        """
        self.data = t
        return self

    def get_points(self) -> Points:
        """
        Return a Points from this tensor

        Parameters
        ----------

        Returns
        -------
        Points
            A points class

        """
        vertices = []
        for i in range(self.data.shape[0]):
            vertices.append((float(self.data[i][0][0]),
                             float(self.data[i][1][0]),
                             float(self.data[i][2][0]), 1.0))

        points = Points().from_iterable(vertices)
        return points


def mat_to_rod(mat: torch.Tensor) -> tuple:
    """
    Given a matrix, return the angle axis tuple.

    Parameters
    ----------
    mat : torch.Tensor
        A 3x3 or 4x4 rotation matrix.

    Returns
    -------
    tuple
        first part is the axis as x, y and z. Second part
        is the angle in radians.

    """
    u = mat.new_tensor(
        [mat[2][1] - mat[1][2], mat[0][2] - mat[2][0], mat[1][0] - mat[0][1]]
    )
    t = mat[0][0] + mat[1][1] + mat[2][2]
    a = math.acos((t - 1.0) / 2.0)
    m = math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    u = u / m * a
    return (u, a)


def gen_perspective(
    fov: float, aspect: float, near: float, far: float, device="cpu"
) -> torch.Tensor:
    """
    Generate a perspective matrix. It's symetric and uses
    near, far, aspect ratio and field-of-view.
    We don't use the OpenGL type here, we go with DirectX.

    Parameters
    ----------
    fov : float
        The field of view in radians.
    aspect : float
        The aspect ratio
    near : float
        The near plane
    far : float
        The far plane
    device : str
        The device to hold this matrix - cuda / cpu.
        Default - cpu.

    Returns
    -------
    torch.Tensor
       A 4x4 perspective matrix.
    """

    D = 1.0 / math.tan(fov / 2.0)
    A = aspect
    NF = near * far
    # B = 2.0 * NF / (near - far)

    pm = torch.tensor(
        [
            [D / A, 0, 0, 0],
            [0, D, 0, 0],
            [0, 0, far / (far - near), -NF / (far - near)],
            [0, 0, 1, 0],
        ],
        dtype=torch.float32,
        requires_grad=False,
        device=device,
    )
    return pm


def gen_identity(device="cpu") -> torch.Tensor:
    """
    Create an identity matrix

    Parameters
    ----------
    device : str
        The device to hold this matrix - cuda / cpu.
        Default - cpu.

    Returns
    -------
    torch.Tensor
       A 4x4 identity matrix.
    """
    return torch.tensor(
        [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]],
        dtype=torch.float32,
        requires_grad=False,
        device=device,
    )


def gen_scale(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    """
    Given three numbers, produce a scale matrix. This *should
    be differentiable.

    Parameters
    ----------
    x : float
    y : float
    z : float

    Returns
    -------
    torch.Tensor
       A 4x4 scale matrix.
    """
    x_mask = x.new_tensor(
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    y_mask = y.new_tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    z_mask = z.new_tensor(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])

    base = x.new_tensor([[0, 0, 0, 0], [0, 0, 0, 0],
                         [0, 0, 0, 0], [0, 0, 0, 1]])

    t_x = x.expand_as(x_mask) * x_mask
    t_y = y.expand_as(y_mask) * y_mask
    t_z = z.expand_as(z_mask) * z_mask

    tm = t_x + t_y + t_z + base

    return tm


def gen_ndc(size, device="cpu"):
    """
    Generate a normalised-device-coordinates to screen matrix.

    Parameters
    ----------
    size : tuple
        A tuple of float or int, width x height in pixels for the
        final image size.
    device : str
        The device to hold this matrix - cuda / cpu.
        Default - cpu.

    Returns
    -------
    torch.Tensor
       A 4x4 ndc-to-screen matrix.
    """
    ds = 1.0
    sx = 0
    sy = 0
    ms = torch.tensor(
        [
            [size[0] / 2.0, 0, 0, size[0] / 2.0 + sx],
            [0, -size[1] / 2.0, 0, size[1] / 2.0 + sy],
            [0, 0, ds / 2.0, ds / 2.0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )

    return ms


def gen_trans(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Generate a translation matrix in x,y,z. It's
    convoluted, just as gen_rot is as we want to keep
    the ability to use backward() and autograd.

    All three tensors should have matching devices.

    Parameters
    ----------
    x : torch.Tensor
        Translation in x, as a shape (1) tensor.
    y : torch.Tensor
        Translation in y, as a shape (1) tensor.
    z : torch.Tensor
        Translation in z, as a shape (1) tensor.

    Returns
    -------
    torch.Tensor
       A 4x4 ndc-to-screen matrix.
    """
    assert x.device == y.device == z.device

    x_mask = x.new_tensor(
        [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    y_mask = y.new_tensor(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])

    z_mask = z.new_tensor(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])

    base = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0],
                         [0, 0, 1, 0], [0, 0, 0, 1]])

    t_x = x.expand_as(x_mask) * x_mask
    t_y = y.expand_as(y_mask) * y_mask
    t_z = z.expand_as(z_mask) * z_mask

    tm = t_x + t_y + t_z + base

    return tm


def gen_trans_xy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Generate a translation matrix in x and y. It's
    convoluted, just as gen_rot is as we want to keep
    the ability to use backward() and autograd.

    Parameters
    ----------
    x : torch.Tensor
        Translation in x, as a shape (1) tensor.
    y : torch.Tensor
        Translation in y, as a shape (1) tensor.

    Returns
    -------
    torch.Tensor
       A 4x4 ndc-to-screen matrix.
    """
    assert x.device == y.device

    x_mask = x.new_tensor(
        [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    y_mask = y.new_tensor(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])

    base = x.new_tensor([[1, 0, 0, 0], [0, 1, 0, 0],
                         [0, 0, 1, 0], [0, 0, 0, 1]])

    t_x = x.expand_as(x_mask) * x_mask
    t_y = y.expand_as(y_mask) * y_mask
    tm = t_x + t_y + base

    return tm


def gen_mat_from_rod(a: VecRotTen) -> torch.Tensor:
    """
    Generate a rotation matrix from our VecRotTen class.
    It's a little better than 3 rotations as
    there are no singularities at the poles. 0,0,0 results in
    badness so we add a small epsilon. xr and yr and zr
    are all tensors.

    Parameters
    ----------
    a : VecRotTen
        A VecRotTen instance of a rotation.

    Returns
    -------
    torch.Tensor
       A 4x4 rotation matrix.
    """
    assert a.x.device == a.y.device == a.z.device
    [xr, yr, zr] = a.as_list()

    if xr == 0 and yr == 0 and zr == 0:
        xr = xr + 1e-3
        yr = yr + 1e-3
        zr = zr + 1e-3

    theta = torch.sqrt(torch.pow(xr, 2) + torch.pow(yr, 2) + torch.pow(zr, 2))

    x = torch.div(xr, theta)
    y = torch.div(yr, theta)
    z = torch.div(zr, theta)

    t_cos = torch.cos(theta)
    t_sin = torch.sin(theta)
    m_cos = 1.0 - t_cos

    x0 = torch.add(torch.mul(torch.pow(x, 2), m_cos), t_cos)
    x1 = torch.sub(torch.mul(torch.mul(x, y), m_cos), torch.mul(z, t_sin))
    x2 = torch.add(torch.mul(torch.mul(x, z), m_cos), torch.mul(y, t_sin))

    y0 = torch.add(torch.mul(torch.mul(y, x), m_cos), torch.mul(z, t_sin))
    y1 = torch.add(torch.mul(torch.pow(y, 2), m_cos), t_cos)
    y2 = torch.sub(torch.mul(torch.mul(y, z), m_cos), torch.mul(x, t_sin))

    z0 = torch.sub(torch.mul(torch.mul(z, x), m_cos), torch.mul(y, t_sin))
    z1 = torch.add(torch.mul(torch.mul(z, y), m_cos), torch.mul(x, t_sin))
    z2 = torch.add(torch.mul(torch.pow(z, 2), m_cos), t_cos)

    x0_mask = xr.new_tensor(
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    x1_mask = xr.new_tensor(
        [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    x2_mask = xr.new_tensor(
        [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    y0_mask = yr.new_tensor(
        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    y1_mask = yr.new_tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    y2_mask = yr.new_tensor(
        [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    z0_mask = zr.new_tensor(
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])

    z1_mask = zr.new_tensor(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])

    z2_mask = zr.new_tensor(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])

    base = zr.new_tensor(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])

    rot_x = (
        x0.expand_as(x0_mask) * x0_mask
        + x1.expand_as(x1_mask) * x1_mask
        + x2.expand_as(x2_mask) * x2_mask
    )

    rot_y = (
        y0.expand_as(y0_mask) * y0_mask
        + y1.expand_as(y1_mask) * y1_mask
        + y2.expand_as(y2_mask) * y2_mask
    )

    rot_z = (
        z0.expand_as(z0_mask) * z0_mask
        + z1.expand_as(z1_mask) * z1_mask
        + z2.expand_as(z2_mask) * z2_mask
    )

    tmat = torch.add(rot_x, rot_y)
    tmat2 = torch.add(tmat, rot_z)
    rot_mat = torch.add(tmat2, base)
    return rot_mat


def gen_rot_rod_single(sx: torch.Tensor) -> torch.Tensor:
    """
    Generate a rotation matrix from a (4,1) shape tensor.

    Parameters
    ----------
    sx : torch.Tensor
        A (4, 1) shape tensor with the x, y, z and 0.

    Returns
    -------
    torch.Tensor
       A 4x4 rotation matrix.
    """
    # TODO - this seems to have a perf hit on the GPU
    x = torch.mul(sx, sx.new_tensor([[1.0], [0.0], [0.0], [0.0]]))
    x = torch.sum(x)

    y = torch.mul(sx, sx.new_tensor([[0.0], [1.0], [0.0], [0.0]]))
    y = torch.sum(y)

    z = torch.mul(sx, sx.new_tensor([[0.0], [0.0], [1.0], [0.0]]))
    z = torch.sum(z)

    a = VecRotTen(x, y, z)

    return gen_mat_from_rod(a)


def normalize(tvec: torch.Tensor) -> torch.Tensor:
    """
    Normalise a vector that is in tensor format.

    Parameters
    ----------
    tvec : torch.Tensor
        A tensor being used as a vector with a (4, 1) shape.

    Returns
    -------
    torch.Tensor
        Normalised version of the input tensor.
    """
    d = torch.zeros((1))
    for i in range(tvec.shape[0]):
        d += tvec[i] * tvec[i]

    d = torch.sqrt(d)
    ll = []
    for i in range(tvec.shape[0]):
        ll.append(float(tvec[i]) / d)
    return torch.tensor(ll, dtype=tvec.dtype)


def angles_to_axis(x_rot: float, y_rot: float, z_rot: float) -> VecRot:
    """
    Convert Euler angles to angle/axis

    Parameters
    ----------
    x_rot : float
        Rotation around X in radians
    y_rot : float
        Rotation around Y in radians
    z_rot : float
        Rotation around Z in radians

    Returns
    -------
    VecRot

    """
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToAngle/

    c1 = math.cos(y_rot/2)
    s1 = math.sin(y_rot/2)
    c2 = math.cos(z_rot/2)
    s2 = math.sin(z_rot/2)
    c3 = math.cos(x_rot/2)
    s3 = math.sin(x_rot/2)
    c1c2 = c1 * c2
    s1s2 = s1 * s2
    w = c1c2 * c3 - s1s2 * s3
    x = c1c2 * s3 + s1s2 * c3
    y = s1 * c2 * c3 + c1 * s2 * s3
    z = c1 * s2 * c3 - s1 * c2 * s3
    angle = 2 * math.acos(w)
    norm = x * x + y * y + z * z

    if (norm < 0.001):
        x = 1
        y = z = 0
    else:
        norm = math.sqrt(norm)
        x /= norm
        y /= norm
        z /= norm

    x *= angle
    y *= angle
    z *= angle
    return VecRot(x, y, z)
