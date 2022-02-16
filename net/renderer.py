""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/          # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/          # noqa
Author : Benjamin Blundell - benjamin.blundell@kcl.ac.uk

renderer.py - Perform splatting of gaussians with torch
functions. Based on the DirectX graphics pipeline.
"""

import torch
import math
from util.math import (
    gen_mat_from_rod,
    gen_trans_xy,
    gen_identity,
    gen_ndc,
    gen_scale,
    VecRotTen,
    TransTen,
    PointsTen,
)


class Splat(object):
    """Our splatter class that generates matrices, loads 3D
    points and spits them out to a 2D image with gaussian
    blobs. The gaussians are computed in screen/pixel space
    with everything else following the DirectX style
    pipeline."""

    # TODO - we should really check where requires grad is actually needed.

    def __init__(
        self, size=(128, 128), device=torch.device("cpu")
    ):
        """
        Initialise the renderer.

        Parameters
        ----------

        size : tuple
            The size of the rendered image, in pixels (default: (128, 128))

        Returns
        -------
        Splat
            The renderer itself

        """

        self.size = size
        # self.near = near
        # self.far = far
        self.device = device
        # self.perspective = gen_perspective(fov, aspect, near, far)
        self.modelview = gen_identity(device=self.device)
        self.trans_mat = gen_identity(device=self.device)
        self.rot_mat = gen_identity(device=self.device)
        self.scale_mat = gen_scale(
            torch.tensor([0.5], device=self.device),
            torch.tensor([0.5], device=self.device),
            torch.tensor([0.5], device=self.device),
        )

        self.ndc = gen_ndc(self.size, device=self.device)
        self.xs = torch.tensor([0], dtype=torch.float32)
        self.ys = torch.tensor([0], dtype=torch.float32)
        # self.w_mask = torch.tensor([0])

        mask = []
        for _ in range(0, 200):
            mask.append(1.0)
        self.mask = torch.tensor(mask, device=self.device)

    def _gen_mats(self, points: PointsTen):
        """
        Internal function.
        Generate the matrices we need to do the rendering.
        These are support matrices needed by pytorch in order
        to convert out points to 2D ones all in the same
        final tensor."""

        # X indices
        numbers = list(range(0, self.size[1]))
        rectangle = [numbers for x in range(0, self.size[0])]
        cuboid = []

        for i in range(0, points.data.shape[0]):
            cuboid.append(rectangle)

        self.xs = points.data.new_tensor(cuboid)

        # Y indices
        rectangle = []
        cuboid = []

        for i in range(0, self.size[0]):
            numbers = [i for x in range(self.size[1])]
            rectangle.append(numbers)

        for i in range(0, points.data.shape[0]):
            cuboid.append(rectangle)

        self.ys = torch.tensor(cuboid, device=self.device)

    def transform_points(
        self, points: torch.Tensor, a: VecRotTen, t: TransTen
    ) -> torch.Tensor:
        """
        Transform points with translation and rotation. A utility
        function used in eval.py to produce a list of transformed points
        we can save to a file.

        Parameters
        ----------
        points : torch.Tensor
            The points all converted to a tensor.

        a : VecRotTen
            The rotation to apply to these points.

        t : TransTen
            The translation to apply to these points.

        Returns
        -------
        torch.Tensor
            The converted points as a tensor.

        """
        self.rot_mat = gen_mat_from_rod(a)
        self.trans_mat = gen_trans_xy(t.x, t.y)
        self.modelview = torch.matmul(self.rot_mat, self.trans_mat)
        o = torch.matmul(self.modelview, points.data)
        return o

    def to(self, device):
        """
        Move this class and all it's associated data from
        one device to another.

        Parameters
        ----------
        device : str
            The device we are moving the renderer to - CUDA or cpu.

        Returns
        -------
        Splat
            The renderer itself.

        """
        self.device = torch.device(device)
        # self.perspective = self.perspective.to(device)
        self.modelview = self.modelview.to(device)
        self.trans_mat = self.trans_mat.to(device)
        self.rot_mat = self.rot_mat.to(device)
        self.scale_mat = self.scale_mat.to(device)
        self.ndc = self.ndc.to(device)
        self.xs = self.xs.to(device)
        self.ys = self.ys.to(device)
        # self.w_mask = self.w_mask.to(device)
        return self

    def render(
        self,
        points: PointsTen,
        rot: VecRotTen,
        trans: TransTen,
        mask: torch.Tensor,
        sigma=1.25,
    ):
        """
        Generate an image. We take the points, a mask, an output filename
        and 2 classed that represent the rodrigues vector and the translation.
        Sigma refers to the spread of the gaussian. The mask is used to ignore
        some of the points if necessary.

        Parameters
        ----------
        points : PointsTen
            The points we are predicting.
        rot : VecRotTen
            The rotation as a vector
        trans : TransTen
            The translation of the points.
        mask : torch.Tensor
            A series of 1.0s or 0.0s to mask out certain points.
        sigma : float
            The sigma value to render our image with.

        Returns
        -------
        None

        """

        assert mask is not None

        if self.xs.shape[0] != points.data.shape[0]:
            self._gen_mats(points)

        # This section causes upto a 20% hit on the GPU perf
        self.rot_mat = gen_mat_from_rod(rot)
        self.trans_mat = gen_trans_xy(trans.x, trans.y)
        self.modelview = torch.matmul(
            torch.matmul(self.scale_mat, self.trans_mat), self.rot_mat
        )
        p0 = torch.matmul(self.modelview, points.data)
        p1 = torch.matmul(self.ndc, p0)
        px = p1.narrow(1, 0, 1)
        py = p1.narrow(1, 1, 1)
        ex = px.expand(points.data.shape[0], self.size[0], self.size[1])
        ey = py.expand(points.data.shape[0], self.size[0], self.size[1])

        # Expand the mask out so we can cancel out the contribution
        # of some of the points
        mask = mask.reshape(mask.shape[0], 1, 1)
        mask = mask.expand(mask.shape[0], ey.shape[1], ey.shape[2])

        model = (
            1.0
            / (2.0 * math.pi * sigma ** 2)
            * torch.sum(
                mask
                * torch.exp(
                    -((ex - self.xs) ** 2 + (ey - self.ys) ** 2) / (2 * sigma ** 2)
                ),
                dim=0,
            )
        )

        return model
