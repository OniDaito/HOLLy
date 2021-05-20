""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

dataload.py -The Dataloader is responsible for generating data
for the DataSet and DataBuffer classes. It either generates on demand
or reads images from the disk. It isn't used directly, rather it works
as follows:

DataBuffer -> DataSet  |
DataBuffer -> DataSet  | -> DataLoader / CepLoader
DataBuffer -> DataSet  |

DataLoader provides all the data for as many DataSets and their associated
buffers as one would want. It performs no further processing such as conversion
to Tensor or normalisation. These take place at the DataSet level.

"""

import os
import random
import pickle
import array
import math
from tqdm import tqdm
from enum import Enum
from util.math import Points, Point, Mask, Trans, VecRot
from pyquaternion import Quaternion

ItemType = Enum("SetType", "SIMULATED FITSIMAGE")


class LoaderItem:
    """The item returned by any of the various Loaders.
    This is the base class, expanded upon below."""

    def __init__(self):
        self.type = ItemType.SIMULATED

    def unpack(self):
        assert False
        return []


class ItemSimulated(LoaderItem):
    """ The Simulated items returned by the basic loader."""

    def __init__(
        self, points: Points, mask: Mask, angle_axis: VecRot, trans: Trans, sigma: float
    ):
        """
        Create our ItemSimulated.

        Parameters
        ----------
        points : Points
           The points that make up this datum.
        mask : Mask
            The mask for the points.
        angle_axis : VecRot
            The rotation of this datum.
        trans : Trans
            The translation of this datum.
        sigma : float
            The sigma this datum should be rendered with.

        Returns
        -------
        self
        """
        self.type = ItemType.SIMULATED
        self.points = points
        self.mask = mask
        self.angle_axis = angle_axis
        self.trans = trans
        self.sigma = sigma

    def unpack(self) -> tuple:
        """
        Unpack the item, return a tuple.

        Parameters
        ----------
        None

        Returns
        -------
        tuple
            The item as a tuple in the following order:
            points, mask, rotation, translation, sigma
        """
        return (self.points, self.mask, self.angle_axis, self.trans, self.sigma)


class Loader(object):
    """Our Loader for simulated data. Given an obj file and some parameters,
    it will generate a set of points and transforms, ready for the Set."""

    def __init__(
        self,
        size=1000,
        objpath="torus.obj",
        dropout=0.0,
        wobble=0.0,
        spawn=1.0,
        max_spawn=1,
        sigma=1.25,
        translate=True,
        max_trans=0.1,
        rotate=True,
        augment=False,
        num_augment=10
    ):
        """
        Create our Loader.

        Parameters
        ----------
        size : int
           The number of simulated items to generate
        dropout : float
            The chance that a point will be masked out, normalised, with
            1.0 being a certainty. Default - 0.
        wobble : float
            How far do we randomly peturb each point? Default - 0.0.
        spawn : float
            The chance that point will be rendered at this base point.
            Normalised, with 1.0 being a certainty. Default - 1.0.
        max_spawn : int
            What is the maximum number of points to be spawned at a
            single ground truth point? Default - 1.
        sigma : float
            The sigma we should set on all the data points. Default - 1.25.
        translate : bool
            Should we translate the data by a random amount?
            Default - True.
        rotate : bool
            Should we rotate the data by a random amount?
            Default - True.
        augment : bool
            Do we want to augment the data by performing rotations in the X,Y plane
            Default - False.
        num_augment : int
            How many augmentations per data-point should we use.
            Default - 10

        Returns
        -------
        self
        """

        # Total size of the data available and what we have allocated
        self.size = size
        self.counter = 0
        self.available = array.array("L")

        # Our ground truth object points
        self.gt_points = array.array("d")

        # How far do we translate?
        self.translate = translate
        self.max_trans = max_trans

        # The rotations and translations we shall use
        self.transform_vars = array.array("d")

        # dropout masks (per sigma)
        self.masks = array.array("d")

        # What sigma of data are we at?
        self.sigma = sigma

        # Actual points we are using (generated from groundtruth)
        self.points = array.array("d")

        # How is the data chunked? (i.e how many points and masks)
        self.points_chunk = 0
        self.masks_chunk = 0

        # Augmentation - essentially a number of 2D affine rotations in XY
        self.augment = augment
        self.num_augment = num_augment

        # Paramaters for generating points
        self.dropout = dropout
        self.wobble = wobble
        self.spawn = spawn
        self.rotate = rotate
        self._max_spawn = max_spawn  # Potentially, how many more flurophores

        from util.plyobj import load_obj, load_ply

        if "obj" in objpath:
            self.gt_points = load_obj(objpath=objpath)
        elif "ply" in objpath:
            self.gt_points = load_ply(objpath)

        self._create_basic()

        # Set here as once we've augmented we need a new size
        if self.augment:
            self.size = size * num_augment

    def reset(self):
        """
        Reset the loader. Delete all the data and rebuild.

        Parameters
        ----------
        None

        Returns
        -------
        self
        """

        self.transform_vars = array.array("d")
        self.masks = array.array("d")
        self.points = array.array("d")
        # TODO - should somehow invalidate the sets above?
        self.available = array.array("L")
        self._create_basic()
        return self

    def remaining(self) -> int:
        """
        Return the number of items remaining that can be claimed by the
        dataset.

        Parameters
        ----------
        None

        Returns
        -------
        int
        """
        return len(self.available)

    def __next__(self) -> LoaderItem:
        if self.counter >= self.size:
            print("Reached the end of the dataloader.")
            self.counter = 0
            raise StopIteration
        else:
            rval = self.__getitem__(self.counter)
            self.counter += 1
            return rval

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        return self

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, idx) -> LoaderItem:
        """
        Return the LoaderItem at the position denoted with idx.

        Parameters
        ----------
        idx : int
           The index of the LoaderItem we want.

        Returns
        -------
        LoaderItem
            The item at idx
        """
        points = Points()
        for i in range(int(self.points_chunk / 4)):
            ts = idx * self.points_chunk + i * 4
            point = Point(
                self.points[ts],
                self.points[ts + 1],
                self.points[ts + 2],
                self.points[ts + 3],
            )
            points.append(point)

        tmask = []
        for i in range(self.masks_chunk):
            ts = idx * self.masks_chunk + i
            tmask.append(self.masks[ts])

        mask = Mask(tmask)

        tv = []
        for i in range(5):
            ts = idx * 5 + i
            tv.append(self.transform_vars[ts])

        item = ItemSimulated(
            points, mask, VecRot(tv[0], tv[1], tv[2]), Trans(
                tv[3], tv[4]), self.sigma
        )
        return item

    # @profile
    def _create_points_mask(self):
        """Given the base points, perform dropout, spawn, noise and all the other
        messy functions, creating a new set of points. Internal function."""
        dropout_mask = array.array("d")
        points = array.array("d")

        for bp in self.gt_points:
            sx = 0.0
            sy = 0.0
            sz = 0.0
            tpoints = array.array("d")

            for i in range(self._max_spawn):

                if self.wobble != 0.0:
                    # sx = sy = sz = 0.0001
                    sx = random.gauss(0, self.wobble)
                    sy = random.gauss(0, self.wobble)
                    sz = random.gauss(0, self.wobble)

                # By organising the points as we do below, we get the correct
                # multiplication by matrices / tensors.
                tpoints.append(bp.x + sx)
                tpoints.append(bp.y + sy)
                tpoints.append(bp.z + sz)
                tpoints.append(1.0)

            if random.uniform(0, 1) >= self.dropout:
                for i in range(0, self._max_spawn):
                    if random.uniform(0, 1) < self.spawn:
                        dropout_mask.append(1.0)
                    else:
                        dropout_mask.append(0.0)
                    points.append(tpoints[i * 4])
                    points.append(tpoints[i * 4 + 1])
                    points.append(tpoints[i * 4 + 2])
                    points.append(tpoints[i * 4 + 3])
            else:
                # All dropped
                for i in range(self._max_spawn):
                    dropout_mask.append(0.0)
                    points.append(tpoints[i * 4])
                    points.append(tpoints[i * 4 + 1])
                    points.append(tpoints[i * 4 + 2])
                    points.append(tpoints[i * 4 + 3])

        return (points, dropout_mask)

    # @profile
    def _create_basic(self):
        """Create a set of rotations and set all the set sizes. Then call
        our threaded render to make the actual images. We render on demand at
        the moment, just creating the basics first. Internal function."""
        tx = 0
        ty = 0

        # Ensure an equal spread of data around all the rotation space so
        # we don't miss any particular areas
        rot = VecRot(0, 0, 0)
        if self.rotate:
            rot.random()

        for idx in tqdm(range(self.size), desc="Generating base data"):
            if self.translate:
                tx = ((random.random() * 2.0) - 1.0) * self.max_trans
                ty = ((random.random() * 2.0) - 1.0) * self.max_trans

            points, dropout_mask = self._create_points_mask()

            if self.augment:
                tp = Points().from_chunk(points)
                new_points = rot.rotate_points(tp).get_chunk()

                for j in range(self.num_augment):
                    rot_a = VecRot(0, 0, math.pi * 2.0 * random.random())

                    # q0 = Quaternion(axis=rot.get_normalised(),
                    #                 radians=rot.get_length())
                    # q1 = Quaternion(axis=rot_a.get_normalised(),
                    #                 radians=rot_a.get_length())
                    # q2 = q0 * q1

                    #rot_f = VecRot(q2.axis[0] * q2.radians,
                    #               q2.axis[1] * q2.radians,
                    #               q2.axis[2] * q2.radians)

                    # What transformation do we really store here, as we have two!
                    # Our whole pipeline relies on there being one complete transform
                    # Composing the 3D base, then the 2D one doesn't work.
                    # To get what we want we modify the points by the initial rotation,
                    # keeping the extra augment till later.

                    self.transform_vars.append(rot_a.x)
                    self.transform_vars.append(rot_a.y)
                    self.transform_vars.append(rot_a.z)
                    self.transform_vars.append(tx)
                    self.transform_vars.append(ty)

                    self.points_chunk = len(new_points)
                    self.masks_chunk = len(dropout_mask)

                    for i in range(self.points_chunk):
                        self.points.append(new_points[i])
                    for i in range(self.masks_chunk):
                        self.masks.append(dropout_mask[i])

                    self.available.append(idx * self.num_augment + j)

            else:
                # Should always be the same
                self.transform_vars.append(rot.x)
                self.transform_vars.append(rot.y)
                self.transform_vars.append(rot.z)
                self.transform_vars.append(tx)
                self.transform_vars.append(ty)

                self.points_chunk = len(points)
                self.masks_chunk = len(dropout_mask)

                for i in range(self.points_chunk):
                    self.points.append(points[i])
                for i in range(self.masks_chunk):
                    self.masks.append(dropout_mask[i])

                del points[:]
                del dropout_mask[:]

                self.available.append(idx)

            if self.rotate:
                rot.random()

    def load(self, filename: str):
        """
        Load the data from a file instead of randomly creating them.

        Parameters
        ----------
        filename : str
           The path to the filename in question.

        Returns
        -------
        self
        """
        # Clear out first
        self.transform_vars = array.array("d")
        self.points = array.array("d")

        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                (
                    self.transform_vars,
                    self.points,
                    self.dropouts,
                    self.dropout,
                    self.wobble,
                ) = pickle.load(f)

        self.size = len(self.transform_vars)
        self.available = [i for i in range(0, self.size)]
        return self

    def save(self, filename):
        """
        Save the current loader to a file on disk. The file
        is saved using Python's pickle format.

        Parameters
        ----------
        filename : str
            The full path and filename to save to.

        Returns
        -------
        self
        """
        with open(filename, "wb") as f:
            pickle.dump(
                (
                    self.transform_vars,
                    self.points,
                    self.dropouts,
                    self.dropout,
                    self.wobble,
                ),
                f,
                pickle.HIGHEST_PROTOCOL,
            )
        return self

    def reserve(self, amount, alloc_csv=None):
        """
        A dataset can reserve an amount of data. This is randomly
        chosen by the dataloader, and returned as a large array or one
        can pass in a path to a file and choose that way, or it is
        passed in order to preserve any dataloader batches.

        Parameters
        ----------
        amount : int
            The amount requested by the dataset
        alloc_csv : str
            The path to a CSV file that determines the allocation.
            This is used when running the net deterministically.
            Default - None.
        Returns
        -------
        list
            The selected indexes of the items for the dataset.
        """

        if amount > self.remaining():
            raise ValueError(
                "Amount requested for reservation exceeds\
                amount of data remaining"
            )
        selected = []
        allocs = []
        removals = []

        if alloc_csv is not None:
            import csv

            with open(alloc_csv) as csvfile:
                csvallocs = csv.reader(csvfile)
                for row in csvallocs:
                    allocs = row

        for i in range(amount):
            idx = 0

            if len(allocs) > 0:
                idx = int(allocs[i])
                removals.append(idx)
                selected.append(self.available[idx])

            else:
                idx = random.randrange(len(self.available))
                selected.append(self.available[idx])
                del self.available[idx]

        if len(allocs) > 0:
            removals.sort(reverse=True)

            for r in removals:
                del self.available[r]

        return selected
