""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

dataset.py - a class that holds all our data for a particular run, like test or
validation data.

Sets are mostly responsible for the ordering of the sets than anything else.

"""
import pickle
import random
import os
from data.loader import Loader, LoaderItem
from enum import Enum

SetType = Enum("SetType", "TRAIN TEST VALID")


class DataSet(object):
    """The dataset Set object. This sits above the loader and contains the items
    for a particular set, such as training, test or validation."""

    def __init__(
        self,
        settype: SetType,
        set_size,
        loader: Loader,
        deterministic=False,
        alloc_csv=None,
    ):
        """
        Create our DataSet

        Parameters
        ----------
        settype : SetType
            What is the purpose of this set? TRAIN, TEST or VALID?
        set_size : int
            How big is the set? Must be <= Loader remaining.
        deterministic : bool
            Do we randomise the set, or stick with random? Default - False.
        alloc_csv : str
            A path to a pre-allocation CSV file.

        Returns
        -------
        DataSet
        """

        # TODO - potentially remove device and take it from the loader
        self.settype = settype
        self.loader = loader
        self.size = set_size
        # allocated contains the ids into the dataloader we have asked for
        self.allocated = self.loader.reserve(self.size, alloc_csv)
        self.counter = 0
        self.deterministic = deterministic

    def shuffle(self):
        """
        Shuffle our DataSet, using the random.shuffle function.

        Parameters
        ----------
        None

        Returns
        -------
        self
        """
        random.shuffle(self.allocated)
        return self

    def shuffle_chunk(self, size=20):
        """
        Shuffle, but in chunks of a particular size. This keeps each
        batch the same, but the order of the batches will be different.

        Parameters
        ----------
        size : int
            How large is each chunk.

        Returns
        -------
        self
        """

        blocks = []
        remainder = range(int(self.size % size))
        tremain = list(reversed(self.allocated))
        remainder_block = [tremain[i] for i in remainder]

        for i in range(0, int(self.size / size) * size, size):
            tb = []
            for j in range(size):
                tb.append(self.allocated[i + j])
            blocks.append(tb)

        random.shuffle(blocks)
        self.allocated = []
        for b in blocks:
            for p in b:
                self.allocated.append(p)

        for p in remainder_block:
            self.allocated.append(p)

        return self

    def remaining(self):
        """
        How far through the dataset are we? How many items remain.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The number of remaining items
        """
        return self.size - self.counter

    def image_size(self):
        """
        What size of images should we be creating? This method reaches
        into the dataloader for the answer.

        Parameters
        ----------
        None

        Returns
        -------
        tuple
            The size of the image x,y in pixels
        """
        return self.loader.image_size

    def set_sigma(self, sigma):
        """
        Set the sigma for this set. This is passed down to the dataloader.
        It's a conviniece funtion but shouldn't be used unless the underlying
        dataloader is reset. This function might be removed.

        Parameters
        ----------
        sigma : float
            The sigma to pass to the loader.

        Returns
        -------
        self
        """
        self.loader.set_sigma(sigma)
        return self

    def reset(self):
        """
        Reset our DataSet, by setting the counter to 0.

        Parameters
        ----------
        None

        Returns
        -------
        self
        """
        self.counter = 0
        return self

    def __len__(self):
        return self.size

    def __iter__(self):
        return self

    def __getitem__(self, idx) -> LoaderItem:
        """Return an item from the loader. Should be points,
        mask, transforms."""
        return self.loader.__getitem__(self.allocated[idx])

    def __next__(self):
        if self.remaining() <= 0:
            self.counter = 0
            raise StopIteration
        else:
            item = self.loader[self.allocated[self.counter]]

            self.counter += 1
            return item

    def load(self, filename: str):
        """
        Load the dataset from the pickle file.

        Parameters
        ----------
        filename : str
            The path to the pickle file.

        Returns
        -------
        self
        """
        # Clear out first
        self.allocated = []

        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                self.allocated = pickle.load(f)

        self.size = len(self.allocated)
        return self

    def save(self, filename):
        """
        Save the dataset as a Python pickle

        Parameters
        ----------
        filename : str
            The path to the CSV file.

        Returns
        -------
        self
        """
        with open(filename, "wb") as f:
            pickle.dump(self.allocated, f, pickle.HIGHEST_PROTOCOL)
