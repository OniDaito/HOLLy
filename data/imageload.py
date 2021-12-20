""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

imageload.py - the image loader for our network

"""

import os
from astropy.io import fits
import array
import torch
import pickle
from tqdm import tqdm
from data.loader import Loader, LoaderItem, ItemType


class ItemImage(LoaderItem):
    def __init__(self, path):
        self.type = ItemType.FITSIMAGE
        self.path = path

    def unpack(self):
        return self.path


class ImageLoader(Loader):
    """A class that looks for images, saving the filepaths ready for
    use with the dataset class."""

    def __init__(self, size=1000, image_path=".", sigma=None):
        """
        Create our ImageLoader.

        The image loader expects there to be a directory matching the
        sigma passed in. An example would be '/tmp/1.25', passing in
        '/tmp' as the image_path and 1.25 as sigma. None means there
        is just one directory with either one or unknown sigmas.

        Parameters
        ----------
        size : int
            How big should this loader be? How many images do we want?
            Default: (1000)
        image_path : str
            The path to search for images.
        sigma : float
            The sigma of the images in question - default None

        Returns
        -------
        self
        """

        # Total size of the data available and what we have allocated
        self.size = size
        self.counter = 0
        self.base_image_path = image_path
        self.available = array.array("L")
        self.filenames = []
        self.deterministic = False
        self.sigma = sigma

        self._create_data()

    def _find_files(self, path, max_num):
        """ Find the files from the path. Internal function."""
        img_files = []
        pbar = tqdm(total=max_num)
        mini = 1e10
        maxi = 0
        idx = 0

        for dirname, dirnames, filenames in os.walk(self.base_image_path):
            for i in range(len(filenames)):
                filename = filenames[i]
                img_extentions = ["fits", "FITS"]

                if any(x in filename for x in img_extentions):
                    # We need to check there are no duffers in this list
                    fpath = os.path.join(path, filename)
                    with fits.open(fpath) as w:
                        hdul = w[0].data.byteswap().newbyteorder()
                        timg = torch.tensor(hdul, dtype=torch.float32, device="cpu")
                        intensity = torch.sum(timg)

                        if torch.min(timg) < mini:
                            mini = torch.min(timg)
                        if torch.max(timg) > maxi:
                            maxi = torch.max(timg)

                        if intensity > 0.0:
                            pbar.update(1)
                            img_files.append(fpath)
                            self.available.append(idx)
                            idx += 1

                    if len(img_files) >= max_num:
                        pbar.close()
                        return img_files

        pbar.close()
        return img_files

    def set_sigma(self, sigma):
        """
        Set the sigma and immediately create the data, looking for images
        in the directory that matches the sigma.

        Parameters
        ----------
        sigma : float
            The sigma to use.

        Returns
        -------
        self
        """
        self.sigma = sigma
        self._create_data()
        return self

    def _create_data(self):
        """
        We look for directories fitting the path ceppath + "/<sigma>/"
        Couple of choices here
        Internal function.
        """
        path = self.base_image_path

        if self.sigma is not None:

            path = self.base_image_path + "/" + str(int(self.sigma)).zfill(2)
            path1 = self.base_image_path + "/" + str(int(self.sigma))
            path2 = self.base_image_path + "/" + str(self.sigma)

            if os.path.exists(path1):
                path = path1
            if os.path.exists(path2):
                path = path2

        print("Creating data from", path)
        self.filenames = self._find_files(path, self.size)
        assert len(self.filenames) == self.size

    def remaining(self) -> int:
        """
        Return the number of data remaining that haven't
        been claimed by the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The number of remaining items
        """
        return len(self.available)

    def __next__(self):

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

    def __getitem__(self, idx) -> LoaderItem:
        return ItemImage(self.filenames[idx])

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

        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                (
                    self.size,
                    self.base_image_path,
                    self.deterministic,
                    self.sigma,
                ) = pickle.load(f)

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
                    self.size,
                    self.base_image_path,
                    self.deterministic,
                    self.sigma,
                ),
                f,
                pickle.HIGHEST_PROTOCOL,
            )
        return self