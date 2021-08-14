"""   # noqa 
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/           # noqa 
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/           # noqa 
Author : Benjamin Blundell - k1803390@kcl.ac.uk

util_image.py - save out our images, load images and
normalise fits images.
"""

import torch
import numpy as np
from PIL import Image


def save_image(img_tensor, name="ten.jpg"):
    """
    Save a particular tensor to an image. We add a normalisation here
    to make sure it falls in range.

    Parameters
    ----------
    filename : str
        The path to the CSV file.

    Returns
    -------
    None
    """
    if hasattr(img_tensor, "detach"):
        img_tensor = img_tensor.detach().cpu()
        mm = torch.max(img_tensor)
        img_tensor = img_tensor / mm
        img = Image.fromarray(np.uint8(img_tensor.numpy() * 255))
        img.save(name, "JPEG")
    else:
        # Assume a numpy array
        mm = np.max(img_tensor)
        img_tensor = img_tensor / mm
        img = Image.fromarray(np.uint8(img_tensor * 255))
        img.save(name, "JPEG")


def save_fits(img_tensor, name="ten.fits"):
    """
    Save a particular tensor to an image. We add a normalisation here
    to make sure it falls in range. This version saves as a

    Parameters
    ----------
    filename : str
        The path to the CSV file.

    Returns
    -------
    self
    """

    from astropy.io import fits

    if hasattr(img_tensor, "detach"):
        img_tensor = np.flipud(img_tensor.detach().cpu().numpy())
    hdu = fits.PrimaryHDU(img_tensor)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name)


def load_fits(path):
    """
    Load a FITS image file, converting it to a Tensor
    Parameters
    ----------
    path : str
        The path to the FITS file.

    Returns
    -------
    torch.Tensor
    """
    from astropy.io import fits

    hdul = fits.open(path)
    data = np.array(hdul[0].data, dtype=np.float32)
    # print("data",data)
    return torch.tensor(data, dtype=torch.float32)


def load_image(path):
    """
    Load a standard image like a png or jpg using PIL/Pillow,
    assuming the image has values 0-254. Convert to a
    normalised tensor.

    Parameters
    ----------
    path : str
        The path to the image

    Returns
    -------
    torch.Tensor
    """
    im = Image.open(path)
    nm = np.asarray(im, dtype=np.float32)
    nm = nm / 255.0
    return torch.tensor(nm, dtype=torch.float32)


class NormaliseNull(object):
    """
    A null normaliser that just returns the image
    Useful if we want to test renderer with no
    normalisation.
    """

    def normalise(self, img):
        return img


# we use this in the dataloader too so we have it here
# It needs to be a class so it can be pickled, which is
# less elegant than a closure :/


class NormaliseTorch(object):
    """
    Normalise using the total intensity and a scaling factor.

    Our simulator does no balancing so we perform a scaling
    and a sort-of-normalising to get things into the right
    range. This matches the same output as our network.
    """

    def __init__(self):
        """
        Create the normaliser with a scaling factor.
        The factor starts at 100, based on a 60 x 60
        image. We scale this based on the image size
        going up.

        Parameters
        ----------
        None

        Returns
        -------
        NormaliseTorch
        """
        self.factor = 100

    def normalise(self, img_batch: torch.Tensor):
        """
        Create the normaliser with a scaling factor.

        Parameters
        ----------
        img_batch : torch.Tensor
            The batch of images we want to normalise
            We normalise each image in the batch on it's own
            as oppose to across the entire batch.
            The shape must be in the pytorch shape of
            (batch_size, 1, h, w)

        Returns
        -------
        torch.Tensor
            The normalised batch tensor
        """
        # tfactor = self.factor * (img_batch.shape[2]**2) / 3600  # 60^2 - dora image size
        intensity = torch.sum(img_batch, [2, 3])
        intensity = self.factor / intensity
        intensity = intensity.reshape(img_batch.shape[0], 1, 1, 1)
        dimg = img_batch * intensity
        return dimg


class NormaliseMinMax(object):
    """ Normalise using min and max values."""

    def __init__(
        self,
        min_intensity: float,
        max_intensity: float,
        scalar=1.0,
        image_size=(128, 128),
    ):
        """
        Create the normaliser.

        Parameters
        ----------
        min_intensity : float
            The minimum intensity.
        max_intensity : float
            The Maximum intensity.
        scalar : float
            An option to scale the result.
            Default - 1.0.
        image_size : tuple
            The image / tensor size we are expecting. We don't rely on the
            shape() function on tensor as we might be operating on a batch.
            Default - (128, 128)

        Returns
        -------
        self
        """
        self.set(min_intensity, max_intensity, image_size)

    def set(self, min_intensity, max_intensity, scalar=1.0, image_size=(128, 128)):
        """
        Set the normaliser's parameters.

        Parameters
        ----------
        min_intensity : float
            The minimum intensity.
        max_intensity : float
            The Maximum intensity.
        scalar : float
            An option to scale the result.
            Default - 1.0.
        image_size : tuple
            The image / tensor size we are expecting. We don't rely on the
            shape() function on tensor as we might be operating on a batch.
            Default - (128, 128)

        Returns
        -------
        torch.Tensor
            The normalised tensor
        """
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.image_size = image_size
        self.scalar = 1.0

    def normalise(self, img: torch.Tensor):
        """
        Given a min max, scale each pixel so the entire count sums to 1. We can
        then scale it up with the scalar parameter.

        Parameters
        ----------
        img : torch.Tensor
            The tensor to normalise

        Returns
        -------
        torch.Tensor
            The normalised tensor
        """
        di = (torch.sum(img) - self.min_intensity) / (
            self.max_intensity - self.min_intensity
        )
        dimg = (
            img * (1.0 / (self.image_size[0] * self.image_size[1]) * di) * self.scalar
        )
        return dimg


class NormaliseMaxIntense(object):
    """Normalise using the maximum summed intensity from a number of
    images rather than the single, highest value."""

    def __init__(self, max_intensity: float, scalar=1.0):
        """
        Create the normaliser.

        Parameters
        ----------
        max_intensity : float
            The maximum intensity to use.
        scalar : float
            An optional scalar (default - 1.0).

        Returns
        -------
        self
        """
        self.set(max_intensity, scalar)

    def set(self, max_intensity: float, scalar=1.0):
        """
        Set the normaliser params

        Parameters
        ----------
        max_intensity : float
            The maximum intensity to use.
        scalar : float
            An optional scalar (default - 1.0).
        image_size : tuple
            The size of the images we are expecting in x,y pixels.
            Default - (128, 128).

        Returns
        -------
        self
        """
        self.max_intensity = max_intensity
        self.scalar = scalar

    def normalise(self, img):
        """
        Perform the normalisation.

        Parameters
        ----------
        img : torch.Tensor
            The image / tensor to normalisation.

        Returns
        -------
        torch.Tensor
            The normalised tensor.
        """
        return img / self.max_intensity * self.scalar
