"""   # noqa 
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/           # noqa 
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/           # noqa 
Author : Benjamin Blundell - benjamin.blundell@kcl.ac.uk

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
        img.close()
    else:
        # Assume a numpy array
        mm = np.max(img_tensor)
        img_tensor = img_tensor / mm
        img = Image.fromarray(np.uint8(img_tensor * 255))
        img.save(name, "JPEG")
        img.close()


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


def load_fits(path, flip=False):
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
    t = torch.tensor(data, dtype=torch.float32)
    if flip:
        t = torch.flipud(t)

    # print("data",data)
    return t


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

class NormaliseBasic(object):
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
        NormaliseBasic
        """
        self.factor = 100.0

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
