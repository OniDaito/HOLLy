""" # noqa 
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa 
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa 
Author : Benjamin Blundell - k1803390@kcl.ac.uk

stats.py - a selection of functions to save files, stats and
similar during training of the network but also during running.

"""

from util.plyobj import save_ply, save_obj
import util.image
import numpy as np
import torch
import json
import redis
import psycopg2
import os
from util.math import PointsTen, VecRot, VecRotTen


class Stats(object):
    """We used a stats object as we have streams and all that sort of
    thing."""

    def __init__(self):
        # create a stream for logging
        self.watching = {}

    def on(self, savedir: str):
        self.savedir = savedir
        path = os.path.normpath(savedir)
        parts = path.split(os.sep)
        self.exp_name = parts[-1]  # WARNING - overwrite potential in the REDIS
        try:
            self.R = redis.Redis(host="localhost", port=6379, db=0)
            conn_string = "host='localhost' dbname='phd' user='postgres' \
                password='postgres'"
            self.pconn = psycopg2.connect(conn_string)
            self.P = self.pconn.cursor()
            # The number of seconds in a month. Used to invalidate Redis Keys
            self._redis_ttl = 2629800
        except Exception:
            print(
                "Cannot connect to PostgreSQL or Redis. Only immediate images \
will be recorded."
            )

        # Create the subdirs we need if they aren't there already
        if not os.path.exists(os.path.join(savedir, "fits")):
            os.mkdir(os.path.join(savedir, "fits"))
        if not os.path.exists(os.path.join(savedir, "jpgs")):
            os.mkdir(os.path.join(savedir, "jpgs"))
        if not os.path.exists(os.path.join(savedir, "objs")):
            os.mkdir(os.path.join(savedir, "objs"))
        if not os.path.exists(os.path.join(savedir, "plys")):
            os.mkdir(os.path.join(savedir, "plys"))

    def watch(self, obj, name: str):
        """Add something to be watched via tensorwatch. This may already be
        watched but we want to update the reference as we may have created a
        new object in the meantime."""
        self.watching[name] = obj

    def close(self):
        """ Make sure we write to the DB. """
        pass
        # TODO - will probably get rid
        # self.db.close()
        # Zip now happens in the generate_stats.sh script
        # with ZipFile(self.savedir + "/stats.zip", 'w',
        #             compression=ZIP_DEFLATED) as myzip:
        #    myzip.write(self.savedir + "/stats.json", arcname="stats.json")

    def tensor_to_list(self, tensr: torch.Tensor) -> list:
        """ Convert our tensor to a list to write out. """
        if tensr.device.type == "cuda":
            tensr = tensr.detach().cpu()
        np_array = np.asarray(tensr.detach().numpy())
        return np_array.tolist()

    def _rconv(self, list_obj: list):
        """ recursive look through the lists to do conversion. """
        new_contain = []
        for item in list_obj:
            if isinstance(item, list):
                new_contain.append(self._rconv(item))
            elif isinstance(item, torch.Tensor):
                new_contain.append(self.tensor_to_list(item))
            elif isinstance(item, VecRotTen):
                new_list = [item.x.tolist(), item.y.tolist(), item.z.tolist()]
                new_contain.append(new_list)
            else:
                new_contain.append(item)
        return new_contain

    def _cxkey(self, key):
        """ Check if this key already exists in the postgres table."""
        self.P.execute("SELECT * from experiments where pname = %s", (key,))
        res = self.P.fetchone()
        return res is not None

    def _padd(self, name: str, epoch: int, step: int, idx: int, key, obj):
        """ Add to postgres."""
        fd = {}
        fd[idx] = {"epoch": epoch, "step": step, "data": obj}

        if not self._cxkey(key):
            self.P.execute(
                "INSERT INTO experiments \
                            VALUES (%s, ARRAY[%s::jsonb])",
                [key, json.dumps(fd)],
            )
        else:
            self.P.execute(
                "UPDATE experiments SET pdata = \
                    array_cat(pdata, ARRAY[%s::jsonb])\
                    WHERE pname = %s;",
                [json.dumps(fd), key],
            )

        self.pconn.commit()

    def _conv(self, obj, name: str, epoch: int, step: int, idx: int):
        # Now check what the object is and write it out properly
        # Using one letter keys to save a little space.
        key = self.exp_name + ":" + name
        if isinstance(obj, str):
            self._padd(name, epoch, step, idx, key, obj)
            self.R.zadd(
                key, {json.dumps({"epoch": epoch, "step": step, "data": obj}): idx}
            )
            self.R.expire(key, self._redis_ttl)

        elif isinstance(obj, torch.Tensor):
            converted = self.tensor_to_list(obj)
            self._padd(name, epoch, step, idx, key, converted)
            self.R.zadd(
                key,
                {json.dumps({"epoch": epoch, "step": step, "data": converted}): idx},
            )
            self.R.expire(key, self._redis_ttl)

        elif isinstance(obj, list):
            new_list = self._rconv(obj)
            self._padd(name, epoch, step, idx, key, new_list)
            self.R.zadd(
                key, {json.dumps({"epoch": epoch, "step": step, "data": new_list}): idx}
            )
            self.R.expire(key, self._redis_ttl)

        elif isinstance(obj, VecRotTen):
            new_list = [obj.x[0], obj.y[0], obj.z[0]]
            self.R.zadd(
                key, {json.dumps({"epoch": epoch, "step": step, "data": new_list}): idx}
            )
            self.R.expire(key, self._redis_ttl)

        elif isinstance(obj, VecRot):
            new_list = [obj.x, obj.y, obj.z]
            self.R.zadd(
                key, {json.dumps({"epoch": epoch, "step": step, "data": new_list}): idx}
            )
            self.R.expire(key, self._redis_ttl)

        else:
            self._padd(name, epoch, step, idx, key, obj)
            self.R.zadd(
                key, {json.dumps({"epoch": epoch, "step": step, "data": obj}): idx}
            )
            self.R.expire(key, self._redis_ttl)

    def update(self, epoch: int, set_size: int, batch_size: int, step: int):
        idx = epoch * set_size + step * batch_size
        """ Update all our streams with the current idx value. """
        for name in self.watching.keys():
            obj = self.watching[name]
            self._conv(obj, name, epoch, step, idx)

    def write_immediate(self, obj, name, epoch, step, idx):
        try:
            self._conv(obj, name, epoch, step, idx)
        except Exception:
            pass
            # This is naughty but until I find a way for it to be less
            # verbose I'm leaving it in for this version.
            #print("No database to store statistic.")

    def save_jpg(
        self,
        data: torch.Tensor,
        savedir: str,
        prefix: str,
        epoch: int,
        step: int,
        idx: int,
    ):
        util.image.save_image(
            data,
            name=savedir
            + "/jpgs/"
            + prefix
            + str(epoch).zfill(3)
            + "_s"
            + str(step).zfill(5)
            + "_i"
            + str(idx).zfill(5)
            + ".jpg",
        )

    def save_fits(
        self,
        data: torch.Tensor,
        savedir: str,
        prefix: str,
        epoch: int,
        step: int,
        idx: int,
    ):
        util.image.save_fits(
            data,
            name=savedir
            + "/fits/"
            + prefix
            + str(epoch).zfill(3)
            + "_s"
            + str(step).zfill(5)
            + "_i"
            + str(idx).zfill(5)
            + ".fits",
        )

    def save_points(
        self, points: PointsTen, savedir: str, epoch: int, step: int, ply=False
    ):
        """Save the points as either an obj or ply file."""
        if not os.path.exists(savedir + "/objs"):
            os.makedirs(savedir + "/objs/")

        path_obj = (
            savedir + "/objs/shape_e" + str(epoch).zfill(3) + "_s" + str(step).zfill(5)
        )
        path_ply = (
            savedir + "/plys/shape_e" + str(epoch).zfill(3) + "_s" + str(step).zfill(5)
        )

        vertices = []
        tv = points.data.clone().cpu().detach().numpy()
        for v in tv:
            vertices.append((v[0][0], v[1][0], v[2][0], 1.0))

        if ply:
            save_ply(path_ply + ".ply", vertices)
        else:
            save_obj(path_obj + ".obj", vertices)


# This is the one and only logging object. It's global and we have helper
# functions to access it.


stat = Stats()


def watch(obj, name: str):
    stat.watch(obj, name)


def write_immediate(obj, name: str, epoch: int, step: int, idx: int):
    stat.write_immediate(obj, name, epoch, step, idx)


def on(savedir: str):
    stat.on(savedir)


def save_jpg(data, savedir: str, prefix: str, epoch: int, step: int, idx: int):
    stat.save_jpg(data, savedir, prefix, epoch, step, idx)


def save_fits(data, savedir: str, prefix: str, epoch: int, step: int, idx: int):
    stat.save_fits(data, savedir, prefix, epoch, step, idx)


def save_points(points: torch.Tensor, savedir: str, epoch: int, step: int):
    stat.save_points(points, savedir, epoch, step)
    stat.save_points(points, savedir, epoch, step, True)


def update(epoch: int, set_size: int, batch_size: int, step: int):
    stat.update(epoch, set_size, batch_size, step)


def close():
    pass
