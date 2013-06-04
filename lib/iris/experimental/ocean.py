"""
Synthesis of AVD Oceans micro-sprint development on 09-10 May 2013
in collaboration with Tim Graham and Laura Jackson.

Calculating mass transports along lines of constant latitude
on the NEMO tri-polar grid.

"""

from collections import Iterable, namedtuple
import cPickle
from functools import partial
import glob
from itertools import izip
import math
import multiprocessing
import os
import os.path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from shapely.geometry import LineString
from shapely.geometry.polygon import LinearRing

import cartopy.crs as ccrs
import iris


hosts = ['eld288', 'eld116', 'eld244',
         'eld026', 'eld245', 'eld562',
         'eld114', 'eld118', 'eld238']


def _remote_worker(args):
    """
    Remote host server worker that performs the actual processing of
    determining whether one or more lines of constant latitude intersect
    the bounded cell geometry.

    Used to generate a mask, therefore if a line intersects the cell
    geometry, then the result is False, otherwise True.

    Args:

    * args:
        Tuple containing a sequence of one or more constant lines of latitude,
        and the bounded cell :class:`shapely.geometry.polygon.LinearRing`.

    Returns:
        Boolean.

    """
    lines, ring = args
    result = True
    for line in lines:
        if line.intersects(ring):
            result = False
            break
    return result


def _remote_server():
    """
    Multi-processing remote host server, that distributes its processing
    over all available cpu cores.

    Process communication via the stdin/stdout file-handles.

    Returns:
        Pickled processing result via the stdout file-handle.

    """
    args = []
    # Receive input via stdin file-handle.
    for line in sys.stdin:
        args.append(line)
    args = ''.join(args)

    lines, rings = cPickle.loads(args)
    pool = multiprocessing.Pool()
    chunks = rings.size / multiprocessing.cpu_count()
    result = pool.map(_remote_worker,
                      zip([lines] * rings.size, rings), chunks)
    result = np.array(result, dtype=np.bool)
    result = cPickle.dumps(result, cPickle.HIGHEST_PROTOCOL)
    pool.close()
    pool.join()
    # Return output via stdout file-handle.
    sys.stdout.write(result)


def _client(args):
    """
    Local host client that starts a dedicated remote host server
    and delegates work.

    Starts and communicates with the remote host server via a
    secure shell.

    Args:

    * args:
        Tuple containing the remote server hostname and pickled
        data payload to process.

    Returns:
        The processed remote server data payload.

    """
    host, data = args
#    sys.stderr.write('child-{}: starting ...\n'.format(host))
    cmd = 'import iris.experimental.ocean as ocean; ' \
        'ocean._remote_server()'
    p = subprocess.Popen(['ssh',
                          '-o UserKnownHostsFile=/dev/null',
                          '-o StrictHostKeyChecking=no',
                          '{}@{}'.format(os.getlogin(), host),
                          '/usr/local/sci/bin/python -c "{}"'.format(cmd)],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    out, err = p.communicate(data)
    result = cPickle.loads(out)
#    sys.stderr.write('child-{}: Done!\n'.format(host))
    return result


class Grid(namedtuple('Grid', 't u v')):
    """
    Named tuple collection of gridded cubes.

    Args:

    * t:
        The T-cell grid :class:`iris.cube.Cube`.

    * u:
        The U-cell grid :class:`iris.cube.Cube`.

    * v:
        The V-cell grid :class:`iris.cube.Cube`.

    """
    def as_data(self):
        """
        Return the data payload of each cube, apart for the
        T-cell grid :class:`iris.cube.Cube`.

        """
        return Grid(*[getattr(self, attr).data if attr != 't' else None
                      for attr in self._fields])


class Mesh(namedtuple('Mesh', 'dx dy dzu dzv')):
    """
    Named tuple collection of mesh cubes.

    Args:

    * dx:
        The cube containing the delta-x components on the V grid.

    * dy:
        The cube containing the delta-y components on the U grid.

    * dzu:
        The cube containing the delta-z components on the U grid.

    * dzv:
        The cube containing the delta-z components on the V grid.

    """
    def as_data(self):
        """
        Return the data payload of each :class:`iris.cube.Cube`.

        """
        return Mesh(*[getattr(self, attr).data for attr in self._fields])


class PathData(namedtuple('PathData', 'uv dxdy dz')):
    """
    Named tuple collection of combined path data for
    n vertices.

    Args:

    * uv:
        A :class:`numpy.ndarray` of combined U and V data.

    * dxdy:
        A :class:`numpy.ndarray` of combined DX and DY data.

    * dz:
        A :class:`numpy.ndarray` of combined DZU and DZV data.

    """


#
# For want of a better class name ...
#
class Ocean(object):
    def __init__(self, grid_cubes, mesh_cubes,
                 hosts=None, cache_dir=None):
        """
        Container class responsible for calculating paths of
        constant latitude and associated transport.

        Args:

        * grid_cubes
            An ordered sequence of T, U, V grid :class:`iris.cube.Cube` s.
            Also see :class:`Grid` named tuple.

        * mesh_cubes:
            An ordered sequence of DX, DY, DZU, DZV mesh
            :class:`iris.cube.Cube` s. Also see :class:`Mesh` named tuple.

        Kwargs:

        * hosts:
            Sequence of one or more hostnames designated as processing
            resources.

        * cache_dir:
            Directory where intermediate results are cached for performace
            purposes.

        """
        self.grid_cubes = Grid(*grid_cubes)
        self.mesh_cubes = Mesh(*mesh_cubes)
        if hosts is None:
            hosts = []
        if not isinstance(hosts, Iterable):
            hosts = [hosts]
        self.hosts = hosts
        self.cache_dir = cache_dir
        self._transport = None
        self.path_data = None

    def __repr__(self):
        args = [repr(cube) for cube in self.grid_cubes]
        return '<{}({!r}, {!r})>'.format(type(self).__name__,
                                         repr(self.grid_cubes),
                                         repr(self.mesh_cubes))

    @staticmethod
    def named_constraint(name):
        """
        Convenience function that generates a load constraint based
        on the name of the target cube to be loaded.

        Args:

        * name:
            The name of the cube to be loaded.

        Returns:
            A :class:`iris.Constraint`.

        """
        return iris.Constraint(cube_func=lambda cube: cube.name() == name)

    @staticmethod
    def load_grid_cubes(uris, constraints=None, callbacks=None, prefix=None):
        """
        Convenience function that loads the T, U and V cubes from three
        separate datasets.

        Args:

        * uris:
            Sequence of three T, U and V filenames/URIs.

        Kwargs:

        * constraints:
            Sequence of three T, U and V load constraints.

        * callbacks:
            Sequence of three T, U and V load modifier/filter functions.

        * prefix:
            Common prefix to be pre-pended to each URI.

        Returns:
            A :class:`Grid` named tuple of three :class:`iris.cube.Cube`s
            in T, U, V order.

        """
        #
        # TODO: May want to expand the functionality to load the
        #       cubes from a single URI.
        #
        if not isinstance(uris, Iterable) or len(uris) != 3:
            raise ValueError('Require exactly three URIs to load, '
                             'in T, U, V order.')
        if constraints is None:
            constraints = [None] * 3
        elif not isinstance(constraints, Iterable) or len(constraints) != 3:
            raise ValueError('Require exactly three load constraints, '
                             'in T, U, V order.')
        if callbacks is None:
            callbacks = [None] * 3
        elif not isinstance(callbacks, Iterable) or len(callbacks) != 3:
            raise ValueError('Require exactly three load callbacks, '
                             'in T, U, V order.')
        if prefix is not None:
            prefix = os.path.abspath(os.path.expanduser(prefix))
            uris = [os.path.join(prefix, uri) for uri in uris]
        zipper = izip(uris, constraints, callbacks)
        cubes = [iris.load_cube(uri, constraint, callback)
                 for uri, constraint, callback in zipper]
        return Grid(*cubes)

    @staticmethod
    def load_region_mask(uri, constraint=None, callback=None, negate=True):
        """
        Convenience function that loads the land/sea region mask.

        Args:

        * uri:
            The region mask filename/URI.

        Kwargs:

        * constraint:
            A load constraint for the region mask.

        * callback:
            A load modifier/filter function for the region mask.

        * negate:
            Flag whether the region mask is to be logically negated.
            Defaults to True.

        Returns:
            A boolean :class:`numpy.ndarray` of the region mask.

        """
        region_mask = iris.load_cube(uri, constraint, callback).data
        if negate:
            region_mask = ~region_mask.astype(np.bool)
        return region_mask

    @staticmethod
    def load_mesh_cubes(uri, constraints=None, callback=None):
        """
        Convenience function that loads the DX, DY, DZU, and DZV cubes
        from a single dataset.

        Args:

        * uri:
            Filename/URI containing the mesh cubes.

        Kwargs:

        * constraints:
            Sequence of four load constraints, in DX, DY, DZU, DZV order.

        * callback:
            A load modifier/filter function.

        Returns:
            A :class:`Mesh` named tuple of four :class:`iris.cube.Cube`s
            in DX, DY, DZU, DZV order.

        """
        #
        # TODO: May want to expand the functionality to load
        #       the cubes from multiple URI's.
        #
        if constraints is not None:
            if not isinstance(constraints, Iterable):
                constraints = [constraints]
            if len(constraints) != 4:
                raise ValueError('Require exactly four load constraints, '
                                 'in DX, DY, DZU, DZV order.')
        return Mesh(*iris.load_cubes(uri, constraints, callback))

    def purge(self):
        """Purge the cache directory of intermediate cache files."""
        if self.cache_dir is not None:
            TCell.purge(self.cache_dir)

    def path(self, latitudes, path_func=None):
        """
        Calculate the path along one or more constant lines of latitude.

        Args:

        * latitudes:
            One or more target latitudes, in Geodetic decimal degrees.

        Kwargs:

        * path_func:
            The function responsible for generating the path list
            from a :class:`TCell` mask, see :meth:`TCell.latitude`.
            Defaults to :func:`top_edge_path`.

        Returns:
            A tuple pair containing the mask and path.

        """
        tcell = TCell(self.grid_cubes.t, self.hosts, cache_dir=self.cache_dir)
        mask = tcell.latitude(latitudes)
        if path_func is None:
            path_func = top_edge_path
        path = path_func(mask)
        return mask, path

    def transport(self, path, region_mask=None, grid_type='C'):
        """
        Calculate the transport over the vertices traversed by
        the path.

        Args:

        * path:
            A list containing one or more sub-paths. Each sub-path is a
            list of (row, column) tuple pairs.

        Kwargs:

        * region_mask:
            A boolean :class:`numpy.ndarray` land/sea region mask.

        * grid_type:
            The type of Arakawa mesh grid, either 'B' or 'C'.
            Defaults to 'C'.

        Returns:
            The transport :class:`numpy.ndarray`.

        """
        if self._transport is None:
            self._transport = Transport(self.grid_cubes, self.mesh_cubes)
        self.path_data = self._transport.path_data(path,
                                                   region_mask=region_mask,
                                                   grid_type=grid_type)
        return self._transport.path_transport(*self.path_data)

    def _common(self, latitudes,
                path_func=None, region_mask=None, grid_type='C'):
        """Convenience method for common path transport computation."""
        mask, path = self.path(latitudes, path_func=path_func)
        if self._transport is None:
            self._transport = Transport(self.grid_cubes, self.mesh_cubes)
        self.path_data = self._transport.path_data(path,
                                                   region_mask=region_mask,
                                                   grid_type=grid_type)

    def stream_function(self, latitudes,
                        path_func=None, region_mask=None, grid_type='C'):
        """
        Calculate the cumulative sum transport over the one or more lines
        of constant latitude.

        Args:

        * latitudes:
            One or more target latitudes, in Geodetic decimal degrees.

        Kwargs:

        * path_func:
            The function responsible for generating the path list
            from a :class:`TCell` mask, see :meth:`TCell.latitude`.
            Defaults to :func:`top_edge_path`.

        * region_mask:
            A boolean :class:`numpy.ndarray` land/sea region mask.

        * grid_type:
            The type of Arakawa mesh grid, either 'B' or 'C'.
            Defaults to 'C'.

        Returns:
            The cumulative sum transport :class:`numpy.ndarray`.

        """
        self._common(latitudes,
                     path_func=path_func,
                     region_mask=region_mask,
                     grid_type=grid_type)
        return self._transport.stream_function(*self.path_data)

    def net_transport(self, latitudes,
                      path_func=None, region_mask=None, grid_type='C'):
        """
        Calculate the sum transport over the one or more lines of
        constant latitude.

        Args:

        * latitudes:
            One or more target latitudes, in Geodetic decimal degrees.

        Kwargs:

        * path_func:
            The function responsible for generating the path list
            from a :class:`TCell` mask, see :meth:`TCell.latitude`.
            Defaults to :func:`top_edge_path`.

        * region_mask:
            A boolean :class:`numpy.ndarray` land/sea region mask.

        * grid_type:
            The type of Arakawa mesh grid, either 'B' or 'C'.
            Defaults to 'C'.

        Returns:
            The sum transport :class:`numpy.ndarray`.

        """
        self._common(latitudes,
                     path_func=path_func,
                     region_mask=region_mask,
                     grid_type=grid_type)
        return self._transport.net_transport(*self.path_data)


###############################################################################


def top_edge_path(mask):
    """
    Calculates a top-edge path containing one or more sub-paths.

    Args:

    * mask:
        The boolean :class:`numpy.ndarray` of T-cells participating
        in one or more lines of constant latitude, which are set to False.

    Returns:
        A list containing one or more top-edge sub-paths. Each sub-path is a
        list of (row, column) tuple pairs.

    """
    mask = np.logical_not(mask)
    path = []
    sub_path = []
    limit_j = mask.shape[0] - 1
    limit_i = mask.shape[1] - 1
    for i in xrange(mask.shape[1]):
        j_vals = mask[:, i].nonzero()[0]
        if not j_vals.size:
            if sub_path:
                path.append(sub_path)
                sub_path = []
            continue
        # Assume they are in order from nonzero()
        max_j = j_vals[-1]
        min_j = j_vals[0]
        if max_j != min_j + j_vals.size - 1:
            raise ValueError('Cannot handle path bifurcation.')
        max_j += 1
        if not sub_path:
            # Add new sub-path left-hand starting point.
            sub_path.append((max_j, i))
        elif sub_path[-1][0] != max_j:
            # Add left-hand vertical point/s up to new j.
            previous_j = sub_path[-1][0]
            if previous_j < max_j:
                # Step up new j left-hand points.
                previous_j += 1
                for j in range(previous_j, max_j):
                    sub_path.append((j, i))
            else:
                # Step down new j left-hand points.
                previous_j -= 1
                for j in range(previous_j, max_j, -1):
                    sub_path.append((j, i))
            sub_path.append((max_j, i))
        # Append right-hand vertex point.
        sub_path.append((max_j, i + 1))
    # Add last path (if any).
    if sub_path:
        path.append(sub_path)

    return path


def plot_mask_path(mask, path):
    """
    Convenience function to plot the T-cell mask and path.

    Args:

    * mask:
        The boolean :class:`numpy.ndarray` of T-cells participating
        in one or more lines of constant latitude, which are set to False.

    * path:
        A list containing one or more top-edge sub-paths. Each sub-path is a
        list of (row, column) tuple pairs.

    """
    nj, ni = mask.shape
    ax = plt.axes()
    ax.set_xlim(0, ni)
    ax.set_ylim(0, nj)
    x, y = np.meshgrid(np.arange(ni), np.arange(nj))
    ax.pcolormesh(mask)
    ax.grid(color='red')
    ax.set_xticks(np.arange(ni + 1))
    ax.set_yticks(np.arange(nj + 1))
    for sub_path in path:
        j_vals = [pair[0] for pair in sub_path]
        i_vals = [pair[1] for pair in sub_path]
        plt.plot(i_vals, j_vals, lw=3, color='g')


###############################################################################


class Transport(object):
    def __init__(self, grid_cubes, mesh_cubes):
        """
        Container class responsible for collating the path data for paths
        of constant latitude and associated path transport.

        Args:

        * grid_cubes:
            An ordered sequence of T, U, V grid :class:`iris.cube.Cube`s.

        * mesh_cubes:
            An ordered sequence of DX, DY, DZU, DZV mesh
            :class:`iris.cube.Cube`s.

        """
        self.grid_data = Grid(*grid_cubes).as_data()
        self.mesh_data = Mesh(*mesh_cubes).as_data()

    def _up_U(self, start_yx, end_yx, grid_type):
        if grid_type == 'C':
            y = start_yx[0]
            x = start_yx[1] - 1
        elif grid_type == 'B1':
            y = start_yx[0]
            x = start_yx[1]
        elif grid_type == 'B2':
            y = end_yx[0]
            x = start_yx[1]
        return y, x

    def _down_U(self, start_yx, end_yx, grid_type):
        if grid_type == 'C':
            y = end_yx[0]
            x = end_yx[1] - 1
        elif grid_type == 'B1':
            y = start_yx[0]
            x = start_yx[1]
        elif grid_type == 'B2':
            y = end_yx[0]
            x = start_yx[1]
        return y, x

    def _right_V(self, start_yx, end_yx, grid_type):
        if grid_type == 'C':
            y = start_yx[0] - 1
            x = start_yx[1]
        elif grid_type == 'B1':
            y = start_yx[0]
            x = start_yx[1]
        elif grid_type == 'B2':
            y = start_yx[0]
            x = end_yx[1]
        return y, x

    def _left_V(self, start_yx, end_yx, grid_type):
        if grid_type == 'C':
            y = end_yx[0] - 1
            x = end_yx[1]
        elif grid_type == 'B1':
            y = start_yx[0]
            x = start_yx[1]
        elif grid_type == 'B2':
            y = start_yx[0]
            x = end_yx[1]
        return y, x

    def _get_points(self, path, u, v, grid_type):
        """
        Calculate the path data for each sub-path vertex.

        * Args:

        * path:
            A list containing one or more sub-paths. Each sub-path is a
            list of (row, column) tuple pairs.

        * u:
            A :class:`numpy.ndarray` of U (eastward) data.

        * v:
            A :class:`numpy.ndarray` of V (northward) data.

        * grid_type:
            The type of Arakawa mesh grid.

        Returns:
            One or more :class:`PathData` named tuples.

        """
        dx, dy, dzu, dzv = self.mesh_data

        # Determine the total number of edges in the path.
        n = sum(len(sub_path) - 1 for sub_path in path)

        # Prepare empty arrays for our results
        uv = ma.empty((u.shape[:-2] + (n,)))
        dxdy = ma.empty(n)
        dz = ma.empty((u.shape[-3], n))

        ni = 0
        for sub_path in path:
            for start_yx, end_yx in zip(sub_path[:-1], sub_path[1:]):
                if not (0 <= start_yx[0] <= u.shape[-2] and
                        0 <= start_yx[1] <= u.shape[-1]):
                    msg = 'Invalid sub-path point: {}'.format(start_yx)
                    raise ValueError(msg)

                # Up => U
                if start_yx[0] + 1 == end_yx[0] and start_yx[1] == end_yx[1]:
                    y, x = self._up_U(start_yx, end_yx, grid_type)
                    scale = 1
                    uv_src = u
                    dxdy_src = dy
                    dz_src = dzu
                # Down => -U
                elif start_yx[0] - 1 == end_yx[0] and start_yx[1] == end_yx[1]:
                    y, x = self._down_U(start_yx, end_yx, grid_type)
                    scale = -1
                    uv_src = u
                    dxdy_src = dy
                    dz_src = dzu
                # Right => -V
                elif start_yx[1] + 1 == end_yx[1] and start_yx[0] == end_yx[0]:
                    y, x = self._right_V(start_yx, end_yx, grid_type)
                    scale = -1
                    uv_src = v
                    dxdy_src = dx
                    dz_src = dzv
                # Left => V
                elif start_yx[1] - 1 == end_yx[1] and start_yx[0] == end_yx[0]:
                    y, x = self._left_V(start_yx, end_yx, grid_type)
                    scale = 1
                    uv_src = v
                    dxdy_src = dx
                    dz_src = dzv
                else:
                    msg = 'Invalid sub-path segment: ' \
                        '{0} -> {1})'.format(start_yx, end_yx)
                    raise RuntimeError(msg)

                uv[..., ni] = scale * uv_src[..., y, x]
                dxdy[ni] = dxdy_src[y, x]
                dz[:, ni] = dz_src[:, y, x]
                ni += 1

        return PathData(uv, dxdy, dz)

    def path_data(self, path, region_mask=None, grid_type='C'):
        """
        Calculate the path data for each vertex traversed within the
        one or more sub-paths of the given Awakawa grid.

        Args:

        * path:
            A list containing one or more sub-paths. Each sub-path is a
            list of (row, column) tuple pairs.

        Kwargs:

        * region_mask:
            A boolean :class:`numpy.ndarray` land/sea region mask.

        * grid_type:
            The type of Arakawa mesh grid, either 'B' or 'C'.
            Defaults to 'C'.

        Returns:
            A 'C' grid :class:`PathData`, or a list containing two
            :class:`PathData`s of a 'B' grid.

        """
        grid_type = grid_type.upper()
        if grid_type not in ['B', 'C']:
            raise ValueError('Invalid grid type {!r}.'.format(grid_type))

        _, u, v = self.grid_data

        if region_mask is not None:
            u[..., region_mask] = ma.masked
            v[..., region_mask] = ma.masked

        if grid_type == 'C':
            data = self._get_points(path, u, v, 'C')
        else:
            data1 = self._get_points(path, u, v, 'B1')
            data2 = self._get_points(path, u, v, 'B2')
            data = [data1, data2]

        return data

    def path_transport(self, uv, dxdy, dz):
        """
        Calculate the transport over the vertices traversed by
        the path data.

        Args:

        * uv:
            A :class:`numpy.ndarray` of combined U and V data.

        * dxdy:
            A :class:`numpy.ndarray` of combined DX and DY data.

        * dz:
            A :class:`numpy.ndarray` of combined DZ data.

        Returns:
            The transport :class:`numpy.ndarray`.

        """
        if dz.shape[-1] != uv.shape[-1]:
            dz = dz[:, np.newaxis]
        edge_transport = (uv * dz) + dxdy
        return edge_transport.sum(axis=-1)

    def stream_function(self, uv, dxdy, dz):
        """
        Convenience method to calculate the cumulative sum transport
        over the vertices traversed by the path data.

        Args:

        * uv:
            A :class:`numpy.ndarray` of combined U and V data.

        * dxdy:
            A :class:`numpy.ndarray` of combined DX and DY data.

        * dz:
            A :class:`numpy.ndarray` of combined DZ data.

        Returns:
            The cumulative sum transport :class:`numpy.ndarray`.

        """
        path_transport = self.path_transport(uv, dxdy, dz)
        return path_transport.cumsum(axis=-1)

    def net_transport(self, uv, dxdy, dz):
        """
        Convenience method to calculate the sum transport over
        the vertices traversed by the path data.

        Args:

        * uv:
            A :class:`numpy.ndarray` of combined U and V data.

        * dxdy:
            A :class:`numpy.ndarray` of combined DX and DY data.

        * dz:
            A :class:`numpy.ndarray` of combined DZ data.

        Returns:
            The sum transport :class:`numpy.ndarray`.

        """
        path_transport = self.path_transport(uv, dxdy, dz)
        return path_transport.sum(axis=-1)


###############################################################################


class TCell(object):
    def __init__(self, cube, hosts=None, cache_dir=None, ping=True):
        """
        Determine the T-cells participating in one or more lines
        of constant latitude.

        Args:

        * cube:
            The T-cell :class:`iris.cube.Cube`.

        Kwargs:

        * hosts:
            A sequence of one or more hostnames designated as
            potential processing resources.

        * cache_dir:
            Directory where intermediate processing results are cached.

        * ping:
            Determine the availability of the hosts before calling
            on them as a processing resource. Defaults to True.

        """
        self.cube = cube
        self._assert_cube()
        self.hosts = self.remote_hosts(hosts, ping)
        if not self.hosts:
            self.hosts = [os.uname()[1]]
        self.cache_dir = cache_dir
        self._hash = self._generate_hash()
        if cache_dir is not None:
            self.cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            else:
                if not os.path.isdir(cache_dir):
                    raise ValueError('Invalid cache directory.')
        self._cache = set()
        self._populate_cache()

    def remote_hosts(self, hosts, ping=True):
        """
        Determine the availability of a farm of remote hosts.

        Always filters out the current hostname.

        Args:

        * hosts:
            A sequence of one or more hostnames designated as
            potential processing resources.

        Kwargs:

        * ping:
            Determine the availability of the hosts before calling
            on them as a processing resource. Defaults to True.

        Returns:
            A list of remote hostnames.

        """
        available = []
        if hosts is None:
            hosts = []
        if not isinstance(hosts, Iterable):
            hosts = [hosts]
        for host in hosts:
            # Don't include the current host.
            if host != os.uname()[1]:
                if ping:
                    try:
                        subprocess.check_output(['ping', '-c1', host])
                        available.append(host)
                    except subprocess.CalledProcessError:
                        # Host invalid or unavailable.
                        pass
                else:
                    available.append(host)
        return available

    def _assert_cube(self):
        """Sanity check the cube."""
        assert len(self.cube.coords('latitude')) == 1
        assert len(self.cube.coords('longitude')) == 1
        lat = self.cube.coord('latitude')
        lon = self.cube.coord('longitude')
        assert lat.shape == lon.shape
        assert self.cube.coord_dims(lat) == self.cube.coord_dims(lon)
        assert lat.has_bounds()
        assert lon.has_bounds()
        assert lat.bounds.shape == lon.bounds.shape
        assert lat.bounds.shape[-1] == 4
        assert lon.bounds.shape[-1] == 4

    def _generate_hash(self):
        """Attempt to generate a unique identifier for the cube."""
        lat = self.cube.coord('latitude').bounds.flatten()
        lon = self.cube.coord('longitude').bounds.flatten()
        _hash = '{}{}'.format(hash(tuple(lon)),
                              hash(tuple(lat)))
        return _hash

    def _populate_cache(self):
        """Populate the cache with pre-existing cache blobs."""
        if self.cache_dir is not None:
            for cache in glob.iglob(os.path.join(self.cache_dir, '*.tcell')):
                self._cache.add(cache)

    def _purge(self):
        """Purge the cache of T-cell intermediate results."""
        for cache in self._cache:
            try:
                os.remove(cache)
            except OSError:
                pass
        self._cache = set()

    @staticmethod
    def purge(cache_dir):
        """Purge the cache directory of cache files."""
        cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        if os.path.isdir(cache_dir):
            for cache in glob.iglob(os.path.join(cache_dir, '*.tcell')):
                try:
                    os.remove(cache)
                except OSError:
                    pass

    def cache_name(self, latitudes=None, indicies=None):
        """
        Generate a unique cache filename.

        Cache filename generate based on the T-cell :class:`iris.cube.Cube`
        and the constant lines of latitude or i-space indicies.

        Kwargs:

        * latitudes:
            A sequence of one or more constant lines of latitude.

        * indicies:
            A sequence of :class:`numpy.ndarray` indicies.

        Returns:
           Unique cache filename within the nominated cache directory.

        """
        name = None
        if self.cache_dir:
            ext = type(self).__name__.lower()
            if latitudes is None and indicies is None:
                _hash = '{}.{}'.format(self._hash, ext)
            elif latitudes is not None:
                _hash = '{}{}.mask.{}'.format(self._hash,
                                              hash(tuple(sorted(latitudes))),
                                              ext)
            else:
                hash_indicies = hash(tuple(map(tuple, indicies)))
                _hash = '{}{}.rings.{}'.format(self._hash,
                                               hash_indicies,
                                               ext)
            name = os.path.join(self.cache_dir, _hash)
        return name

    def _generate_cell_rings(self, indicies):
        """
        Generate a :class:`shapely.geometry.polygon.LinearRing` for the bounds
        of each target T-cell.

        Args:

        * indicies:
            A sequence containing the two :class:`numpy.ndarray` indicies
            of the target T-cells.

        Returns:
            A flattened :class:`numpy.ndarray` of T-cell geometries.

        """
        rings = None
        # Determine whether a ring cache can be loaded.
        if self.cache_dir is not None:
            cache_name = self.cache_name(indicies=indicies)
            if cache_name in self._cache:
#                sys.stderr.write('loading ring cache ... ')
                try:
                    with open(cache_name, 'r') as fh:
                        _, rings = cPickle.load(fh)
                except OSError:
                    self._purge()
#                sys.stderr.write('done!\n')

        # Generate the cell rings if no cache is available.
        if rings is None:
            lat = self.cube.coord('latitude')
            lon = self.cube.coord('longitude')
            rings = np.empty(indicies[0].size, dtype=np.object)

#            sys.stderr.write('Generating cell rings ... ')
            for index, (i, j) in enumerate(zip(*indicies)):
                cell = zip(lon.bounds[i, j].flatten(),
                           lat.bounds[i, j].flatten())
                rings[index] = LinearRing(cell)
#            sys.stderr.write('done!\n')

            # Determine whether to cache the rings.
            if self.cache_dir is not None:
#                sys.stderr.write('writing ring cache ... ')
                try:
                    with open(cache_name, 'w') as fh:
                        cPickle.dump((indicies, rings), fh,
                                     cPickle.HIGHEST_PROTOCOL)
                    self._cache.add(cache_name)
                except OSError:
                    self._purge()
#                sys.stderr.write('done!\n')

        return rings

    def _reduce(self, latitudes):
        """
        Limit the search space to those T-cells approximately around
        the one or more constant lines of latitude.

        Args:

        * latitudes:
            One or more target latitudes, in Geodetic decimal degrees.

        Returns:
            A tuple pair contaning the final mask shape, and the i-j
            indicies of the target T-cells.

        """
        coord = self.cube.coord('latitude')
        shape = coord.shape

        # Discover the maximum bounded cell tolerance.
        bounds = coord.bounds
        diff = map(np.max, [np.diff(bounds, axis=0), np.diff(bounds, axis=1)])
        tolerance = np.round(np.max(diff), decimals=2)

        # Identify target T-cells.
        indicies = set()
        for latitude in latitudes:
            min_lat = latitude - tolerance
            max_lat = latitude + tolerance
            temp = np.where(bounds >= min_lat, bounds, np.inf)
            temp = np.where(temp <= max_lat, temp, np.inf)
            ii, jj, _ = np.where(temp != np.inf)
            # Determine unique pairs.
            for i, j in zip(ii, jj):
                indicies.add((i, j))

        ii = []
        jj = []
        for (i, j) in indicies:
            ii.append(i)
            jj.append(j)

        indicies = (np.array(ii), np.array(jj))
        return shape, indicies

    def latitude(self, latitudes):
        """
        Determine the T-cells participating in one or more lines of
        constant latitude.

        Args:

        * latitudes:
            One or more target latitudes, in Geodetic decimal degrees.

        Returns:
            A boolean :class:`numpy.ndarray` mask, where False represents
            a T-cell that participates in a target latitude.

        """
        mask = None
        if not isinstance(latitudes, Iterable):
            latitudes = [latitudes]

        # Determine whether the mask cache can be loaded.
        if self.cache_dir is not None:
            cache_name = self.cache_name(latitudes=latitudes)
            if cache_name in self._cache:
#                sys.stderr.write('loading mask cache ... ')
                try:
                    with open(cache_name, 'r') as fh:
                        _, mask = cPickle.load(fh)
                except OSError:
                    self._purge()
#                sys.stderr.write('done!\n')

        if mask is None:
            # Generate projected lines of constant latitude.
            lines = []
            projection = ccrs.PlateCarree()
            source = ccrs.PlateCarree()
            for latitude in latitudes:
                line = LineString([(-180., latitude), (180., latitude)])
                lines.append(projection.project_geometry(line, source))

            # Optimisation - reduce the bounded T-cell search space.
            final_shape, indicies = self._reduce(latitudes)
            mask = np.ones(final_shape, dtype=np.bool)

            # Generate the geometries for the gridded cells.
            rings = self._generate_cell_rings(indicies)

            payload = []
            step = int(math.ceil(rings.size / float(len(self.hosts))))
            for offset in range(0, rings.size, step):
                data = (lines, rings[offset:offset + step])
                payload.append(cPickle.dumps(data, cPickle.HIGHEST_PROTOCOL))

            pool = multiprocessing.Pool()
            chunks = 1
            result = pool.map(_client, zip(self.hosts, payload), chunks)
            pool.close()
            pool.join()
            mask_result = np.concatenate(result)
            ii, jj = indicies
            mask[ii, jj] = mask_result
            del rings

            # Determine whether to cache the mask.
            if self.cache_dir is not None:
#                sys.stderr.write('writing mask cache ... ')
                try:
                    with open(cache_name, 'w') as fh:
                        cPickle.dump((latitudes, mask), fh,
                                     cPickle.HIGHEST_PROTOCOL)
                    self._cache.add(cache_name)
                except OSError:
                    self._purge()
#                sys.stderr.write('done!\n')

        return mask


if __name__ == '__main__':
    #
    # Load the T, U and V gridded cubes.
    #
    tuv_dname = '/project/hadgem3/data/anbag/ony'
    tuv_fnames = ['anbago_1y_19781201_19791130_grid_T.nc',
                  'anbago_1y_19781201_19791130_grid_U.nc',
                  'anbago_1y_19781201_19791130_grid_V.nc']
    tuv_constraints = ['sea_water_potential_temperature',
                       'sea_water_x_velocity',
                       'sea_water_y_velocity']
    grid_cubes = Ocean.load_grid_cubes(tuv_fnames, tuv_constraints,
                                       prefix=tuv_dname)

    #
    # Load the region mask.
    #
    mask_fname = '/project/ujcc/CDFTOOLS/mesh_ORCA025L75/' \
        'subbasins_orca025_070909_rename.nc'
    mask_constraint = Ocean.named_constraint('tmaskatl')
    region_mask = Ocean.load_region_mask(mask_fname, mask_constraint)

    #
    # Load the mesh DX, DY, DZU and DZV cubes.
    #
    mesh_fname = '/project/ujcc/CDFTOOLS/mesh_ORCA025L75/mesh_mask_GO5.nc'
    mesh_names = ['e1v', 'e2v', 'e3u', 'e3v']
    mesh_constraints = [Ocean.named_constraint(name) for name in mesh_names]
    mesh_cubes = Ocean.load_mesh_cubes(mesh_fname, mesh_constraints)

    ocean = Ocean(grid_cubes, mesh_cubes)
