# (C) British Crown Copyright 2013, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Provides loader cache support for quicker loading.

.. warning::

    TODO ...

"""

from collections import Iterable, MutableMapping, namedtuple
import cPickle
import cStringIO
import itertools
import os
import site
import struct
import tempfile
import warnings

import iris


_DEFAULT_BASE = os.path.join(site.getuserbase(), 'share', 'iris', 'cache')
_WORD_DEPTH = 4   # In bytes.
_HEADER_SIZE = 2  # In words.
_DEFAULT_LWM = 100 * 1024 * 1024  # Cacheable Low Water Mark - in bytes.


def writeable(directory):
    """
    Determines whether the provided directory has sufficient permissions
    to allow writing.

    Args:
    * directory:
        String directory or if filename, parent directory to be verified.

    Returns:
        Boolean.

    """
    result = True
    directory = os.path.abspath(os.path.expanduser(directory))
    if os.path.isfile(directory):
        directory = os.path.dirname(directory)
    try:
        fname = tempfile.mkstemp(dir=directory)[1]
        with open(fname, 'w') as fh:
            fh.write('Iris')
        os.remove(fname)
    except (IOError, OSError):
        result = False
    return result


def readable(filename):
    """
    Determines whether the provided filename has sufficient permissions
    to allow reading.

    Args:
    * filename:
        String filename to be verified.

    Returns:
        Boolean.

    """
    result = True
    filename = os.path.abspath(os.path.expanduser(filename))
    try:
        with open(filename, 'r') as fh:
            pass
    except (IOError, OSError):
        result = False
    return result


def _valid_base(base):
    """Ensure base directory is writable, if not return writable default."""
    if base is not None:
        base = os.path.abspath(os.path.expanduser(base))
        if not (os.path.isdir(base) and writeable(base)):
            base = None
    if base is None:
        dname = _DEFAULT_BASE
        if not os.path.isdir(dname):
            os.makedirs(dname)
        base = dname
    return base


class IndexKey(namedtuple('IndexKey', 'fname constraints raw')):
    """
    Named tuple representing a hashable key for the cache index lookup table.

    Args:
    * fname:
        The names of one or more source files to be loaded.

    * constraints:
        One or more loader :class:`iris.Constraint`. No constraint
        may be represented by None.

    * raw:
        Boolean representing whether the combination of fname,
        constraints (and callback) are hashable.

    """
    def __new__(cls, fname, constraints, callback):
        raw = False
        if fname is None:
            fname = []
        if isinstance(fname, basestring) or not isinstance(fname, Iterable):
            fname = [fname]
        fname = sorted([os.path.abspath(os.path.expanduser(fn))
                        for fn in fname])
        constraints = iris._constraints.list_of_constraints(constraints)
        for item in constraints:
            if item.callable():
                constraints = [iris.Constraint()]
                raw = True
                break
        if callable(callback) or (isinstance(callback, bool) and callback):
            constraints = [iris.Constraint()]
            raw = True
        return super(IndexKey, cls).__new__(cls, tuple(fname),
                                            tuple(constraints),
                                            raw)


class IndexItem(namedtuple('IndexItem', 'cache mtime size')):
    """
    Named tuple representing a cache index lookup table entry.

    Args:
    * cache:
        The name of the loader cache file.

    * mtime:
        The modification time of each source file to be loaded.
        See :func:`os.path.getmtime`.

    * size:
        The size (in bytes) of each source file to be loaded.
        See :func:`os.path.getsize`.

    """
    def __new__(cls, cache, mtime, size):
        cache = os.path.abspath(os.path.expanduser(cache))
        if isinstance(mtime, Iterable) and not isinstance(mtime, basestring):
            mtime = tuple(mtime)
        else:
            mtime = (mtime,)
        if isinstance(size, Iterable) and not isinstance(size, basestring):
            size = tuple(size)
        else:
            size = (size,)
        if len(mtime) != len(size):
            msg = '{} contains unequal length fields.'.format(cls.__name__)
            raise ValueError(msg)
        return super(IndexItem, cls).__new__(cls, cache, mtime, size)


class _Index(MutableMapping):
    """
    Supports the cache index lookup table, mapping load queries
    to cached content.

    """
    def __init__(self, *args, **kwargs):
        """
        Represents the cache index lookup table. This hash table provides
        quick lookup capability in order to map a loader query to cached
        content.

        """
        self._index = {}
        self.strict = kwargs.pop('strict', False)
        self.base = _valid_base(kwargs.pop('base', None))
        self._fname = 'cache.index'
        self._read()
        self.update(dict(*args, **kwargs))

    def __repr__(self):
        index = self._index.copy()
        index.update(dict(base=self.base, strict=self.strict))
        args = []
        for key, value in index.iteritems():
            args.append('{!r}: {!r}'.format(key, value))
        return '{}({})'.format(_Index.__name__, ', '.join(args))

    def __getitem__(self, key):
        return self._index[key]

    def __setitem__(self, key, item):
        if not isinstance(key, IndexKey):
            raise TypeError('Expected an {!r} instance, '
                            'got {!r}.'.format(IndexKey.__name__, type(key)))
        if not isinstance(item, IndexItem):
            raise TypeError('Expected an {!r} instance, '
                            'got {!r}.'.format(IndexItem.__name__, type(item)))
        # Must be a 1-to-1 mapping between fname to mtime & size.
        if len(key.fname) != len(item.mtime):
            raise ValueError('Invalid index key/item pair.')
        self._index[key] = item
        self._write()

    def __delitem__(self, key):
        del self._index[key]
        self._write()

    def __iter__(self):
        return iter(self._index)

    def __len__(self):
        return len(self._index)

    def _read(self):
        """Unpickle archived cache index from file."""
        fname = os.path.join(self.base, self._fname)
        if os.path.isfile(fname):
            with open(fname, 'rb') as fh:
                self._index = cPickle.load(fh)
        if self.strict:
            self.validate()

    def _write(self):
        """Pickle the cache index to file."""
        if self.strict:
            self.validate()
        fname = os.path.join(self.base, self._fname)
        with open(fname, 'wb') as fh:
            cPickle.dump(self._index, fh, cPickle.HIGHEST_PROTOCOL)

    def add(self, key, directory=None):
        """
        Insert a new index key, item pair into the cache index.

        Args:
        * key:
            The :class:`IndexKey` cache index key.

        Kwargs:
        * directory:
            The target directory for the cached content.
            Defaults to the cache index base directory.

        Returns:
            The :class:`IndexItem` cache index entry.

        """
        if not isinstance(key, IndexKey):
            raise TypeError('Expected an {!r} instance, '
                            'got {!r}.'.format(IndexKey.__name__, type(key)))
        if directory is None:
            directory = self.base
        # Create a unique name for the target cache file.
        cache = self.cache_name(key, directory)
        # Collate the modification time and size of each source file.
        mtime = []
        size = []
        for fname in key.fname:
            if not os.path.isfile(fname):
                msg = 'Index key file {!r} does not exist.'.format(fname)
                raise IOError(msg)
            if not readable(fname):
                msg = 'Cannot open file {!r} for reading.'.format(fname)
                raise IOError(msg)
            mtime.append(os.path.getmtime(fname))
            size.append(os.path.getsize(fname))
        # Create the associated cache index entry item.
        item = IndexItem(cache, mtime, size)
        # Add the entry to the cache index.
        self[key] = item
        return item

    def as_raw(self, key):
        """
        Convert the cache index key to a raw key.

        Args:
        * key:
            The :class:`IndexKey` cache index key to be converted.

        Returns:
            The raw :class:`IndexKey`.

        """
        return IndexKey(key.fname, key.constraints, lambda x: x)

    def validate(self):
        """
        Ensure that the cache index is not stale, and consistent with
        the source file/s referenced by the cached content.

        All stale entries are purged from the cache index. An entry is
        deemed stale whenever a referenced source file has a different
        modification time or size, a source file is missing or non-readable.

        The file archive of the cache index is automatically updated.

        """
        invalid = []
        # Validate each cache index entry.
        for key, item in self._index.iteritems():
            for fname, mtime, size in itertools.izip(key.fname,
                                                     item.mtime,
                                                     item.size):
                if os.path.isfile(fname) and readable(fname):
                    actual = os.path.getmtime(fname), os.path.getsize(fname)
                    if actual != (mtime, size):
                        invalid.append((key, item))
                        break
                else:
                    invalid.append((key, item))
                    break
        # Purge all stale entries.
        for key, item in invalid:
            try:
                # Attempt to remove the stale cache file.
                if os.path.isfile(item.cache):
                    os.remove(item.cache)
            except OSError:
                msg = 'Failed to remove stale ' \
                    'cache file {!r}.'.format(item.cache)
                warnings.warn(msg)
            # Remove the stale cache index entry.
            del self._index[key]

    @staticmethod
    def cache_name(key, directory):
        """
        Generate a unique name for a cache file.

        Args:
        * key:
            The :class:`IndexKey` cache index key.

        * directory:
            The target directory of the cache file.

        Returns:
            The cache filename.

        """
        if not isinstance(key, IndexKey):
            raise TypeError('Expected an {!r} instance, '
                            'got {!r}.'.format(IndexKey.__name__, type(key)))
        if len(key.fname) == 1:
            fname = key.fname[0].split('/', 1)[1].replace('/', '-')
            if key.raw:
                fname += '.raw'
            else:
                fname = '{}.{}'.format(fname, hash(key.constraints))
        else:
            fname = '{}'.format(hash(key))
            if key.raw:
                fname += '.raw'
        return os.path.join(directory, fname)


class Cache(object):
    """
    Provides support for managing a cache index and the loading and saving
    of cached content.

    """
    MAGIC = 0xEF0FF5C1

    def __init__(self, active=False, cache_base=None, index_base=None,
                 lwm=None, strict=False, sync=False):
        """
        Manages the cache index lookup table, index key and item generation,
        and the loading/saving behaviour of cached content.

        Args:
        * active:
            Boolean flag controls whether loader caching is performed.

        * cache_base:
            The directory where source file cached content is stored.
            If set to None, then the cache file will be stored in the
            first writable directory of a source file. Otherwise, the
            cache file will be stored in the :data:`index_base` directory.

        * index_base:
            The directory where the cache index archive file is stored.

        * lwm:
            The Low-Water-Mark (i.e. minimum) threshold in bytes which
            the total source file size must meet or exceed to be cached.

        * strict:
            Boolean flag that determines whether the cache index verifies
            its cached content when reading and writing to archive.

        * sync:
            Boolean flag that determines whether the :data:`cache_base`
            directory is synchronized with the :data:`index_base`
            directory.

        """
        self.active = active
        self.cache_base = cache_base
        self.index_base = index_base
        self.lwm = lwm
        if self.lwm is None:
            self.lwm = _DEFAULT_LWM
        self.strict = strict
        self.sync = sync
        self._index = _Index(base=self.index_base, strict=self.strict)
        # The following attributes are the source for key generation
        # i.e. the loader candidate key.
        self.constraints = None
        self.fname = None
        self.callback = None
        self.force_raw = False

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name == 'cache_base':
            if 'sync' in self.__dict__ and 'index_base' in self.__dict__:
                if self.sync:
                    self.__dict__[name] = self.index_base
                elif value is not None:
                    self.__dict__[name] = _valid_base(value)
            elif value is not None:
                self.__dict__[name] = _valid_base(value)
        elif name == 'constraints':
            self.__dict__['fname'] = None
            self.__dict__['callback'] = None
            self.__dict__['force_raw'] = False
        elif name == 'fname' and value is not None:
            if isinstance(value, basestring) or \
                    not isinstance(value, Iterable):
                value = [value]
            self.__dict__[name] = sorted(value)
        elif name == 'index_base':
            self.__dict__[name] = _valid_base(value)
            if '_index' in self.__dict__ and 'strict' in self.__dict__:
                self.__dict__['_index'] = _Index(base=self.index_base,
                                                 strict=self.strict)
            if 'cache_base' in self.__dict__ and 'sync' in self.__dict__:
                if self.sync:
                    self.__dict__['cache_base'] = self.index_base
        elif name == 'sync' and value:
            if 'cache_base' in self.__dict__ and 'index_base' in self.__dict__:
                self.__dict__['cache_base'] = self.index_base

    def __repr__(self):
        args = []
        for name in sorted(self.__dict__):
            if not name.startswith('_'):
                args.append('{}={!r}'.format(name, self.__dict__[name]))
        return '{}({})'.format(Cache.__name__, ', '.join(args))

    def _read_header(self, fh):
        """
        Read and return the cache file header, consisting of the header id
        and number of cache file entries.

        """
        fh.seek(0)
        fmt = '>{}I'.format(_HEADER_SIZE)
        return struct.unpack_from(fmt, fh.read(_HEADER_SIZE * _WORD_DEPTH))

    def _write_header(self, fh, count):
        """
        Write the cache file header, consisting of the header id and
        number of cache file entries.

        """
        fh.seek(0)
        fmt = '>{}I'.format(_HEADER_SIZE)
        fh.write(struct.pack(fmt, self.MAGIC, count))

    def _valid_header_id(self, key):
        """
        Determines whether the cache file associated with the cache index key
        exists and contains a valid header.

        If invalid, the associated cache index entry is purged.

        """
        valid = False
        item = self._index[key]
        if os.path.isfile(item.cache):
            with open(item.cache, 'rb') as fh:
                try:
                    magic, _ = self._read_header(fh)
                    valid = magic == self.MAGIC
                except struct.error:
                    pass
        if not valid:
            del self._index[key]
        return valid

    def is_cacheable(self):
        """
        Determine whether the cache is active and whether the collective
        source files to be loaded  are sufficiently large enough to
        warrant the effort of caching.

        Returns:
            Boolean.

        """
        result = False
        if self.active:
            key = IndexKey(self.fname, self.constraints, self.callback)
            total = sum([os.path.getsize(fname) for fname in key.fname])
            result = total >= self.lwm
        return result

    def is_full_key(self):
        """
        Determine whether the loader candidate key is not raw i.e. the
        loader query of filenames, constraints and callback are
        hashable and do not contain any callables.

        Returns:
            Boolean.

        """
        key = IndexKey(self.fname, self.constraints, self.callback)
        return not key.raw

    def full_key_available(self):
        """
        Determine whether the loader candidate key is not raw, is
        already present in the cache index and the associated cache file
        is valid.

        Returns:
            Boolean.

        """
        result = False
        key = IndexKey(self.fname, self.constraints, self.callback)
        if not key.raw and not self.force_raw:
            result = key in self._index and self._valid_header_id(key)
        return result

    def raw_key_available(self):
        """
        Determine whether the raw loader candidate key, is already
        present in the cache index and the associated cache file is
        valid.

        Returns:
            Boolean.

        """
        result = False
        key = IndexKey(self.fname, self.constraints, self.callback)
        raw = self._index.as_raw(key)
        return raw in self._index and self._valid_header_id(raw)

    def _load_common(self, key, item):
        count = 0
        with open(item.cache, 'rb') as fh:
            _, expected = self._read_header(fh)
            while True:
                try:
                    payload = cPickle.load(fh)
                    count += 1
                    yield payload
                except EOFError:
                    if count != expected:
                        plural = 's' if count > 1 else ''
                        cache_type = 'Raw' if key.raw else 'Full'
                        msg = '{} cache {!r} contains {} pickle object{}, ' \
                            'expected {}'.format(cache_type, item.cache, count,
                                                 plural, expected)
                        warnings.warn(msg)
                        del self._index[key]
                    break

    def load_full_cache(self):
        """
        Given the current loader candidate key, load the associated
        full cache result.

        Returns:
            The cached :class:`iris.cube.Cube` or :class:`iris.cube.CubeList`.

        """
        print 'Loading full cache ...'
        key = IndexKey(self.fname, self.constraints, self.callback)
        item = self._index[key]
        return self._load_common(key, item)

    def load_raw_cache(self):
        """
        Given the current loader candidate key, load the associated
        raw cache result.

        Returns:
            The cached :class:`iris.cube.CubeList`.

        """
        print 'Loading raw cache ...'
        key = IndexKey(self.fname, self.constraints, self.callback)
        raw = self._index.as_raw(key)
        item = self._index[raw]
        return self._load_common(raw, item)

    def _resolve_cache_base(self, fnames):
        cache_base = self.cache_base
        if isinstance(fnames, basestring) or not isinstance(fnames, Iterable):
            fnames = [fnames]
        if cache_base is None:
            bases = sorted({os.path.dirname(fname) for fname in fnames})
            for base in bases:
                if writeable(base):
                    cache_base = base
                    break
            if cache_base is None:
                cache_base = self.index_base
        return cache_base

    def save_full_cache(self, cubes):
        """
        Given the current loader candidate key, create an entry in the
        cache index and pickle the cubes.

        Args:
        * cubes:
            A :class:`iris.cube.CubeList` to be cached.

        """
        print 'Saving full cache ...'
        key = IndexKey(self.fname, self.constraints, self.callback)
        if key.raw:
            msg = 'Unexpectedly received raw cache key when saving full cache.'
            raise ValueError(msg)
        cache_base = self._resolve_cache_base(key.fname)
        item = self._index.add(key, directory=cache_base)
        with open(item.cache, 'wb') as fh:
            self._write_header(fh, len(cubes))
            dump = cPickle.dump
            for cube in cubes:
                dump(cube, fh, cPickle.HIGHEST_PROTOCOL)

    def save_raw_cache(self, cubes):
        """
        Given the current loader candidate key, create an entry in the
        cache index and pickle the cubes.

        Args:
        * cubes:
            A generator of cubes to be cached.

        Yields:
            The cached :class:`iris.cube.Cube`.

        """
        print 'Saving raw cache ...'
        key = IndexKey(self.fname, self.constraints, self.callback)
        raw = self._index.as_raw(key)
        cache_base = self._resolve_cache_base(key.fname)
        item = self._index.add(raw, directory=cache_base)
        with open(item.cache, 'wb') as fh:
            self._write_header(fh, 0)
            dump = cPickle.dump
            for count, cube in enumerate(cubes):
                dump(cube, fh, cPickle.HIGHEST_PROTOCOL)
                yield cube
            fh.seek(_WORD_DEPTH)
            fh.write(struct.pack('>I', count + 1))
