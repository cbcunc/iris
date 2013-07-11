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
"""Test load caching."""

# Import iris tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import glob
import os
import site
import stat
import tempfile
import time
import warnings

import iris
import iris.fileformats._cache as _cache


class TestIndexKeyItem(tests.IrisTest):
    def test_index_key(self):
        def func(args):
            pass

        key = _cache.IndexKey(None, None, None)
        self.assertEqual(key.fname, ())
        self.assertEqual(key.constraints, (iris.Constraint(),))
        self.assertFalse(key.raw)

        fname = '/one/two'
        constraint = iris.Constraint(model=10)
        callback = None

        key = _cache.IndexKey(fname, constraint, callback)
        self.assertEqual(key.fname, (fname,))
        self.assertEqual(key.constraints, (constraint,))
        self.assertFalse(key.raw)

        with self.assertRaises(TypeError):
            _cache.IndexKey(fname, 1.618, callback)

        key = _cache.IndexKey(fname, 'wibble', callback)
        self.assertEqual(key.fname, (fname,))
        self.assertEqual(key.constraints, (iris.Constraint('wibble'),))
        self.assertFalse(key.raw)

        key = _cache.IndexKey(fname, iris.Constraint(model=lambda x: x),
                              callback)
        self.assertEqual(key.fname, (fname,))
        self.assertEqual(key.constraints, (iris.Constraint(),))
        self.assertTrue(key.raw)

        key = _cache.IndexKey(fname, iris.Constraint(model=func), callback)
        self.assertEqual(key.fname, (fname,))
        self.assertEqual(key.constraints, (iris.Constraint(),))
        self.assertTrue(key.raw)

        key = _cache.IndexKey(fname, iris.Constraint(cube_func=func), callback)
        self.assertEqual(key.fname, (fname,))
        self.assertEqual(key.constraints, (iris.Constraint(),))
        self.assertTrue(key.raw)

        fnames = ['/one/two', '/three/four']
        key = _cache.IndexKey(fnames, constraint, constraint)
        self.assertEqual(key.fname, tuple(fnames))
        self.assertEqual(key.constraints, (constraint,))
        self.assertFalse(key.raw)

        key = _cache.IndexKey(fname, ['wibble', 'wobble', constraint],
                              callback)
        self.assertEqual(key.fname, (fname,))
        self.assertEqual(key.constraints, (iris.Constraint('wibble'),
                                           iris.Constraint('wobble'),
                                           constraint))
        self.assertFalse(key.raw)

        key = _cache.IndexKey(fname, ['wibble', iris.Constraint(model=func)],
                              callback)
        self.assertEqual(key.fname, (fname,))
        self.assertEqual(key.constraints, (iris.Constraint(),))
        self.assertTrue(key.raw)

        key = _cache.IndexKey(fname, constraint, func)
        self.assertEqual(key.fname, (fname,))
        self.assertEqual(key.constraints, (iris.Constraint(),))
        self.assertTrue(key.raw)

    def test_index_item(self):
        cache, mtime, size = '/one/two', 10, 20
        item = _cache.IndexItem(cache, mtime, size)
        self.assertEqual(item.cache, cache)
        self.assertEqual(item.mtime, (mtime,))
        self.assertEqual(item.size, (size,))

        mtime = size = range(5)
        item = _cache.IndexItem(cache, mtime, size)
        self.assertEqual(item.cache, cache)
        self.assertEqual(item.mtime, tuple(mtime))
        self.assertEqual(item.size, tuple(size))

        mtime, size = range(5), range(10)
        with self.assertRaises(ValueError):
            _cache.IndexItem(cache, mtime, size)


class TestWriteable(tests.IrisTest):
    def setUp(self):
        self.base = tempfile.mkdtemp()

    def tearDown(self):
        mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
        os.chmod(self.base, mode)
        target = os.path.join(self.base, '*')
        for fname in glob.glob(target):
            os.remove(fname)
        os.rmdir(self.base)

    def test_writeable_valid(self):
        self.assertTrue(_cache.writeable(self.base))

        fname = os.path.join(self.base, 'wibble')
        open(fname, 'w')
        self.assertTrue(_cache.writeable(fname))

    def test_writeable_invalid(self):
        self.assertFalse(_cache.writeable('/wibble/wobble'))

        mode = stat.S_IRUSR | stat.S_IXUSR
        os.chmod(self.base, mode)
        self.assertFalse(_cache.writeable(self.base))


class TestIndex(tests.IrisTest):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.fname1 = os.path.join(self.base, 'one')
        open(self.fname1, 'wb').write('1')
        self.fname2 = os.path.join(self.base, 'two')
        open(self.fname2, 'wb').write('12')
        self.fname3 = os.path.join(self.base, 'three')
        open(self.fname3, 'wb').write('123')
        self.fnames = [self.fname1, self.fname2, self.fname3]

    def tearDown(self):
        target = os.path.join(self.base, '*')
        for fname in glob.glob(target):
            os.remove(fname)
        os.rmdir(self.base)

    def test_init_simple(self):
        index = _cache._Index()
        self.assertEqual(index.base, _cache._DEFAULT_BASE)

        index = _cache._Index(base=self.base)
        self.assertEqual(index.base, self.base)

    def test_cache_name(self):
        with self.assertRaises(TypeError):
            _cache._Index.cache_name('wibble', None)

        key = _cache.IndexKey('/one/two/three', None, None)
        cache = _cache._Index.cache_name(key, '/wibble/wobble')
        expected = '{}.{}'.format('/wibble/wobble/one-two-three',
                                  hash((iris.Constraint(),)))
        self.assertEqual(cache, expected)

        key = _cache.IndexKey('/one/two/three', None, lambda x: x)
        cache = _cache._Index.cache_name(key, '/wibble/wobble')
        expected = '{}.raw'.format('/wibble/wobble/one-two-three')
        self.assertEqual(cache, expected)

        fname = ['/one/two/three', '/four/five/six']
        key = _cache.IndexKey(fname, None, None)
        cache = _cache._Index.cache_name(key, '/wibble/wobble')
        expected = '/wibble/wobble/{}'.format(hash(key))
        self.assertEqual(cache, expected)

        fname = ['/one/two/three', '/four/five/six']
        key = _cache.IndexKey(fname, None, lambda x: x)
        cache = _cache._Index.cache_name(key, '/wibble/wobble')
        expected = '/wibble/wobble/{}.raw'.format(hash(key))
        self.assertEqual(cache, expected)

    def test_index_primatives(self):
        index = _cache._Index(base=self.base)
        for fname in self.fnames:
            key = _cache.IndexKey(fname, None, None)
            index.add(key)
        self.assertEqual(len(index), 3)

        key = _cache.IndexKey(self.fname1, None, None)
        self.assertTrue(index[key])
        key = _cache.IndexKey(self.fname2, None, None)
        self.assertTrue(index[key])
        key = _cache.IndexKey(self.fname3, None, None)
        self.assertTrue(index[key])

        with self.assertRaises(KeyError):
            index[_cache.IndexKey('/wibble', None, None)]

        self.assertIsNone(index.get(None))
        self.assertIsNotNone(index.get(key))

        del index[key]
        with self.assertRaises(KeyError):
            index[key]
        self.assertEqual(len(index), 2)

    def test_index_set(self):
        index = _cache._Index(base=self.base)
        with self.assertRaises(TypeError):
            index['wibble'] = 'wobble'

        key = _cache.IndexKey(self.fname1, None, None)
        with self.assertRaises(TypeError):
            index[key] = 'wobble'

        item = _cache.IndexItem('/one/two/three', range(5), range(5))
        with self.assertRaises(ValueError):
            index[key] = item

        item = _cache.IndexItem('/one/two/three', 1, 2)
        index[key] = item
        self.assertEqual(len(index), 1)
        self.assertEqual(index[key], item)

    def test_index_add_bad(self):
        index = _cache._Index(base=self.base)
        with self.assertRaises(TypeError):
            index.add('wibble')

        key = _cache.IndexKey('/wibble/wobble', None, None)
        with self.assertRaises(IOError):
            index.add(key)

    def test_index_add_single(self):
        index = _cache._Index(base=self.base)
        key = _cache.IndexKey(self.fname1, None, None)
        item = index.add(key)
        self.assertEqual(len(index), 1)
        self.assertEqual(index[key], item)
        self.assertEqual(os.path.dirname(item.cache), self.base)
        self.assertEqual(item.size, (1,))

        key = _cache.IndexKey(self.fname2, None, None)
        item = index.add(key, directory='/wibble/wobble')
        self.assertEqual(len(index), 2)
        self.assertEqual(index[key], item)
        self.assertEqual(os.path.dirname(item.cache), '/wibble/wobble')
        self.assertEqual(item.size, (2,))

    def test_index_add_multi(self):
        index = _cache._Index(base=self.base)
        key = _cache.IndexKey(self.fnames, None, None)
        item = index.add(key)
        self.assertEqual(len(index), 1)
        self.assertEqual(index[key], item)
        self.assertEqual(os.path.dirname(item.cache), self.base)
        self.assertEqual(item.size, (1, 3, 2))

    def test_strict_read(self):
        writer = _cache._Index(base=self.base)
        key = _cache.IndexKey(self.fname1, None, None)
        item = writer.add(key)
        self.assertEqual(len(writer), 1)
        open(item.cache, 'w')
        os.remove(self.fname1)
        reader = _cache._Index(base=self.base)
        self.assertEqual(len(reader), 1)
        self.assertEqual(reader[key], item)
        strict = _cache._Index(base=self.base, strict=True)
        self.assertEqual(len(strict), 0)

    def test_strict_write(self):
        index = _cache._Index(base=self.base)
        for i, fname in enumerate([self.fname1, self.fname2]):
            key = _cache.IndexKey(fname, None, None)
            item = index.add(key)
            self.assertEqual(len(index), i + 1)
            open(item.cache, 'w')
            os.remove(fname)
        index.strict = True
        key = _cache.IndexKey(self.fname3, None, None)
        item = index.add(key)
        self.assertEqual(len(index), 1)
        self.assertEqual(index[key], item)
        reader = _cache._Index(base=self.base)
        self.assertEqual(len(reader), 1)

    def test_validate(self):
        index = _cache._Index(base=self.base)
        key = _cache.IndexKey(self.fname1, None, None)
        item = index.add(key)
        self.assertEqual(len(index), 1)
        open(item.cache, 'w')
        index.validate()
        self.assertEqual(len(index), 1)

    def test_validate_file_missing(self):
        index = _cache._Index(base=self.base)
        key = _cache.IndexKey(self.fname1, None, None)
        item = index.add(key)
        self.assertEqual(len(index), 1)
        open(item.cache, 'w')
        os.remove(self.fname1)
        index.validate()
        self.assertEqual(len(index), 0)
        with self.assertRaises(IOError):
            open(item.cache, 'r')

    def test_validate_file_mtime(self):
        index = _cache._Index(base=self.base)
        key = _cache.IndexKey(self.fname1, None, None)
        item = index.add(key)
        self.assertEqual(len(index), 1)
        open(item.cache, 'w')
        # Delay to ensure difference modification time.
        time.sleep(1)
        with open(self.fname1, 'wb') as fh:
            fh.write('1')
        index.validate()
        self.assertEqual(len(index), 0)
        with self.assertRaises(IOError):
            open(item.cache, 'r')

    def test_validate_file_size(self):
        index = _cache._Index(base=self.base)
        key = _cache.IndexKey(self.fname1, None, None)
        item = index.add(key)
        self.assertEqual(len(index), 1)
        open(item.cache, 'w')
        with open(self.fname1, 'wb') as fh:
            fh.write('12')
        index.validate()
        self.assertEqual(len(index), 0)
        with self.assertRaises(IOError):
            open(item.cache, 'r')

    def test_cache_index_write_read(self):
        writer = _cache._Index(base=self.base)
        for fname in self.fnames:
            key = _cache.IndexKey(fname, None, None)
            writer.add(key)
        reader = _cache._Index(base=self.base)
        self.assertEqual(writer._index, reader._index)


class TestCache(tests.IrisTest):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.fname1 = os.path.join(self.base, 'one')
        open(self.fname1, 'wb').write('1')
        self.fname2 = os.path.join(self.base, 'two')
        open(self.fname2, 'wb').write('12')
        self.fname3 = os.path.join(self.base, 'three')
        open(self.fname3, 'wb').write('123')
        self.fnames = [self.fname1, self.fname2, self.fname3]

    def tearDown(self):
        target = os.path.join(self.base, '*')
        for fname in glob.glob(target):
            os.remove(fname)
        os.rmdir(self.base)

    def test_default_init(self):
        cache = _cache.Cache()
        self.assertFalse(cache.active)
        self.assertIsNone(cache.cache_base)
        self.assertEqual(cache.index_base, _cache._DEFAULT_BASE)
        self.assertEqual(cache.lwm, _cache._DEFAULT_LWM)
        self.assertFalse(cache.strict)
        self.assertFalse(cache.sync)
        self.assertIsNone(cache.constraints)
        self.assertIsNone(cache.fname)
        self.assertIsNone(cache.callback)

    def test_attributes(self):
        cache = _cache.Cache()

        cache.cache_base = None
        self.assertIsNone(cache.cache_base)
        cache.cache_base = '/wibble/wobble'
        self.assertEqual(cache.cache_base, _cache._DEFAULT_BASE)
        cache.cache_base = '~'
        self.assertEqual(cache.cache_base, os.path.expanduser('~'))

        cache.fname = 'wibble'
        self.assertEqual(cache.fname, ['wibble'])
        cache.fname = ['wobble', 'wibble']
        self.assertEqual(cache.fname, ['wibble', 'wobble'])
        cache.callback = open
        self.assertEqual(cache.callback, open)
        cache.force_raw = True
        self.assertTrue(cache.force_raw)
        cache.constraints = iris.Constraint()
        self.assertEqual(cache.constraints, iris.Constraint())
        self.assertIsNone(cache.fname)
        self.assertIsNone(cache.callback)
        self.assertFalse(cache.force_raw)

        cache.index_base = None
        self.assertEqual(cache.index_base, _cache._DEFAULT_BASE)
        cache.index_base = '/wibble/wobble'
        self.assertEqual(cache.index_base, _cache._DEFAULT_BASE)
        cache.index_base = '~'
        self.assertEqual(cache.cache_base, os.path.expanduser('~'))

        cache.cache_base = None
        cache.index_base = None
        cache.sync = True
        self.assertEqual(cache.cache_base, cache.index_base)
        cache.cache_base = '/wibble/wobble'
        self.assertEqual(cache.cache_base, cache.index_base)
        cache.index_base = '~'
        self.assertEqual(cache.cache_base, os.path.expanduser('~'))

    def test_header(self):
        cache = _cache.Cache(index_base=self.base)
        fname = os.path.join(self.base, 'cache')
        expected = 123

        with open(fname, 'wb') as fh:
            cache._write_header(fh, expected)
        with open(fname, 'rb') as fh:
            magic, actual = cache._read_header(fh)
        self.assertEqual(magic, cache.MAGIC)
        self.assertEqual(actual, expected)

        key = _cache.IndexKey(self.fname1, None, None)
        item = cache._index.add(key)
        self.assertEqual(len(cache._index), 1)
        self.assertFalse(cache._valid_header_id(key))
        self.assertEqual(len(cache._index), 0)

        item = cache._index.add(key)
        self.assertEqual(len(cache._index), 1)
        with open(item.cache, 'wb') as fh:
            fh.write('wibble')
        self.assertFalse(cache._valid_header_id(key))
        self.assertEqual(len(cache._index), 0)

        item = cache._index.add(key)
        self.assertEqual(len(cache._index), 1)
        with open(item.cache, 'wb') as fh:
            cache._write_header(fh, 0)
        self.assertTrue(cache._valid_header_id(key))

    def test_cacheable(self):
        cache = _cache.Cache(index_base=self.base)

        self.assertFalse(cache.is_cacheable())
        cache.active = True
        self.assertFalse(cache.is_cacheable())

        cache.lwm = 3
        for fname, expected in zip(self.fnames, [False, False, True]):
            cache.fname = fname
            self.assertEqual(cache.is_cacheable(), expected)
        cache.fname = self.fnames
        self.assertTrue(cache.is_cacheable())

    def test_is_full_key(self):
        cache = _cache.Cache(index_base=self.base)

        self.assertTrue(cache.is_full_key())
        cache.fname = '/wibble/wobble'
        self.assertTrue(cache.is_full_key())
        cache.callback = open
        self.assertFalse(cache.is_full_key())
        cache.constraints = iris.Constraint(model=lambda x: x)
        self.assertFalse(cache.is_full_key())
        cache.constraints = iris.Constraint(model=open)
        self.assertFalse(cache.is_full_key())

    def test_full_key_available(self):
        cache = _cache.Cache(index_base=self.base)

        self.assertFalse(cache.full_key_available())
        cache.fname = self.fname1
        self.assertFalse(cache.full_key_available())
        key = _cache.IndexKey(self.fname1, None, None)
        item = cache._index.add(key)
        self.assertFalse(cache.full_key_available())
        self.assertEqual(len(cache._index), 0)
        item = cache._index.add(key)
        with open(item.cache, 'wb') as fh:
            cache._write_header(fh, 0)
        self.assertTrue(cache.full_key_available())
        self.assertEqual(len(cache._index), 1)
        cache.force_raw = True
        self.assertFalse(cache.full_key_available())
        self.assertEqual(len(cache._index), 1)

        cache.callback = open
        self.assertFalse(cache.full_key_available())

    def test_raw_key_available(self):
        cache = _cache.Cache(index_base=self.base)

        self.assertFalse(cache.raw_key_available())
        cache.fname = self.fname1
        cache.callback = open
        self.assertFalse(cache.raw_key_available())
        raw = cache._index.as_raw(_cache.IndexKey(self.fname1, None, None))
        item = cache._index.add(raw)
        self.assertFalse(cache.raw_key_available())
        self.assertEqual(len(cache._index), 0)
        item = cache._index.add(raw)
        with open(item.cache, 'wb') as fh:
            cache._write_header(fh, 0)
        self.assertTrue(cache.raw_key_available())
        self.assertEqual(len(cache._index), 1)

    def test_cache_base(self):
        cache = _cache.Cache(index_base=self.base)
        dname1 = tempfile.mkdtemp(dir=self.base)
        fname1 = os.path.join(dname1, 'fname1')
        open(fname1, 'w').close()
        dname2 = tempfile.mkdtemp(dir=self.base)
        fname2 = os.path.join(dname2, 'fname2')
        open(fname2, 'w').close()
        mode = stat.S_IRUSR | stat.S_IXUSR
        os.chmod(dname2, mode)

        cache.cache_base = '/wibble/wobble'
        self.assertEqual(cache._resolve_cache_base(fname1), cache.cache_base)
        cache.cache_base = None
        self.assertEqual(cache._resolve_cache_base(fname1), dname1)
        self.assertEqual(cache._resolve_cache_base(fname2), cache.index_base)
        self.assertEqual(cache._resolve_cache_base([fname1, fname2]), dname1)
        os.chmod(dname1, mode)
        self.assertEqual(cache._resolve_cache_base([fname1, fname2]),
                         cache.index_base)

        for dname, fname in zip([dname1, dname2], [fname1, fname2]):
            mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
            os.chmod(dname, mode)
            os.remove(fname)
            os.rmdir(dname)

    def test_full_cycle(self):
        cache = _cache.Cache(index_base=self.base, sync=True)
        expected = range(10)

        cache.callback = open
        with self.assertRaises(ValueError):
            cache.save_full_cache(None)

        cache.constraints = None
        cache.fname = self.fname1
        cache.save_full_cache(expected)
        self.assertTrue(cache.full_key_available())
        self.assertEqual([item for item in cache.load_full_cache()], expected)

    def test_raw_cycle(self):
        cache = _cache.Cache(index_base=self.base, sync=True)
        expected = range(10)

        cache.fname = self.fname1
        [item for item in cache.save_raw_cache(expected)]
        self.assertTrue(cache.raw_key_available())
        self.assertEqual([item for item in cache.load_raw_cache()], expected)


if __name__ == "__main__":
    tests.main()
