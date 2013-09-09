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

import iris.tests as tests
import iris.tests.stock
import numpy as np
import numpy.ma as ma


class TestPeakAggregator(tests.IrisTest):
    def test_peak_coord_length_1(self):
        # Coordinate contains a single point.
        latitude = iris.coords.DimCoord(np.array([0]),
                                        standard_name='latitude',
                                        units='degrees')
        cube = iris.cube.Cube(np.array([1]),
                              standard_name='air_temperature',
                              units='kelvin')
        cube.add_dim_coord(latitude, 0)

        collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK)
        self.assertArrayAlmostEqual(collapsed_cube.data,
                                    np.array([1], dtype=np.float32))

    def test_peak_coord_length_2(self):
        # Coordinate contains 2 points.
        latitude = iris.coords.DimCoord(range(0, 2, 1),
                                        standard_name='latitude',
                                        units='degrees')
        cube = iris.cube.Cube(np.array([1, 2]),
                              standard_name='air_temperature',
                              units='kelvin')
        cube.add_dim_coord(latitude, 0)

        collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK)
        self.assertArrayAlmostEqual(collapsed_cube.data,
                                    np.array([2], dtype=np.float32))

    def test_peak_coord_length_3(self):
        # Coordinate contains 3 points.
        latitude = iris.coords.DimCoord(range(0, 3, 1),
                                        standard_name='latitude',
                                        units='degrees')
        cube = iris.cube.Cube(np.array([1, 2, 1]),
                              standard_name='air_temperature',
                              units='kelvin')
        cube.add_dim_coord(latitude, 0)

        collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK)
        self.assertArrayAlmostEqual(collapsed_cube.data,
                                    np.array([2], dtype=np.float32))

    def test_peak_1d(self):
        # Collapse a 1d cube.
        latitude = iris.coords.DimCoord(range(0, 11, 1),
                                        standard_name='latitude',
                                        units='degrees')
        cube = iris.cube.Cube(np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]),
                              standard_name='air_temperature',
                              units='kelvin')
        cube.add_dim_coord(latitude, 0)

        collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK)
        self.assertArrayAlmostEqual(collapsed_cube.data,
                                    np.array([6], dtype=np.float32))

    def test_peak_duplicate_coords(self):
        # Collapse cube along 2 coordinates (both the same).
        latitude = iris.coords.DimCoord(range(0, 4, 1),
                                        standard_name='latitude',
                                        units='degrees')
        cube = iris.cube.Cube(np.array([1, 2, 3, 1]),
                              standard_name='air_temperature',
                              units='kelvin')
        cube.add_dim_coord(latitude, 0)

        collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK)
        self.assertArrayAlmostEqual(collapsed_cube.data,
                                    np.array([3], dtype=np.float32))

        collapsed_cube = cube.collapsed(('latitude', 'latitude'),
                                        iris.analysis.PEAK)
        self.assertArrayAlmostEqual(collapsed_cube.data,
                                    np.array([3], dtype=np.float32))

    def test_peak_2d(self):
        # Collapse a 2d cube.
        longitude = iris.coords.DimCoord(range(0, 4, 1),
                                         standard_name='longitude',
                                         units='degrees')
        latitude = iris.coords.DimCoord(range(0, 3, 1),
                                        standard_name='latitude',
                                        units='degrees')
        cube = iris.cube.Cube(np.array([[1, 2, 3, 1], [4, 5, 6, 4],
                                        [2, 3, 4, 2]]),
                              standard_name='air_temperature',
                              units='kelvin')

        cube.add_dim_coord(latitude, 0)
        cube.add_dim_coord(longitude, 1)

        collapsed_cube = cube.collapsed('longitude', iris.analysis.PEAK)
        self.assertArrayAlmostEqual(collapsed_cube.data,
                                    np.array([3, 6, 4], dtype=np.float32))

        collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK)
        self.assertArrayAlmostEqual(collapsed_cube.data,
                                    np.array([4.024977, 5.024977,
                                              6.024977, 4.024977],
                                             dtype=np.float32))

        collapsed_cube = cube.collapsed(('longitude', 'latitude'),
                                        iris.analysis.PEAK)
        self.assertArrayAlmostEqual(collapsed_cube.data,
                                    np.array([6.024977], dtype=np.float32))

        collapsed_cube = cube.collapsed(('latitude', 'longitude'),
                                        iris.analysis.PEAK)
        self.assertArrayAlmostEqual(collapsed_cube.data,
                                    np.array([6.024977], dtype=np.float32))

    def test_peak_without_peak_value(self):
        # No peak in column (values equal).
        latitude = iris.coords.DimCoord(range(0, 4, 1),
                                        standard_name='latitude',
                                        units='degrees')
        cube = iris.cube.Cube(np.array([1, 1, 1, 1]),
                              standard_name='air_temperature',
                              units='kelvin')
        cube.add_dim_coord(latitude, 0)

        collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK)
        self.assertArrayAlmostEqual(collapsed_cube.data,
                                    np.array([1], dtype=np.float32))

        # No peak in column.
        cube = iris.cube.Cube(np.array([1, 2, 3, 4]),
                              standard_name='air_temperature',
                              units='kelvin')
        cube.add_dim_coord(latitude, 0)

        collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK)
        self.assertArrayAlmostEqual(collapsed_cube.data,
                                    np.array([4], dtype=np.float32))

    def test_peak_with_nan(self):
        # Single nan in column.
        latitude = iris.coords.DimCoord(range(0, 5, 1),
                                        standard_name='latitude',
                                        units='degrees')
        cube = iris.cube.Cube(np.array([1, 4, 2, 3, 1], dtype=np.float32),
                              standard_name='air_temperature',
                              units='kelvin')
        cube.add_dim_coord(latitude, 0)

        cube.data[3] = np.nan

        collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK)
        self.assertArrayAlmostEqual(collapsed_cube.data,
                                    np.array([4.024977], dtype=np.float32))

        # Only nans in column.
        cube.data[:] = np.nan

        collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK)
        self.assertTrue(np.isnan(collapsed_cube.data).all())

    def test_peak_with_mask(self):
        # Single value in column masked.
        latitude = iris.coords.DimCoord(range(0, 5, 1),
                                        standard_name='latitude',
                                        units='degrees')
        cube = iris.cube.Cube(ma.array([1, 4, 2, 3, 2], dtype=np.float32),
                              standard_name='air_temperature',
                              units='kelvin')
        cube.add_dim_coord(latitude, 0)

        cube.data[3] = ma.masked

        collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK)
        self.assertArrayAlmostEqual(collapsed_cube.data,
                                    np.array([4.024977], dtype=np.float32))

        # Whole column masked.
        cube.data[:] = ma.masked

        collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK)
        masked_array = ma.array(ma.masked)
        np.testing.assert_equal(True, ma.allequal(collapsed_cube.data,
                                                  masked_array))

    def test_peak_with_nan_and_mask(self):
        # Single nan in column with single value masked.
        latitude = iris.coords.DimCoord(range(0, 5, 1),
                                        standard_name='latitude',
                                        units='degrees')
        cube = iris.cube.Cube(ma.array([1, 4, 2, 3, 1], dtype=np.float32),
                              standard_name='air_temperature',
                              units='kelvin')
        cube.add_dim_coord(latitude, 0)

        cube.data[3] = np.nan
        cube.data[4] = ma.masked

        collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK)
        self.assertArrayAlmostEqual(collapsed_cube.data,
                                    np.array([4.024977], dtype=np.float32))

        # Only nans in column where values not masked.
        cube.data[0:3] = np.nan

        collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK)
        self.assertTrue(np.isnan(collapsed_cube.data).all())

if __name__ == "__main__":
    tests.main()
