# -*- coding: iso-8859-1 -*-
import iris.tests as tests
import iris.tests.stock
import numpy as np
import numpy.ma as ma

class TestPeakAggregator(tests.IrisTest):
    def test_peak_coord_length_1(self):
	#coordinate contains a single point
	latitude = iris.coords.DimCoord(np.array([0]), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(np.array([1]), 
			      standard_name='air_temperature', 
			      units='kelvin')
	cube.add_dim_coord(latitude, 0)

	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
        self.assertArrayAlmostEqual(collapsed_cube.data, np.array([1], dtype=np.float32))

    def test_peak_coord_length_2(self):
	#coordinate contains 2 points
	latitude = iris.coords.DimCoord(range(0, 2, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(np.array([1, 2]), 
			      standard_name='air_temperature', 
			      units='kelvin')
	cube.add_dim_coord(latitude, 0)

	#result will be different if _peak changed to raise an error if no fitted peak found, instead 
	#of taking the maximum value in the column
	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
        self.assertArrayAlmostEqual(collapsed_cube.data, np.array([2], dtype=np.float32))

    def test_peak_coord_length_3(self):
	#coordinate contains 3 points
	latitude = iris.coords.DimCoord(range(0, 3, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(np.array([1, 2, 1]), 
			      standard_name='air_temperature',
			      units='kelvin')
	cube.add_dim_coord(latitude, 0)

	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
        self.assertArrayAlmostEqual(collapsed_cube.data, np.array([2], dtype=np.float32))

    def test_peak_1d(self):
	#collapse a 1d cube
	latitude = iris.coords.DimCoord(range(0, 11, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]), 
			      standard_name='air_temperature', 
			      units='kelvin')
	cube.add_dim_coord(latitude, 0)

        collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
        self.assertArrayAlmostEqual(collapsed_cube.data, np.array([6], dtype=np.float32))

    def test_peak_duplicate_coords(self):
	#collapse cube along 2 coordinates - both the same
	latitude = iris.coords.DimCoord(range(0, 4, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(np.array([1, 2, 3, 1]), 
			      standard_name='air_temperature', 
			      units='kelvin')
	cube.add_dim_coord(latitude, 0)

	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	self.assertArrayAlmostEqual(collapsed_cube.data, np.array([3], dtype=np.float32))

	collapsed_cube = cube.collapsed(('latitude', 'latitude'), iris.analysis.PEAK, wibble=True)
	self.assertArrayAlmostEqual(collapsed_cube.data, np.array([3], dtype=np.float32))

    def test_peak_2d(self):
	#collapse a 2d cube
	longitude = iris.coords.DimCoord(range(0, 4, 1), standard_name='longitude', units='degrees')
	latitude = iris.coords.DimCoord(range(0, 3, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(np.array([[1, 2, 3, 1], [4, 5, 6, 4], [2, 3, 4, 2]]), 
			      standard_name='air_temperature', 
			      units='kelvin')
	cube.add_dim_coord(latitude, 0)
	cube.add_dim_coord(longitude, 1)

	collapsed_cube = cube.collapsed('longitude', iris.analysis.PEAK, wibble=True)
	self.assertArrayAlmostEqual(collapsed_cube.data, np.array([3, 6, 4], dtype=np.float32))

	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	self.assertArrayAlmostEqual(collapsed_cube.data, np.array([4.025, 5.025, 6.025, 4.025], dtype=np.float32))

	collapsed_cube = cube.collapsed(('longitude', 'latitude'), iris.analysis.PEAK, wibble=True)
	self.assertArrayAlmostEqual(collapsed_cube.data, np.array([6.025], dtype=np.float32))

	collapsed_cube = cube.collapsed(('latitude', 'longitude'), iris.analysis.PEAK, wibble=True)
	self.assertArrayAlmostEqual(collapsed_cube.data, np.array([6.025], dtype=np.float32))
    
    def test_peak_without_peak_value(self):
	#no peak in column - values equal
	latitude = iris.coords.DimCoord(range(0, 4, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(np.array([1, 1, 1, 1]), 
			      standard_name='air_temperature', 
			      units='kelvin')
	cube.add_dim_coord(latitude, 0)

	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	self.assertArrayAlmostEqual(collapsed_cube.data, np.array([1], dtype=np.float32))

	#no peak in column
	cube = iris.cube.Cube(np.array([1, 2, 3, 4]), 
			      standard_name='air_temperature', 
			      units='kelvin')
	cube.add_dim_coord(latitude, 0)

	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	self.assertArrayAlmostEqual(collapsed_cube.data, np.array([4], dtype=np.float32))

    def test_peak_with_nan(self):
	#single nan in column
	latitude = iris.coords.DimCoord(range(0, 4, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(np.array([1, 2, 3, 1], dtype=np.float32), 
			      standard_name='air_temperature', 
			      units='kelvin')
	cube.add_dim_coord(latitude, 0)

	cube.data[2] = np.nan

	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	self.assertArrayAlmostEqual(collapsed_cube.data, np.array([2], dtype=np.float32))
	
	#only nans in column
	cube.data[:] = np.nan

	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	self.assertArrayAlmostEqual(collapsed_cube.data, np.array([np.nan], dtype=np.float32))

    def test_peak_with_mask(self):
	#single value in column masked
	latitude = iris.coords.DimCoord(range(0, 4, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(ma.array([1, 2, 3, 2], dtype=np.float32), 
			      standard_name='air_temperature',
			      units='kelvin')
	cube.add_dim_coord(latitude, 0)

	cube.data[2] = ma.masked

	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	self.assertArrayAlmostEqual(collapsed_cube.data, np.array([2], dtype=np.float32))

	#whole column masked
	cube.data[:] = ma.masked

	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	masked_array = ma.array(ma.masked)
	#self.assertMaskedArrayAlmostEqual(collapsed_cube.data, masked_array)

	#latitude = iris.coords.DimCoord(range(0, 2, 1), standard_name='latitude', units='degrees')
	#longitude = iris.coords.DimCoord(range(0, 2, 1), standard_name='longitude', units='degrees')
	#cube = iris.cube.Cube(ma.array([[1, 2], [1, 2]], dtype=np.float32), 
	#		      standard_name='air_temperature',
	#		      units='kelvin')
	#cube.add_dim_coord(latitude, 0)
	#cube.add_dim_coord(longitude, 1)

	#cube.data[:] = ma.masked
	#collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)

	#masked_array = ma.array([np.nan, np.nan], dtype=np.float32)
	#masked_array[:] = ma.masked

	#self.assertMaskedArrayAlmostEqual(collapsed_cube.data, masked_array)
	np.testing.assert_equal(True, ma.allequal(collapsed_cube.data, masked_array))

    def test_peak_with_nan_and_mask(self):
	#single nan in column with single value masked
	latitude = iris.coords.DimCoord(range(0, 4, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(ma.array([1, 2, 3, 1], dtype=np.float32), 
			      standard_name='air_temperature', 
			      units='kelvin')
	cube.add_dim_coord(latitude, 0)

	cube.data[2] = np.nan
	cube.data[3] = ma.masked

	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	self.assertArrayAlmostEqual(collapsed_cube.data, np.array([2], dtype=np.float32))
	
	#only nans in column where values not masked
	cube.data[0:2] = np.nan

	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	self.assertArrayAlmostEqual(collapsed_cube.data, np.array([np.nan], dtype=np.float32))

if __name__ == "__main__":
    tests.main()