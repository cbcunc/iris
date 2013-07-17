# -*- coding: iso-8859-1 -*-
import iris.tests as tests
import iris.tests.stock
import numpy as np
import numpy.ma as ma

class TestPeakAggregator(tests.IrisTest):
    def test_peak_coord_length_1(self):
	#coordinate contains a single point
	latitude = iris.coords.DimCoord(np.array([0]), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(np.array([1]), standard_name='air_temperature', 
			  units='kelvin', dim_coords_and_dims=[(latitude, 0)])

	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
        np.testing.assert_array_almost_equal(collapsed_cube.data, np.array([1], dtype=np.float32))
	#self.assertCML(collapsed_cube, ('analysis', 'peak_length_1.cml'), checksum=False)

    def test_peak_coord_length_2(self):
	#coordinate contains 2 points
	latitude = iris.coords.DimCoord(range(0, 2, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(np.array([1, 2]), 
			  standard_name='air_temperature', units='kelvin', 
			  dim_coords_and_dims=[(latitude, 0)])

	#result will be different if _peak changed to raise an error if no fitted peak found, instead 
	#of taking the maximum value in the column
	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
        np.testing.assert_array_almost_equal(collapsed_cube.data, np.array([2], dtype=np.float32))
	#self.assertCML(collapsed_cube, ('analysis', 'peak_length_2.cml'), checksum=False)

    def test_peak_coord_length_3(self):
	#coordinate contains 3 points
	latitude = iris.coords.DimCoord(range(0, 3, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(np.array([1, 2, 1]), 
			  standard_name='air_temperature', units='kelvin', 
			  dim_coords_and_dims=[(latitude, 0)])

	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
        np.testing.assert_array_almost_equal(collapsed_cube.data, np.array([2], dtype=np.float32))
	#self.assertCML(collapsed_cube, ('analysis', 'peak_length_3.cml'), checksum=False)

    def test_peak_1d(self):
	#collapse a 1d cube
	latitude = iris.coords.DimCoord(range(0, 11, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]), standard_name='air_temperature', 
			  units='kelvin', dim_coords_and_dims=[(latitude, 0)])

        collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
        np.testing.assert_array_almost_equal(collapsed_cube.data, np.array([6], dtype=np.float32))
        #collapsed_cube.data = collapsed_cube.data.astype('i8')
        #self.assertCML(collapsed_cube, ('analysis', 'peak_1d.cml'), checksum=False)

    def test_peak_duplicate_coords(self):
	#collapse cube along 2 coordinates - both the same
	latitude = iris.coords.DimCoord(range(0, 4, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(np.array([1, 2, 3, 1]), 
			  standard_name='air_temperature', units='kelvin', 
			  dim_coords_and_dims=[(latitude, 0)])

	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	np.testing.assert_array_almost_equal(collapsed_cube.data, np.array([3], dtype=np.float32))
        #self.assertCML(collapsed_cube, ('analysis', 'peak_latitude_duplicate.cml'), checksum=False)

	collapsed_cube = cube.collapsed(('latitude', 'latitude'), iris.analysis.PEAK, wibble=True)
	np.testing.assert_array_almost_equal(collapsed_cube.data, np.array([3], dtype=np.float32))
	#self.assertCML(collapsed_cube, ('analysis', 'peak_latitude_latitude_duplicate.cml'), checksum=False)

    def test_peak_2d(self):
	#collapse a 2d cube
	longitude = iris.coords.DimCoord(range(0, 4, 1), standard_name='longitude', units='degrees')
	latitude = iris.coords.DimCoord(range(0, 3, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(np.array([[1, 2, 3, 1], [4, 5, 6, 4], [2, 3, 4, 2]]), 
			  standard_name='air_temperature', units='kelvin', 
			  dim_coords_and_dims=[(latitude, 0), (longitude, 1)])

	collapsed_cube = cube.collapsed('longitude', iris.analysis.PEAK, wibble=True)
	np.testing.assert_array_almost_equal(collapsed_cube.data, np.array([3, 6, 4], dtype=np.float32))
        #self.assertCML(collapsed_cube, ('analysis', 'peak_longitude_2d.cml'), checksum=False)

	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	np.testing.assert_array_almost_equal(collapsed_cube.data, np.array([4.025, 5.025, 6.025, 4.025], dtype=np.float32))
        #self.assertCML(collapsed_cube, ('analysis', 'peak_latitude_2d.cml'), checksum=False)

	collapsed_cube = cube.collapsed(('longitude', 'latitude'), iris.analysis.PEAK, wibble=True)
	np.testing.assert_array_almost_equal(collapsed_cube.data, np.array([6.025], dtype=np.float32))
	#self.assertCML(collapsed_cube, ('analysis', 'peak_longitude_latitude_2d.cml'), checksum=False)

	collapsed_cube = cube.collapsed(('latitude', 'longitude'), iris.analysis.PEAK, wibble=True)
	np.testing.assert_array_almost_equal(collapsed_cube.data, np.array([6.025], dtype=np.float32))
	#self.assertCML(collapsed_cube, ('analysis', 'peak_latitude_longitude_2d.cml'), checksum=False)
    
    def test_peak_without_peak_value(self):
	#no peak in column
	latitude = iris.coords.DimCoord(range(0, 4, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(np.array([1, 1, 1, 1]), 
			  standard_name='air_temperature', units='kelvin', 
			  dim_coords_and_dims=[(latitude, 0)])

	#result will be different if _peak changed to raise an error if no fitted peak found, instead 
	#of taking the maximum value in the column
	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	np.testing.assert_array_almost_equal(collapsed_cube.data, np.array([1], dtype=np.float32))
        #self.assertCML(collapsed_cube, ('analysis', 'peak_equal.cml'), checksum=False)

	cube = iris.cube.Cube(np.array([1, 2, 3, 4]), 
			  standard_name='air_temperature', units='kelvin', 
			  dim_coords_and_dims=[(latitude, 0)])

	#result will be different if _peak changed to raise an error if no fitted peak found, instead 
	#of taking the maximum value in the column
	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	np.testing.assert_array_almost_equal(collapsed_cube.data, np.array([4], dtype=np.float32))
        #self.assertCML(collapsed_cube, ('analysis', 'peak_no_peak.cml'), checksum=False)

    def test_peak_with_nan(self):
	#single nan in column
	latitude = iris.coords.DimCoord(range(0, 4, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(np.array([1, 2, 3, 1]), 
			  standard_name='air_temperature', units='kelvin', 
			  dim_coords_and_dims=[(latitude, 0)])

        data = cube.data.astype(np.float32)
        cube.data = data.copy()
	cube.data[2] = np.nan

	#result will be different if _peak changed to remove nans and interpolate the data, 
	#instead of taking the maximum value in the column
	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	np.testing.assert_array_almost_equal(collapsed_cube.data, np.array([2], dtype=np.float32))
	#self.assertCML(collapsed_cube, ('analysis', 'peak_single_nan.cml'), checksum=False)
	
	#only nans in column
	cube.data[0] = np.nan
	cube.data[1] = np.nan
	cube.data[3] = np.nan

	#result will be different if _peak changed to remove nans and interpolate the data, 
	#instead of taking the maximum value in the column
	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	np.testing.assert_array_almost_equal(collapsed_cube.data, np.array([np.nan], dtype=np.float32))
	#self.assertCML(collapsed_cube, ('analysis', 'peak_nan_column.cml'), checksum=False)

    def test_peak_with_mask(self):
	#single value in column masked
	latitude = iris.coords.DimCoord(range(0, 4, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(np.array([1, 2, 3, 1]), 
			  standard_name='air_temperature', units='kelvin', 
			  dim_coords_and_dims=[(latitude, 0)])

        data = cube.data.astype(np.float32)
        cube.data = data.copy()
	cube.data[2] = np.nan

	cube_with_mask = iris.cube.Cube(np.array([1, 1, 1, 1]), 
				    standard_name='air_temperature', units='kelvin', 
				    dim_coords_and_dims=[(latitude, 0)])
        cube_with_mask.data = ma.array(cube.data, mask=np.isnan(cube.data))

	#result will be different if _peak changed to remove nans and interpolate the data, 
	#instead of taking the maximum value in the column
	collapsed_cube = cube_with_mask.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	np.testing.assert_array_almost_equal(collapsed_cube.data, np.array([2], dtype=np.float32))
	#self.assertCML(collapsed_cube, ('analysis', 'peak_single_value_masked.cml'), checksum=False)

	#whole column masked
	cube.data[0] = np.nan
	cube.data[1] = np.nan
	cube.data[3] = np.nan

	cube_with_mask.data = ma.array(cube.data, mask=np.isnan(cube.data))

	#result will be different if _peak changed to remove nans and interpolate the data, 
	#instead of taking the maximum value in the column
	collapsed_cube = cube_with_mask.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	np.testing.assert_array_almost_equal(collapsed_cube.data, np.array([np.nan], dtype=np.float32))
	#self.assertCML(collapsed_cube, ('analysis', 'peak_column_masked.cml'), checksum=False)

if __name__ == "__main__":
    tests.main()