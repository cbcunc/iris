# -*- coding: iso-8859-1 -*-
import iris.tests as tests
import iris.tests.stock
import numpy as np
import numpy.ma as ma

class TestPeakAggregator(tests.IrisTest):
    def test_peak_with_mask(self):
	#single value in column masked
	latitude = iris.coords.DimCoord(range(0, 4, 1), standard_name='latitude', units='degrees')
	cube = iris.cube.Cube(ma.array([1, 2, 3, 1], dtype=np.float32), 
			      standard_name='air_temperature',
			      units='kelvin')
	cube.add_dim_coord(latitude, 0)
	cube.data[2] = ma.masked

	#result will be different if _peak changed to remove nans and interpolate the data, 
	#instead of taking the maximum value in the column
	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	#np.testing.assert_array_almost_equal(collapsed_cube.data, np.array([2], dtype=np.float32))
	self.assertArrayAlmostEqual(collapsed_cube.data, np.array([2], dtype=np.float32))

	#whole column masked
	#cube.data[:] = ma.masked

	#result will be different if _peak changed to remove nans and interpolate the data, 
	#instead of taking the maximum value in the column
	#collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	#masked_array = ma.array(ma.masked)
	#print collapsed_cube.data, masked_array
	#self.assertMaskedArrayAlmostEqual(collapsed_cube.data, masked_array)

	latitude = iris.coords.DimCoord(range(0, 2, 1), standard_name='latitude', units='degrees')
	longitude = iris.coords.DimCoord(range(0, 2, 1), standard_name='longitude', units='degrees')
	cube = iris.cube.Cube(ma.array([[1, 2], [1, 2]], dtype=np.float32), 
			      standard_name='air_temperature',
			      units='kelvin')
	cube.add_dim_coord(latitude, 0)
	cube.add_dim_coord(longitude, 1)
	cube.data[:] = ma.masked

	collapsed_cube = cube.collapsed('latitude', iris.analysis.PEAK, wibble=True)
	masked_array = ma.array([np.nan, np.nan], dtype=np.float32)
	masked_array[:] = ma.masked
	print 'collapsed = ', collapsed_cube.data.data
	print 'masked = ', masked_array.data
	self.assertMaskedArrayAlmostEqual(collapsed_cube.data, masked_array)

if __name__ == "__main__":
    tests.main()