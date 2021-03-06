# (C) British Crown Copyright 2010 - 2012, Met Office
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
Test cube indexing, slicing, and extracting, and also the dot graphs.

"""
# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import os
import re

import numpy

import iris
import iris.analysis
import iris.coords
import iris.cube
import iris.fileformats
import iris.unit
import iris.tests.pp as pp
import iris.tests.stock


class IrisDotTest(tests.IrisTest):
    def check_dot(self, cube, reference_filename):
        test_string = iris.fileformats.dot.cube_text(cube)
        reference_path = tests.get_result_path(reference_filename)
        if os.path.isfile(reference_path):
            reference = ''.join(open(reference_path, 'r').readlines())
            self._assert_str_same(reference, test_string, reference_filename, type_comparison_name='DOT files')
        else:
            tests.logger.warning('Creating result file: %s', reference_path)
            open(reference_path, 'w').writelines(test_string)


class TestBasicCubeConstruction(tests.IrisTest):
    def setUp(self):
        self.cube = iris.cube.Cube(numpy.arange(12, dtype=numpy.int32).reshape((3, 4)), long_name='test cube')
        self.x = iris.coords.DimCoord(numpy.array([ -7.5,   7.5,  22.5,  37.5]), long_name='x')
        self.y = iris.coords.DimCoord(numpy.array([  2.5,   7.5,  12.5]), long_name='y')
        self.xy = iris.coords.AuxCoord(numpy.arange(12).reshape((3, 4)) * 3.0, long_name='xy')

    def test_add_dim_coord(self):
        # Lengths must match
        with self.assertRaises(ValueError):
            self.cube.add_dim_coord(self.y, 1)
        with self.assertRaises(ValueError):
            self.cube.add_dim_coord(self.x, 0)

        # Must specify a dimension
        with self.assertRaises(ValueError):
            self.cube.add_dim_coord(self.y)

        # Add y
        self.cube.add_dim_coord(self.y, 0)
        self.assertEqual(self.cube.coords(), [self.y])
        self.assertEqual(self.cube.dim_coords, (self.y,))
        # Add x
        self.cube.add_dim_coord(self.x, 1)
        self.assertEqual(self.cube.coords(), [self.y, self.x])
        self.assertEqual(self.cube.dim_coords, (self.y, self.x))

        # Cannot add a coord twice
        with self.assertRaises(ValueError):
            self.cube.add_dim_coord(self.y, 0)
        # ... even to cube.aux_coords
        with self.assertRaises(ValueError):
            self.cube.add_aux_coord(self.y, 0)

        # Can't add AuxCoord to dim_coords
        y_other = iris.coords.AuxCoord(numpy.array([  2.5,   7.5,  12.5]), long_name='y_other')
        with self.assertRaises(ValueError):
            self.cube.add_dim_coord(y_other, 0)

    def test_add_scalar_coord(self):
        scalar_dim_coord = iris.coords.DimCoord(23, long_name='scalar_dim_coord')
        scalar_aux_coord = iris.coords.AuxCoord(23, long_name='scalar_aux_coord')
        # Scalars cannot be in cube.dim_coords
        with self.assertRaises(ValueError):
            self.cube.add_dim_coord(scalar_dim_coord)
        with self.assertRaises(ValueError):
            self.cube.add_dim_coord(scalar_dim_coord, None)
        with self.assertRaises(ValueError):
            self.cube.add_dim_coord(scalar_dim_coord, [])
        with self.assertRaises(ValueError):
            self.cube.add_dim_coord(scalar_dim_coord, ())
        
        # Make sure that's still the case for a 0-dimensional cube.
        cube = iris.cube.Cube(666)
        self.assertEqual(cube.ndim, 0)
        with self.assertRaises(ValueError):
            self.cube.add_dim_coord(scalar_dim_coord)
        with self.assertRaises(ValueError):
            self.cube.add_dim_coord(scalar_dim_coord, None)
        with self.assertRaises(ValueError):
            self.cube.add_dim_coord(scalar_dim_coord, [])
        with self.assertRaises(ValueError):
            self.cube.add_dim_coord(scalar_dim_coord, ())

        cube = self.cube.copy()
        cube.add_aux_coord(scalar_dim_coord)
        cube.add_aux_coord(scalar_aux_coord)
        self.assertEqual(set(cube.aux_coords), {scalar_dim_coord, scalar_aux_coord})
        
        # Various options for dims
        cube = self.cube.copy()
        cube.add_aux_coord(scalar_dim_coord, [])
        self.assertEqual(cube.aux_coords, (scalar_dim_coord,))

        cube = self.cube.copy()
        cube.add_aux_coord(scalar_dim_coord, ())
        self.assertEqual(cube.aux_coords, (scalar_dim_coord,))

        cube = self.cube.copy()
        cube.add_aux_coord(scalar_dim_coord, None)
        self.assertEqual(cube.aux_coords, (scalar_dim_coord,))

        cube = self.cube.copy()
        cube.add_aux_coord(scalar_dim_coord)
        self.assertEqual(cube.aux_coords, (scalar_dim_coord,))

    def test_add_aux_coord(self):
        y_another = iris.coords.DimCoord(numpy.array([  2.5,   7.5,  12.5]), long_name='y_another')
        
        # DimCoords can live in cube.aux_coords
        self.cube.add_aux_coord(y_another, 0)
        self.assertEqual(self.cube.dim_coords, ())
        self.assertEqual(self.cube.coords(), [y_another])
        self.assertEqual(self.cube.aux_coords, (y_another,))

        # AuxCoords in cube.aux_coords
        self.cube.add_aux_coord(self.xy, [0, 1])
        self.assertEqual(self.cube.dim_coords, ())
        self.assertEqual(self.cube.coords(), [y_another, self.xy])
        self.assertEqual(set(self.cube.aux_coords), {y_another, self.xy})

        # Lengths must match up
        cube = self.cube.copy()
        with self.assertRaises(ValueError):
            cube.add_aux_coord(self.xy, [1, 0])

    def test_remove_coord(self):
        self.cube.add_dim_coord(self.y, 0)
        self.cube.add_dim_coord(self.x, 1)
        self.cube.add_aux_coord(self.xy, (0, 1))
        self.assertEqual(set(self.cube.coords()), {self.y, self.x, self.xy})
        
        self.cube.remove_coord('xy')
        self.assertEqual(set(self.cube.coords()), {self.y, self.x})

        self.cube.remove_coord('x')
        self.assertEqual(self.cube.coords(), [self.y])

        self.cube.remove_coord('y')
        self.assertEqual(self.cube.coords(), [])


class TestStockCubeStringRepresentations(tests.IrisTest):
    def test_4d(self):
        cube = iris.tests.stock.realistic_4d()
        self.assertString(str(cube), ('cdm', 'string_representations', 'realistic_4d.__str__.txt'))

    def test_3d(self):
        cube = iris.tests.stock.realistic_4d()
        self.assertString(str(cube[0]), ('cdm', 'string_representations', 'realistic_3d.__str__.txt'))

    def test_2d(self):
        cube = iris.tests.stock.realistic_4d()
        self.assertString(str(cube[0, 0]), ('cdm', 'string_representations', 'realistic_2d.__str__.txt'))

    def test_1d(self):
        cube = iris.tests.stock.realistic_4d()
        self.assertString(str(cube[0, 0, 0]), ('cdm', 'string_representations', 'realistic_1d.__str__.txt'))

    def test_0d(self):
        cube = iris.tests.stock.realistic_4d()
        self.assertString(str(cube[0, 0, 0, 0]), ('cdm', 'string_representations', 'realistic_0d.__str__.txt'))


@iris.tests.skip_data
class TestCubeStringRepresentations(IrisDotTest):
    def setUp(self):
        self.cube_2d = iris.load_strict(tests.get_data_path(('PP', 'simple_pp', 'global.pp')))

    def multi_dim_cubes(self):
        if not hasattr(self, '_multi_dim_cubes'):
            self._multi_dim_cubes = iris.load(tests.get_data_path(('PP', 'COLPEX', 'uwind_and_orog.pp')))
        return self._multi_dim_cubes

    def test_dot_simple_pp(self):
        # Test dot output of a 2d cube loaded from pp.
        cube = self.cube_2d
        cube.attributes['my_attribute'] = 'foobar'
        self.check_dot(cube, ('file_load', 'global_pp.dot'))
        
        pt = cube.coord('time')
        # and with custom coord attributes
        pt.attributes['monty'] = 'python'
        pt.attributes['brain'] = 'hurts'
        self.check_dot(cube, ('file_load', 'coord_attributes.dot'))
        
        del pt.attributes['monty']
        del pt.attributes['brain']
        del cube.attributes['my_attribute']
       
    # TODO hybrid height and dot output - relatitionship links
    def test_dot_5d(self):
        # Test dot output of a 5d cube loaded from pp.
        cube = [c for c in self.multi_dim_cubes() if c.name() == "eastward_wind"][0]
        self.check_dot(cube, ('file_load', '5d_pp.dot'))
        
    def test_cube_string(self):
        cube = [c for c in self.multi_dim_cubes() if c.name() == "eastward_wind"][0]
        cube.attributes['my_attribute'] = 'foobar'
        self.assertString(str(cube), ('cdm', 'string_representations', '5d_pp.__str__.txt'))
        self.assertString(repr(cube), ('cdm', 'string_representations', '5d_pp.__repr__.txt'))
        
        del cube.attributes['my_attribute']
        
    def test_multiline_history_summary(self):
        c = self.cube_2d
        # subtract two cubes from each other to make 2 lines of history
        c = (c - c) - (c - c)
        self.assertString(str(c), ('cdm', 'string_representations', 'muliple_history.__str__.txt'))
        
    def test_cubelist_string(self):
        self.assertString(str(self.multi_dim_cubes()), ('cdm', 'string_representations', 'cubelist.__str__.txt'))
        self.assertString(repr(self.multi_dim_cubes()), ('cdm', 'string_representations', 'cubelist.__repr__.txt'))

    def test_basic_0d_cube(self):
        self.assertString(repr(self.cube_2d[0, 0]), ('cdm', 'string_representations', '0d_cube.__repr__.txt'))
        self.assertString(str(self.cube_2d[0, 0]), ('cdm', 'string_representations', '0d_cube.__str__.txt'))

    def test_similar_coord(self):
        cube = self.cube_2d.copy()

        lon = cube.coord('longitude')
        lon.attributes['flight'] = '218BX'
        lon.attributes['sensor_id'] = 808
        lon.attributes['status'] = 2
        lon2 = lon.copy()
        lon2.attributes['sensor_id'] = 810
        lon2.attributes['ref'] = 'A8T-22'
        del lon2.attributes['status']
        cube.add_aux_coord(lon2, [1])

        lat = cube.coord('latitude')
        lat2 = lat.copy()
        lat2.attributes['test'] = 'True'
        cube.add_aux_coord(lat2, [0])

        src = cube.coord('source').copy()
        src.attributes['reliability'] = 'low'
        cube.add_aux_coord(src)

        self.assertString(str(cube), ('cdm', 'string_representations', 'similar.__str__.txt'))

    def test_cube_summary_cell_methods(self):
        
        cube = self.cube_2d.copy()
        
        # Create a list of values used to create cell methods
        test_values = ((("mean",), (u'longitude', 'latitude'), (u'6 minutes', '12 minutes'), (u'This is a test comment',)),
                        (("average",), (u'longitude', 'latitude'), (u'6 minutes', '15 minutes'), (u'This is another test comment','This is another comment')),
                        (("average",), (u'longitude', 'latitude'), (), ()),
                        (("percentile",), (u'longitude',), (u'6 minutes',), (u'This is another test comment',)))
        
        for x in test_values:
            # Create a cell method
            cm = iris.coords.CellMethod(method=x[0][0], coords=x[1], intervals=x[2], comments=x[3])
            cube.add_cell_method(cm)
        
        self.assertString(str(cube), ('cdm', 'string_representations', 'cell_methods.__str__.txt'))            


@iris.tests.skip_data
class TestValidity(tests.IrisTest):
    def setUp(self):
        self.cube_2d = iris.load_strict(tests.get_data_path(('PP', 'simple_pp', 'global.pp')))

    def test_wrong_length_vector_coord(self):
        wobble = iris.coords.DimCoord(points=[1, 2], long_name='wobble', units='1')
        with self.assertRaises(ValueError):
            self.cube_2d.add_aux_coord(wobble, 0)

    def test_invalid_dimension_vector_coord(self):
        wobble = iris.coords.DimCoord(points=[1, 2], long_name='wobble', units='1')
        with self.assertRaises(ValueError):
            self.cube_2d.add_dim_coord(wobble, 99)

    def test_invalid_coord_system(self):
        cs = self.cube_2d.coord('latitude').coord_system
        old_cs_type = cs.cs_type
        cs.cs_type = -999
        try:
            self.assertRaises(iris.exceptions.InvalidCubeError, cs.assert_valid)
        finally:
            cs.cs_type = old_cs_type


class TestQueryCoord(tests.IrisTest):
    def setUp(self):
        self.t = iris.tests.stock.simple_2d_w_multidim_and_scalars()

    def test_name(self):
        coords = self.t.coords(name='dim1')
        self.assertEqual([coord.name() for coord in coords], ['dim1'])
        
        coords = self.t.coords(name='dim2')
        self.assertEqual([coord.name() for coord in coords], ['dim2'])
        
        coords = self.t.coords(name='an_other')
        self.assertEqual([coord.name() for coord in coords], ['an_other'])

        coords = self.t.coords(name='air_temperature')
        self.assertEqual([coord.name() for coord in coords], ['air_temperature'])

        coords = self.t.coords(name='wibble')
        self.assertEqual(coords, [])

    def test_long_name(self):
        # Both standard_name and long_name defined
        coords = self.t.coords(long_name='custom long name')
        # coord.name() returns standard_name if available
        self.assertEqual([coord.name() for coord in coords], ['air_temperature'])

    def test_standard_name(self):
        # Both standard_name and long_name defined
        coords = self.t.coords(standard_name='custom long name')
        self.assertEqual([coord.name() for coord in coords], [])
        coords = self.t.coords(standard_name='air_temperature')
        self.assertEqual([coord.name() for coord in coords], ['air_temperature'])

    def test_axis(self):
        cube = self.t.copy()
        cube.coord("dim1").rename("latitude") 
        cube.coord("dim2").rename("longitude")  
        
        coords = cube.coords(axis='y') 
        self.assertEqual([coord.name() for coord in coords], ['latitude'])
        
        coords = cube.coords(axis='x') 
        self.assertEqual([coord.name() for coord in coords], ['longitude']) 

        # Renaming shoudn't be enough
        cube.coord("an_other").rename("time")
        coords = cube.coords(axis='t')
        self.assertEqual([coord.name() for coord in coords], [])
        # Change units to "hours since ..." as it's the presence of a
        # time unit that identifies a time axis.
        cube.coord("time").units = 'hours since 1970-01-01 00:00:00'
        coords = cube.coords(axis='t')
        self.assertEqual([coord.name() for coord in coords], ['time'])
        
        coords = cube.coords(axis='z')
        self.assertEqual(coords, [])

    def test_contains_dimension(self):
        coords = self.t.coords(contains_dimension=0)
        self.assertEqual([coord.name() for coord in coords], ['dim1', 'my_multi_dim_coord'])
        
        coords = self.t.coords(contains_dimension=1)
        self.assertEqual([coord.name() for coord in coords], ['dim2', 'my_multi_dim_coord'])
        
        coords = self.t.coords(contains_dimension=2)
        self.assertEqual(coords, [])

    def test_dimensions(self):
        coords = self.t.coords(dimensions=0)
        self.assertEqual([coord.name() for coord in coords], ['dim1'])
        
        coords = self.t.coords(dimensions=1)
        self.assertEqual([coord.name() for coord in coords], ['dim2'])
        
        # find all coordinates which do not describe a dimension
        coords = self.t.coords(dimensions=[])
        self.assertEqual([coord.name() for coord in coords], ['air_temperature', 'an_other'])
        
        coords = self.t.coords(dimensions=2)
        self.assertEqual(coords, [])
        
        coords = self.t.coords(dimensions=[0, 1])
        self.assertEqual([coord.name() for coord in coords], ['my_multi_dim_coord'])
       
    def test_coord_dim_coords_keyword(self):
        coords = self.t.coords(dim_coords=True)
        self.assertEqual(set([coord.name() for coord in coords]), {'dim1', 'dim2'})

        coords = self.t.coords(dim_coords=False)
        self.assertEqual(set([coord.name() for coord in coords]), {'an_other', 'my_multi_dim_coord', 'air_temperature'})

    def test_coords_empty(self):
        coords = self.t.coords()
        self.assertEqual(set([coord.name() for coord in coords]), {'dim1', 'dim2', 'an_other', 'my_multi_dim_coord', 'air_temperature'})
    
    def test_coord(self):
        coords = self.t.coords(coord=self.t.coord('dim1'))
        self.assertEqual([coord.name() for coord in coords], ['dim1'])
        # check for metadata look-up by modifying points
        coord = self.t.coord('dim1').copy()
        coord.points = numpy.arange(5) * 1.23
        coords = self.t.coords(coord=coord)
        self.assertEqual([coord.name() for coord in coords], ['dim1'])
        
    def test_string_representations(self):
        # TODO consolidate with the TestCubeStringRepresentations class 
        self.assertString(str(self.t), ('cdm', 'string_representations', 'multi_dim_coord.__str__.txt'))
        self.assertString(repr(self.t), ('cdm', 'string_representations', 'multi_dim_coord.__repr__.txt'))
    

class TestCube2d(tests.IrisTest):
    def setUp(self):
        self.t = iris.tests.stock.simple_2d_w_multidim_and_scalars()
        self.t.remove_coord('air_temperature')


class Test2dIndexing(TestCube2d):   
    def test_indexing_of_0d_cube(self):
        c = self.t[0, 0]
        self.assertRaises(IndexError, c.__getitem__, (slice(None, None), ) )
        
    def test_cube_indexing_0d(self):
        self.assertCML([self.t[0, 0]], ('cube_slice', '2d_to_0d_cube_slice.cml'))
        
    def test_cube_indexing_1d(self):
        self.assertCML([self.t[0, 0:]], ('cube_slice', '2d_to_1d_cube_slice.cml'))
    
    def test_cube_indexing_1d_multi_slice(self):
        self.assertCML([self.t[0, (0, 1)]], ('cube_slice', '2d_to_1d_cube_multi_slice.cml'))
        self.assertCML([self.t[0, numpy.array([0, 1])]], ('cube_slice', '2d_to_1d_cube_multi_slice.cml'))
    
    def test_cube_indexing_1d_multi_slice2(self):
        self.t.coord('dim1')._TEST_COMPAT_force_explicit = True
        self.t.coord('dim1')._TEST_COMPAT_definitive = True        
        self.assertCML([self.t[(0, 2), (0, 1, 3)]], ('cube_slice', '2d_to_1d_cube_multi_slice2.cml'))
        self.assertCML([self.t[numpy.array([0, 2]), (0, 1, 3)]], ('cube_slice', '2d_to_1d_cube_multi_slice2.cml'))
        self.assertCML([self.t[numpy.array([0, 2]), numpy.array([0, 1, 3])]], ('cube_slice', '2d_to_1d_cube_multi_slice2.cml'))
        
    def test_cube_indexing_1d_multi_slice3(self):
        self.t.coord('dim1')._TEST_COMPAT_force_explicit = True
        self.t.coord('dim1')._TEST_COMPAT_definitive = True        
        self.assertCML([self.t[(0, 2), :]], ('cube_slice', '2d_to_1d_cube_multi_slice3.cml'))
        self.assertCML([self.t[numpy.array([0, 2]), :]], ('cube_slice', '2d_to_1d_cube_multi_slice3.cml'))

    def test_cube_indexing_no_change(self):
        self.assertCML([self.t[0:, 0:]], ('cube_slice', '2d_orig.cml'))
    
    def test_cube_indexing_reverse_coords(self):
        self.assertCML([self.t[::-1, ::-1]], ('cube_slice', '2d_to_2d_revesed.cml'))
        
    def test_cube_indexing_no_residual_change(self):
        self.t[0:3]
        self.assertCML([self.t], ('cube_slice', '2d_orig.cml'))
        
    def test_overspecified(self):
        self.assertRaises(IndexError, self.t.__getitem__, (0, 0, Ellipsis, 0))
        self.assertRaises(IndexError, self.t.__getitem__, (0, 0, 0))            
    
    def test_ellipsis(self):
        self.assertCML([self.t[Ellipsis]], ('cube_slice', '2d_orig.cml'))
        self.assertCML([self.t[:, :, :]], ('cube_slice', '2d_orig.cml'))
        self.assertCML([self.t[Ellipsis, Ellipsis]], ('cube_slice', '2d_orig.cml'))
        self.assertCML([self.t[Ellipsis, Ellipsis, Ellipsis]], ('cube_slice', '2d_orig.cml'))
       
        self.assertCML([self.t[Ellipsis, 0, 0]], ('cube_slice', '2d_to_0d_cube_slice.cml'))
        self.assertCML([self.t[0, Ellipsis, 0]], ('cube_slice', '2d_to_0d_cube_slice.cml'))
        self.assertCML([self.t[0, 0, Ellipsis]], ('cube_slice', '2d_to_0d_cube_slice.cml'))
        
        self.t.coord('dim1')._TEST_COMPAT_force_explicit = True
        self.t.coord('dim1')._TEST_COMPAT_definitive = True        
        self.assertCML([self.t[Ellipsis, (0, 2), :]], ('cube_slice', '2d_to_1d_cube_multi_slice3.cml'))
        self.assertCML([self.t[(0, 2), Ellipsis, :]], ('cube_slice', '2d_to_1d_cube_multi_slice3.cml'))
        self.assertCML([self.t[(0, 2), :, Ellipsis]], ('cube_slice', '2d_to_1d_cube_multi_slice3.cml'))
        

class Test2dSlicing(TestCube2d):
    def test_cube_slice_all_dimensions(self):
        for cube in self.t.slices(['dim1', 'dim2']):
            self.assertCML(cube, ('cube_slice', '2d_orig.cml'))
            
    def test_cube_slice_with_transpose(self):
        for cube in self.t.slices(['dim2', 'dim1']):
            self.assertCML(cube, ('cube_slice', '2d_transposed.cml'))
            
    def test_cube_slice_1dimension(self):
        slices = [res for res in self.t.slices(['dim2'])]
        # Result came from the equivalent test test_cube_indexing_1d which does self.t[0, 0:]
        self.assertCML(slices[0], ('cube_slice', '2d_to_1d_cube_slice.cml'))
    
    def test_cube_slice_zero_len_slice(self):
        self.assertRaises(IndexError, self.t.__getitem__, (slice(0, 0)))
    
    def test_cube_slice_with_non_existant_coords(self):
        self.assertRaises(iris.exceptions.CoordinateNotFoundError, self.t.slices, ['dim2', 'dim1', 'doesnt exist'])
        
    def test_cube_extract_coord_with_non_describing_coordinates(self):
        self.assertRaises(ValueError, self.t.slices, ['an_other'])


class Test2dExtraction(TestCube2d):
    def test_cube_extract_0d(self):
        # Extract the first value from each of the coords in the cube
        # this result is shared with the self.t[0, 0] test
        self.assertCML([self.t.extract(iris.Constraint(dim1=3.0, dim2=iris.coords.Cell(0, (0, 1))))], ('cube_slice', '2d_to_0d_cube_slice.cml'))
    
    def test_cube_extract_1d(self):
        # Extract the first value from the second coord in the cube
        # this result is shared with the self.t[0, 0:] test
        self.assertCML([self.t.extract(iris.Constraint(dim1=3.0))], ('cube_slice', '2d_to_1d_cube_slice.cml'))
        
    def test_cube_extract_2d(self):
        # Do nothing - return the original
        self.assertCML([self.t.extract(iris.Constraint())], ('cube_slice', '2d_orig.cml'))

    def test_cube_extract_coord_which_does_not_exist(self):
        self.assertEqual(self.t.extract(iris.Constraint(doesnt_exist=8.1)), None)
            
    def test_cube_extract_coord_with_non_existant_values(self):
        self.assertEqual(self.t.extract(iris.Constraint(dim1=8)), None)
            
    
class Test2dExtractionByCoord(TestCube2d):
    def test_cube_extract_by_coord_advanced(self):
        # This test reverses the coordinate in the cube and also takes a subset of the original coordinate
        points = numpy.array([9, 8, 7, 5, 4, 3, 2, 1, 0], dtype=numpy.int32)
        bounds = numpy.array([[18, 19], [16, 17], [14, 15], [10, 11], [ 8,  9], [ 6,  7], [ 4,  5], [ 2,  3], [ 0,  1]], dtype=numpy.int32)
        c = iris.coords.DimCoord(points, long_name='dim2', units='meters', bounds=bounds)
        self.assertCML(self.t.subset(c), ('cube_slice', '2d_intersect_and_reverse.cml'))
        

@iris.tests.skip_data
class TestCubeExtract(tests.IrisTest):
    def setUp(self):
        self.single_cube = iris.load_strict(tests.get_data_path(('PP', 'globClim1', 'theta.pp')), 'air_potential_temperature')
        self.single_cube.coord('forecast_period')._TEST_COMPAT_override_axis = 'forecast_period'
        self.single_cube.coord('forecast_reference_time')._TEST_COMPAT_override_axis = 'rt'
        self.single_cube.coord('source')._TEST_COMPAT_override_axis = 'source'        
        self.single_cube.coord('source')._TEST_COMPAT_definitive = False       
        self.single_cube.coord('model_level_number')._TEST_COMPAT_force_explicit = True
        self.single_cube.coord('model_level_number')._TEST_COMPAT_override_axis = 'z'
        self.single_cube.coord('level_height')._TEST_COMPAT_override_axis = 'z'
        self.single_cube.coord('sigma')._TEST_COMPAT_override_axis = 'z'
        self.single_cube.coord('time')._TEST_COMPAT_points = False
        

    def test_simple(self):
        constraint = iris.Constraint(latitude=10)
        cube = self.single_cube.extract(constraint)
        cube.coord('latitude')._TEST_COMPAT_force_regular_scalar = True
        self.assertCML(cube, ('cdm', 'extract', 'lat_eq_10.cml'))
        constraint = iris.Constraint(latitude=lambda c: c > 10)
        self.assertCML(self.single_cube.extract(constraint), ('cdm', 'extract', 'lat_gt_10.cml'))
        
    def test_combined(self):
        constraint = iris.Constraint(latitude=lambda c: c > 10, longitude=lambda c: c >= 10)

        self.assertCML(self.single_cube.extract(constraint), ('cdm', 'extract', 'lat_gt_10_and_lon_ge_10.cml'))
    
    def test_no_results(self):
        constraint = iris.Constraint(latitude=lambda c: c > 1000000)
        self.assertEqual(self.single_cube.extract(constraint), None)
        
        
class TestCubeAPI(TestCube2d):    
    def test_getting_standard_name(self):
        self.assertEqual(self.t.name(), 'test 2d dimensional cube')

    def test_rename(self):
        self.t.rename('foo')
        self.assertEqual(self.t.name(), 'foo')
        
    def test_getting_unit(self):
        self.assertEqual(str(self.t.units), 'meters')

    def test_setting_unit(self):
        self.t.units = iris.unit.Unit('volt')
        self.assertEqual(str(self.t.units), 'volt')

    def test_coords_are_copies(self):
        self.assertIsNot(self.t.coord('dim1'), self.t.copy().coord('dim1'))

    def test_metadata_nop(self):
        self.t.metadata = self.t.metadata
        self.assertIsNone(self.t.standard_name)
        self.assertEqual(self.t.long_name, 'test 2d dimensional cube')
        self.assertEqual(self.t.units, 'meters')
        self.assertEqual(self.t.attributes, {})
        self.assertEqual(self.t.cell_methods, ())

    def test_metadata_tuple(self):
        metadata = ('air_pressure', 'foo', '', {'random': '12'}, ())
        self.t.metadata = metadata
        self.assertEqual(self.t.standard_name, 'air_pressure')
        self.assertEqual(self.t.long_name, 'foo')
        self.assertEqual(self.t.units, '')
        self.assertEqual(self.t.attributes, metadata[3])
        self.assertIsNot(self.t.attributes, metadata[3])
        self.assertEqual(self.t.cell_methods, ())

    def test_metadata_dict(self):
        metadata = {'standard_name': 'air_pressure',
                    'long_name': 'foo', 'units': '',
                    'attributes': {'random': '12'},
                    'cell_methods': ()}
        self.t.metadata = metadata
        self.assertEqual(self.t.standard_name, 'air_pressure')
        self.assertEqual(self.t.long_name, 'foo')
        self.assertEqual(self.t.units, '')
        self.assertEqual(self.t.attributes, metadata['attributes'])
        self.assertIsNot(self.t.attributes, metadata['attributes'])
        self.assertEqual(self.t.cell_methods, ())

    def test_metadata_attrs(self):
        class Metadata(object): pass
        metadata = Metadata()
        metadata.standard_name = 'air_pressure'
        metadata.long_name = 'foo'
        metadata.units = ''
        metadata.attributes = {'random': '12'}
        metadata.cell_methods = ()
        self.t.metadata = metadata
        self.assertEqual(self.t.standard_name, 'air_pressure')
        self.assertEqual(self.t.long_name, 'foo')
        self.assertEqual(self.t.units, '')
        self.assertEqual(self.t.attributes, metadata.attributes)
        self.assertIsNot(self.t.attributes, metadata.attributes)
        self.assertEqual(self.t.cell_methods, ())

    def test_metadata_fail(self):
        with self.assertRaises(TypeError):
            self.t.metadata = ('air_pressure', 'foo', '', {'random': '12'})
        with self.assertRaises(TypeError):
            self.t.metadata = ('air_pressure', 'foo', '', {'random': '12'}, (), ())
        with self.assertRaises(TypeError):
            self.t.metadata = {'standard_name': 'air_pressure',
                               'long_name': 'foo', 'units': '',
                               'attributes': {'random': '12'}}
        with self.assertRaises(TypeError):
            class Metadata(object): pass
            metadata = Metadata()
            metadata.standard_name = 'air_pressure'
            metadata.long_name = 'foo'
            metadata.units = ''
            metadata.attributes = {'random': '12'}
            self.t.metadata = metadata


class TestCubeEquality(TestCube2d):            
    def test_simple_equality(self):
        self.assertEqual(self.t, self.t.copy())
    
    def test_data_inequality(self):
        self.assertNotEqual(self.t, self.t + 1)
    
    def test_coords_inequality(self):
        r = self.t.copy()
        r.remove_coord(r.coord('an_other'))
        self.assertNotEqual(self.t, r)
    
    def test_attributes_inequality(self):
        r = self.t.copy()
        r.attributes['new_thing'] = None
        self.assertNotEqual(self.t, r)
        
    def test_cell_methods_inequality(self):
        r = self.t.copy()
        r.add_cell_method(iris.coords.CellMethod('mean'))
        self.assertNotEqual(self.t, r)


@iris.tests.skip_data
class TestDataManagerIndexing(TestCube2d):
    def setUp(self):
        self.cube = iris.load_strict(tests.get_data_path(('PP', 'aPProt1', 'rotatedMHtimecube.pp')))
        self.pa = self.cube._data
        self.dm = self.cube._data_manager
        self.data_array = self.dm.load(self.pa)

    def test_slices(self):
        lat_cube = self.cube.slices(['grid_latitude', ]).next()
        self.assertIsNotNone(lat_cube._data_manager)
        self.assertIsNotNone(self.cube._data_manager)
 
    def check_indexing(self, keys):
        pa, dm = self.dm.getitem(self.pa, keys)
        r = dm.load(pa)
        numpy.testing.assert_array_equal(r, self.data_array[keys], 
                                         'Arrays were not the same after indexing '
                                         '(original shape %s) using:\n %r' % (self.data_array.shape, keys)
                                         )
        
    def _check_consecutive(self, keys1, keys2):
        pa, dm = self.dm.getitem(self.pa, keys1)
        pa, dm = dm.getitem(pa, keys2)
        # Test the access of the data shape...
        r = dm.shape(pa)
        numpy.testing.assert_array_equal(r, self.data_array[keys1][keys2].shape, 'Reported shapes were not the same after consecutive indexing'
                                         '(original shape %s) using:\n 1:       %r\n 2:       %r' % (self.data_array.shape, keys1, keys2),
                                         )
        
        r = dm.load(pa)
        numpy.testing.assert_array_equal(r, self.data_array[keys1][keys2], 
                                         'Arrays were not the same after consecutive indexing '
                                         '(original shape %s) using:\n 1:       %r\n 2:       %r' % (self.data_array.shape, keys1, keys2),
                                         )
        
    def check_consecutive(self, keys1, keys2):
        self._check_consecutive(keys1, keys2)
        self._check_consecutive(keys2, keys1)
            
    
    def check_indexing_error(self, keys):
        self.assertRaises(IndexError, self.dm.getitem, self.pa, keys)
        
    def test_single_index(self):
        self.check_indexing(2)
        self.check_indexing(-1)
        self.check_indexing(0)
        self.check_indexing(None)
        
    def test_basic(self):
        self.check_indexing( (2, ) )
        self.check_indexing( (slice(None, None), 2) )
        self.check_indexing( (slice(None, None, 2), 2) )
        self.check_indexing( (slice(None, -4, -2), 2) )
        self.check_indexing( (3, slice(None, -4, -2), 2) )
        self.check_indexing( (3, 3, 2) )    
        self.check_indexing( (Ellipsis, 2, 3) )
        self.check_indexing( (slice(3, 4), Ellipsis, 2, 3) )
        self.check_indexing( (numpy.array([3], ndmin=1), Ellipsis, 2, 3) )
        self.check_indexing( (slice(3, 4), Ellipsis, Ellipsis, 3) )
        self.check_indexing( (slice(3, 4), Ellipsis, Ellipsis, Ellipsis) )
        self.check_indexing( (Ellipsis, Ellipsis, Ellipsis, Ellipsis) )
        
    def test_out_of_range(self):
        self.check_indexing_error( tuple([slice(None, None)] * 5) )
        self.check_indexing_error( tuple([slice(None, None)] * 6) )
        self.check_indexing_error( 10000 )
        self.check_indexing_error( (10000, 2) )
        self.check_indexing_error( (10000, ) )
        self.check_indexing_error( (10, 10000) )
                
    def test_consecutive(self):
        self.check_consecutive(3, 2)
        self.check_consecutive(3, slice(None, None))
        self.check_consecutive(1, slice(None, -6, -2))
        self.check_consecutive(3, (slice(None, None), 3))
        self.check_consecutive(1, ((3, 2, 1, 3), 3))
        self.check_consecutive(1, (numpy.array([3, 2, 1, 3]), 3))
        self.check_consecutive(1, (3, numpy.array([3, 2, 1, 3])))
        self.check_consecutive((4, slice(6, 7)), 0)
        self.check_consecutive((Ellipsis, slice(6, 7), 5), 0)
        self.check_consecutive((Ellipsis, slice(7, 5, -1), 5), 0)
        self.check_consecutive((Ellipsis, (3, 2, 1, 3), slice(6, 7)), 0)
        
    def test_cube_empty_indexing(self):
        test_filename = ('cube_slice', 'real_empty_data_indexing.cml')
        r = self.cube[:5, ::-1][3]
        rshape = r.shape
        
        # Make sure the datamanager is still being uses (i.e. is not None)
        self.assertNotEqual( r._data_manager, None )
        # check the CML of this result
        self.assertCML(r, test_filename)
        # The CML was checked, meaning the data must have been loaded. Check that the cube no longer has a datamanager
        self.assertEqual( r._data_manager, None )
        
        r_data = r.data
        
        #finally, load the data before indexing and check that it generates the same result
        c = self.cube
        c.data
        c = c[:5, ::-1][3]
        self.assertCML(c, test_filename)
        
        self.assertEqual(rshape, c.shape)
        
        numpy.testing.assert_array_equal(r_data, c.data)
        
    def test_real_data_cube_indexing(self):
        self.cube.coord('source')._TEST_COMPAT_force_explicit = True
        self.cube.coord('source')._TEST_COMPAT_override_axis = 'source'
        self.cube.coord('source')._TEST_COMPAT_definitive = False
        
        cube = self.cube[(0, 4, 5, 2), 0, 0]
        cube.coord('forecast_period')._TEST_COMPAT_override_axis = 'forecast_period'
        cube.coord('grid_longitude')._TEST_COMPAT_force_regular_scalar = True
        cube.coord('grid_latitude')._TEST_COMPAT_force_regular_scalar = True
        self.assertCML(cube, ('cube_slice', 'real_data_dual_tuple_indexing1.cml'))

        cube = self.cube[0, (0, 4, 5, 2), (3, 5, 5)]
        cube.coord('forecast_period')._TEST_COMPAT_override_axis = 'forecast_period'
        cube.coord('grid_longitude')._TEST_COMPAT_definitive = True
        self.assertCML(cube, ('cube_slice', 'real_data_dual_tuple_indexing2.cml'))
        
        cube = self.cube[(0, 4, 5, 2), 0, (3, 5, 5)]
        cube.coord('forecast_period')._TEST_COMPAT_override_axis = 'forecast_period'
        cube.coord('grid_longitude')._TEST_COMPAT_definitive = True
        cube.coord('grid_latitude')._TEST_COMPAT_force_regular_scalar = True
        self.assertCML(cube, ('cube_slice', 'real_data_dual_tuple_indexing3.cml'))

        self.assertRaises(IndexError, self.cube.__getitem__, ((0, 4, 5, 2), (3, 5, 5), 0, 0, 4) )
        self.assertRaises(IndexError, self.cube.__getitem__, (Ellipsis, Ellipsis, Ellipsis, Ellipsis, Ellipsis, Ellipsis) )


class TestCubeCollapsed(tests.IrisTest):
    def partial_compare(self, dual, single):
        result = iris.analysis.coord_comparison(dual, single)
        self.assertEqual(len(result['not_equal']), 0)
        self.assertNotEqual(dual.attributes['history'], single.attributes['history'], 'dual and single stage history are equal')
        self.assertEqual(dual.name(), single.name(), "dual and single stage standard_names differ")
        self.assertEqual(dual.units, single.units, "dual and single stage units differ")
        self.assertEqual(dual.shape, single.shape, "dual and single stage shape differ")

    def collapse_test_common(self, cube, a_name, b_name, *args, **kwargs):
        
        # preserve filenames from before the introduction of "grid_" in rotated coord names.
        a_filename = a_name.replace("grid_", "")
        b_filename = b_name.replace("grid_", "")
        
        # compare dual and single stage collapsing
        dual_stage = cube.collapsed(a_name, iris.analysis.MEAN)
        dual_stage = dual_stage.collapsed(b_name, iris.analysis.MEAN)
        self.assertCMLApproxData(dual_stage, ('cube_collapsed', '%s_%s_dual_stage.cml' % (a_filename, b_filename)), *args, **kwargs)

        single_stage = cube.collapsed([a_name, b_name], iris.analysis.MEAN)
        self.assertCMLApproxData(single_stage, ('cube_collapsed', '%s_%s_single_stage.cml' % (a_filename, b_filename)), *args, **kwargs)

        # Compare the cube bits that should match
        self.partial_compare(dual_stage, single_stage)

    @iris.tests.skip_data
    def test_multi_d(self):
        cube = iris.load(tests.get_data_path(('PP', 'COLPEX', 'theta_and_orog_subset.pp')))[0]
        cube.coord('forecast_period')._TEST_COMPAT_override_axis = 'forecast_period'
        cube.coord('time')._TEST_COMPAT_force_explicit = True
        cube.coord('grid_latitude')._TEST_COMPAT_force_explicit = True
        cube.coord('source')._TEST_COMPAT_override_axis = 'source'        
        cube.coord('source')._TEST_COMPAT_definitive = False       
        cube.coord('model_level_number')._TEST_COMPAT_force_explicit = True
        cube.coord('model_level_number')._TEST_COMPAT_override_axis = 'z'
        cube.coord('level_height')._TEST_COMPAT_override_axis = 'z'
        cube.coord('sigma')._TEST_COMPAT_override_axis = 'z'

        # TODO: Re-instate surface_altitude & hybrid-height once we're
        # using the post-CF test results.
        cube.remove_aux_factory(cube.aux_factories[0])
        cube.remove_coord('surface_altitude')

        self.assertCML(cube, ('cube_collapsed', 'original.cml'))

        # Compare 2-stage collapsing with a single stage collapse over 2 Coords (ignoring history).
        self.collapse_test_common(cube, 'grid_latitude', 'grid_longitude', decimal=1)
        self.collapse_test_common(cube, 'grid_longitude', 'grid_latitude', decimal=1)

        self.collapse_test_common(cube, 'time', 'grid_latitude', decimal=1)
        self.collapse_test_common(cube, 'grid_latitude', 'time', decimal=1)

        self.collapse_test_common(cube, 'time', 'grid_longitude', decimal=1)
        self.collapse_test_common(cube, 'grid_longitude', 'time', decimal=1)

        self.collapse_test_common(cube, 'grid_latitude', 'model_level_number', decimal=1)
        self.collapse_test_common(cube, 'model_level_number', 'grid_latitude', decimal=1)

        self.collapse_test_common(cube, 'grid_longitude', 'model_level_number', decimal=1)
        self.collapse_test_common(cube, 'model_level_number', 'grid_longitude', decimal=1)

        self.collapse_test_common(cube, 'time', 'model_level_number', decimal=1)
        self.collapse_test_common(cube, 'model_level_number', 'time', decimal=1)

        self.collapse_test_common(cube, 'model_level_number', 'time', decimal=1)
        self.collapse_test_common(cube, 'time', 'model_level_number', decimal=1)

        # Collapse 3 things at once.
        triple_collapse = cube.collapsed(['model_level_number', 'time', 'grid_longitude'], iris.analysis.MEAN)
        self.assertCMLApproxData(triple_collapse, ('cube_collapsed', 'triple_collapse_ml_pt_lon.cml'), decimal=1)

        triple_collapse = cube.collapsed(['grid_latitude', 'model_level_number', 'time'], iris.analysis.MEAN)
        self.assertCMLApproxData(triple_collapse, ('cube_collapsed', 'triple_collapse_lat_ml_pt.cml'), decimal=1)

        # Ensure no side effects
        self.assertCML(cube, ('cube_collapsed', 'original.cml'))
        
        
class TestTrimAttributes(tests.IrisTest):
    def test_non_string_attributes(self):
        cube = iris.tests.stock.realistic_4d()
        attrib_key = "gorf"
        attrib_val = 23
        cube.attributes[attrib_key] = attrib_val
        
        summary = cube.summary() # Get the cube summary
        
        # Check through the lines of the summary to see that our attribute is there
        attrib_re = re.compile("%s.*?%s" % (attrib_key, attrib_val))

        for line in summary.split("\n"):
            result = re.match(attrib_re, line.strip())
            if result:
                break
        else: # No match found for our attribute
            self.fail('Attribute not found in summary output of cube.')


@iris.tests.skip_data
class TestMaskedData(tests.IrisTest, pp.PPTest):
    def _load_3d_cube(self):
        # This 3D data set has a missing a slice with SOME missing values (0)
        return iris.load_strict(tests.get_data_path(["PP", "mdi_handmade_small", "*.pp"]))
    
    def test_complete_field(self):
        # This pp field has no missing data values
        cube = iris.load_strict(tests.get_data_path(["PP", "mdi_handmade_small", "mdi_test_1000_3.pp"]))

        self.assertTrue(isinstance(cube.data, numpy.ndarray), "Expected a numpy.ndarray")

    def test_masked_field(self):
        # This pp field has some missing data values
        cube = iris.load_strict(tests.get_data_path(["PP", "mdi_handmade_small", "mdi_test_1000_0.pp"]))
        self.assertTrue(isinstance(cube.data, numpy.ma.core.MaskedArray), "Expected a numpy.ma.core.MaskedArray")

    def test_missing_file(self):
        cube = self._load_3d_cube()
        self.assertTrue(isinstance(cube.data, numpy.ma.core.MaskedArray), "Expected a numpy.ma.core.MaskedArray")
        cube.coord('forecast_period')._TEST_COMPAT_override_axis = 'forecast_period'
        cube.coord('source')._TEST_COMPAT_override_axis = 'source'
        cube.coord('forecast_period')._TEST_COMPAT_definitive = True
        cube.coord('pressure')._TEST_COMPAT_definitive = True
        cube.coord('time')._TEST_COMPAT_definitive = True
        cube.coord('source')._TEST_COMPAT_definitive = False
        self.assertCML(cube, ('cdm', 'masked_cube.cml'))
        
    def test_slicing(self):
        cube = self._load_3d_cube()

        # Test the slicing before deferred loading
        full_slice = cube[3]
        partial_slice = cube[0]
        self.assertTrue(isinstance(full_slice.data, numpy.ndarray), "Expected a numpy array")
        self.assertTrue(isinstance(partial_slice.data, numpy.ma.core.MaskedArray), "Expected a numpy.ma.core.MaskedArray")
        self.assertEqual(numpy.ma.count_masked(partial_slice._data), 25)

        # Test the slicing is consistent after deferred loading
        cube.data
        full_slice = cube[3]
        partial_slice = cube[0]
        self.assertTrue(isinstance(full_slice.data, numpy.ndarray), "Expected a numpy array")
        self.assertTrue(isinstance(partial_slice.data, numpy.ma.core.MaskedArray), "Expected a numpy.ma.core.MaskedArray")
        self.assertEqual(numpy.ma.count_masked(partial_slice._data), 25)

    def test_save_and_merge(self):
        cube = self._load_3d_cube()

        # extract the 2d field that has SOME missing values
        masked_slice = cube[0]
        masked_slice.data.fill_value = 123456
        
        # test saving masked data
        reference_txt_path = tests.get_result_path(('cdm', 'masked_save_pp.txt'))
        with self.cube_save_test(reference_txt_path, reference_cubes=masked_slice) as temp_pp_path:
            iris.save(masked_slice, temp_pp_path)
        
            # test merge keeps the mdi we just saved
            cube1 = iris.load_strict(temp_pp_path)
            cube2 = cube1.copy()
            # make cube1 and cube2 differ on a scalar coord, to make them mergeable into a 3d cube
            cube2.coord("pressure").points[0] = 1001.0
            merged_cubes = iris.cube.CubeList([cube1, cube2]).merge()
            self.assertEqual(len(merged_cubes), 1, "expected a single merged cube")
            merged_cube = merged_cubes[0]
            self.assertEqual(merged_cube.data.fill_value, 123456)


class TestConversionToCoordList(tests.IrisTest):
    def test_coord_conversion(self):
        cube = iris.tests.stock.realistic_4d()
        
        # Single string
        self.assertEquals(len(cube._as_list_of_coords('grid_longitude')), 1)
        
        # List of string and unicode
        self.assertEquals(len(cube._as_list_of_coords(['grid_longitude', u'grid_latitude'], )), 2)
        
        # Coord object(s)
        lat = cube.coords("grid_latitude")[0]
        lon = cube.coords("grid_longitude")[0]
        self.assertEquals(len(cube._as_list_of_coords(lat)), 1)
        self.assertEquals(len(cube._as_list_of_coords([lat, lon])), 2)
        
        # Mix of string-like and coord
        self.assertEquals(len(cube._as_list_of_coords(["grid_latitude", lon])), 2)

        # Empty list
        self.assertEquals(len(cube._as_list_of_coords([])), 0)
        
        # Invalid coords
        invalid_choices = [iris.analysis.MEAN, # Caused by mixing up argument order in call to cube.collasped for example
                           None,
                           ['grid_latitude', None],
                           [lat, None],
                          ]

        for coords in invalid_choices:
            with self.assertRaises(TypeError):
                cube._as_list_of_coords(coords)


if __name__ == "__main__":
    tests.main()
