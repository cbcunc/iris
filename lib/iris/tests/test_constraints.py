# (C) British Crown Copyright 2010 - 2013, Met Office
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
Test the constrained cube loading mechanism.

"""
# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import cPickle
import cStringIO

import iris
import iris.tests.stock as stock


SN_AIR_POTENTIAL_TEMPERATURE = 'air_potential_temperature'
SN_SPECIFIC_HUMIDITY = 'specific_humidity'


# TODO: Workaround, pending #1262
def workaround_pending_1262(cubes):
    """Reverse the cube if sigma was chosen as a dim_coord."""
    for i, cube in enumerate(cubes):
        ml = cube.coord("model_level_number").points
        if ml[0] > ml[1]:
            cubes[i] = cube[::-1]


class TestSimple(tests.IrisTest):
    slices = iris.cube.CubeList(stock.realistic_4d().slices(['grid_latitude', 'grid_longitude']))

    def test_constraints(self):
        constraint = iris.Constraint(model_level_number=10)
        sub_list = self.slices.extract(constraint)
        self.assertEqual(len(sub_list), 6)

        constraint = iris.Constraint(model_level_number=[10, 22])                  
        sub_list = self.slices.extract(constraint)
        self.assertEqual(len(sub_list), 2 * 6)

        constraint = iris.Constraint(model_level_number=lambda c: ( c > 30 ) | (c <= 3))
        sub_list = self.slices.extract(constraint)
        self.assertEqual(len(sub_list), 43 * 6)

        constraint = iris.Constraint(coord_values={'model_level_number': lambda c: c > 1000})
        sub_list = self.slices.extract(constraint)
        self.assertEqual(len(sub_list), 0)

        constraint = (iris.Constraint(model_level_number=10) &
                      iris.Constraint(time=347922.))
        sub_list = self.slices.extract(constraint)
        self.assertEqual(len(sub_list), 1)

        constraint = iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE)
        sub_list = self.slices.extract(constraint)
        self.assertEqual(len(sub_list), 70 * 6)

    def test_mismatched_type(self):
        constraint = iris.Constraint(model_level_number='aardvark')
        sub_list = self.slices.extract(constraint)
        self.assertEqual(len(sub_list), 0)

    def test_cell(self):
        cell = iris.coords.Cell(10)
        constraint = iris.Constraint(model_level_number=cell)
        sub_list = self.slices.extract(constraint)
        self.assertEqual(len(sub_list), 6)


class TestMixin(object):
    """
    Mix-in class for attributes & utilities common to the "normal" and "strict" test cases.
    
    """
    def setUp(self):
        self.dec_path = tests.get_data_path(['PP', 'globClim1', 'dec_subset.pp'])
        self.theta_path = tests.get_data_path(['PP', 'globClim1', 'theta.pp'])

        self.humidity = iris.Constraint(SN_SPECIFIC_HUMIDITY)
        self.theta = iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE)

        # Coord based constraints
        self.level_10 = iris.Constraint(model_level_number=10)
        self.level_22 = iris.Constraint(model_level_number=22)

        # Value based coord constraint
        self.level_30 = iris.Constraint(model_level_number=30)
        self.level_gt_30_le_3 = iris.Constraint(model_level_number=lambda c: ( c > 30 ) | (c <= 3))
        self.invalid_inequality = iris.Constraint(coord_values={'model_level_number': lambda c: c > 1000})
        
        # bound based coord constraint
        self.level_height_of_model_level_number_10 = iris.Constraint(level_height=1900)
        self.model_level_number_10_22 = iris.Constraint(model_level_number=[10, 22])                  

        # Invalid constraints
        self.pressure_950 = iris.Constraint(model_level_number=950)
        
        self.lat_30 = iris.Constraint(latitude=30)
        self.lat_gt_45 = iris.Constraint(latitude=lambda c: c > 45)


class RelaxedConstraintMixin(TestMixin):
    @staticmethod
    def fixup_sigma_to_be_aux(cubes):
        # XXX Fix the cubes such that the sigma coordinate is always an AuxCoord. Pending gh issue #18
        if isinstance(cubes, iris.cube.Cube):
            cubes = [cubes]
            
        for cube in cubes:
            sigma = cube.coord('sigma')
            sigma = iris.coords.AuxCoord.from_coord(sigma)
            cube.replace_coord(sigma)
    
    def assertCML(self, cubes, filename):
        filename = "%s_%s.cml" % (filename, self.suffix)
        tests.IrisTest.assertCML(self, cubes, ('constrained_load', filename))
        
    def load_match(self, files, constraints):
        raise NotImplementedError()  # defined in subclasses
        
    def test_single_atomic_constraint(self):
        cubes = self.load_match(self.dec_path, self.level_10)
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'all_10')

        cubes = self.load_match(self.dec_path, self.theta)
        self.assertCML(cubes, 'theta')
        
        cubes = self.load_match(self.dec_path, self.model_level_number_10_22)
        self.fixup_sigma_to_be_aux(cubes) 
        workaround_pending_1262(cubes)
        self.assertCML(cubes, 'all_ml_10_22')
        
        # Check that it didn't matter that we provided sets & tuples to the model_level
        for constraint in [iris.Constraint(model_level_number=set([10, 22])), iris.Constraint(model_level_number=tuple([10, 22]))]:
            cubes = self.load_match(self.dec_path, constraint)
            self.fixup_sigma_to_be_aux(cubes)
            workaround_pending_1262(cubes)
            self.assertCML(cubes, 'all_ml_10_22')          
    
    def test_string_standard_name(self):
        cubes = self.load_match(self.dec_path, SN_AIR_POTENTIAL_TEMPERATURE)
        self.assertCML(cubes, 'theta')

        cubes = self.load_match(self.dec_path, [SN_AIR_POTENTIAL_TEMPERATURE])
        self.assertCML(cubes, 'theta')

        cubes = self.load_match(self.dec_path, iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE))
        self.assertCML(cubes, 'theta')
    
        cubes = self.load_match(self.dec_path, iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE, model_level_number=10))
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'theta_10')
    
    def test_latitude_constraint(self):
        cubes = self.load_match(self.theta_path, self.lat_30)
        self.assertCML(cubes, 'theta_lat_30')

        cubes = self.load_match(self.theta_path, self.lat_gt_45)
        self.assertCML(cubes, 'theta_lat_gt_30')
        
    def test_single_expression_constraint(self):
        cubes = self.load_match(self.theta_path, self.theta & self.level_10)
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'theta_10')
        
        cubes = self.load_match(self.theta_path, self.level_10 & self.theta)
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'theta_10')

    def test_dual_atomic_constraint(self):
        cubes = self.load_match(self.dec_path, [self.theta, self.level_10])
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'theta_and_all_10')
    
    def test_dual_repeated_constraint(self):
        cubes = self.load_match(self.dec_path, [self.theta, self.theta])
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'theta_and_theta')

    def test_dual_expression_constraint(self):
        cubes = self.load_match(self.dec_path, [self.theta & self.level_10, self.level_gt_30_le_3 & self.theta])
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'theta_10_and_theta_level_gt_30_le_3')

    def test_invalid_constraint(self):
        cubes = self.load_match(self.theta_path, self.pressure_950)
        self.assertCML(cubes, 'pressure_950')

        cubes = self.load_match(self.theta_path, self.invalid_inequality)
        self.assertCML(cubes, 'invalid_inequality')
        
    def test_inequality_constraint(self):
        cubes = self.load_match(self.theta_path, self.level_gt_30_le_3)
        self.assertCML(cubes, 'theta_gt_30_le_3')


class StrictConstraintMixin(RelaxedConstraintMixin):
    def test_single_atomic_constraint(self):
        cubes = self.load_match(self.theta_path, self.theta)
        self.assertCML(cubes, 'theta')
        
        cubes = self.load_match(self.theta_path, self.level_10)
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'theta_10')
    
    def test_invalid_constraint(self):
        with self.assertRaises(iris.exceptions.ConstraintMismatchError):
            self.load_match(self.theta_path, self.pressure_950)
    
    def test_dual_atomic_constraint(self):
        cubes = self.load_match(self.dec_path, [self.theta, self.level_10 & self.theta])
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'theta_and_theta_10')


@iris.tests.skip_data
class TestCubeLoadConstraint(RelaxedConstraintMixin, tests.IrisTest):
    suffix = 'load_match'
    
    def load_match(self, files, constraints):
        cubes = iris.load(files, constraints)
        if not isinstance(cubes, iris.cube.CubeList):
            raise Exception("NOT A CUBE LIST! " + str(type(cubes)))
        return cubes 
    

@iris.tests.skip_data
class TestCubeListConstraint(RelaxedConstraintMixin, tests.IrisTest):
    suffix = 'load_match'
    
    def load_match(self, files, constraints):
        cubes = iris.load(files).extract(constraints)
        if not isinstance(cubes, iris.cube.CubeList):
            raise Exception("NOT A CUBE LIST! " + str(type(cubes)))
        return cubes 
    

@iris.tests.skip_data    
class TestCubeLoadStrictConstraint(StrictConstraintMixin, tests.IrisTest):
    suffix = 'load_strict'
    
    def load_match(self, files, constraints):
        cubes = iris.load_strict(files, constraints)    
        return cubes 


@iris.tests.skip_data
class TestCubeListStrictConstraint(StrictConstraintMixin, tests.IrisTest):
    suffix = 'load_strict'
    
    def load_match(self, files, constraints):
        cubes = iris.load(files).extract_strict(constraints)
        return cubes 


@iris.tests.skip_data
class TestCubeExtract(TestMixin, tests.IrisTest):
    def setUp(self):
        TestMixin.setUp(self)
        self.cube = iris.load_cube(self.theta_path)

    def test_attribute_constraint(self):
        # there is no my_attribute attribute on the cube, so ensure it returns None
        cube = self.cube.extract(iris.AttributeConstraint(my_attribute='foobar'))
        self.assertIsNone(cube)
        
        orig_cube = self.cube
        # add an attribute to the cubes
        orig_cube.attributes['my_attribute'] = 'foobar'
        
        cube = orig_cube.extract(iris.AttributeConstraint(my_attribute='foobar'))
        self.assertCML(cube, ('constrained_load', 'attribute_constraint.cml'))
        
        cube = orig_cube.extract(iris.AttributeConstraint(my_attribute='not me'))
        self.assertIsNone(cube)
        
        cube = orig_cube.extract(iris.AttributeConstraint(my_attribute=lambda val: val.startswith('foo')))
        self.assertCML(cube, ('constrained_load', 'attribute_constraint.cml'))
        
        cube = orig_cube.extract(iris.AttributeConstraint(my_attribute=lambda val: not val.startswith('foo')))
        self.assertIsNone(cube)
        
        cube = orig_cube.extract(iris.AttributeConstraint(my_non_existant_attribute='hello world'))
        self.assertIsNone(cube)
        
    def test_standard_name(self):
        r = iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE)
        self.assertTrue(self.cube.extract(r).standard_name, SN_AIR_POTENTIAL_TEMPERATURE)
        
        r = iris.Constraint('wibble')
        self.assertEqual(self.cube.extract(r), None)

    def test_empty_data(self):
        # Ensure that the process of WHERE does not load data if there was empty data to start with...
        self.assertNotEquals(None, self.cube._data_manager) 
        
        self.assertNotEquals(None, self.cube.extract(self.level_10)._data_manager)
        
        self.assertNotEquals(None, self.cube.extract(self.level_10).extract(self.level_10)._data_manager)
            
    def test_non_existant_coordinate(self):
        # Check the behaviour when a constraint is given for a coordinate which does not exist/span a dimension
        self.assertEqual(self.cube[0, :, :].extract(self.level_10), None)
        
        self.assertEqual(self.cube.extract(iris.Constraint(wibble=10)), None)
        

@iris.tests.skip_data
class TestConstraints(TestMixin, tests.IrisTest):
    def test_constraint_expressions(self):
        rt = repr(self.theta)
        rl10 = repr(self.level_10)

        rt_l10 = repr(self.theta & self.level_10)
        self.assertEqual(rt_l10, "ConstraintCombination(%s, %s, <built-in function __and__>)" % (rt, rl10))

    def test_string_repr(self):
        rt = repr(iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE))
        self.assertEqual(rt, "Constraint(name='%s')" % SN_AIR_POTENTIAL_TEMPERATURE)

        rt = repr(iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE, model_level_number=10))
        self.assertEqual(rt, "Constraint(name='%s', coord_values={'model_level_number': 10})" % SN_AIR_POTENTIAL_TEMPERATURE)

    def test_number_of_raw_cubes(self):
        # Test the constraints generate the correct number of raw cubes.    
        raw_cubes = iris.load_raw(self.theta_path)
        self.assertEqual(len(raw_cubes), 38)

        raw_cubes = iris.load_raw(self.theta_path, [self.level_10])
        self.assertEqual(len(raw_cubes), 1)

        raw_cubes = iris.load_raw(self.theta_path, [self.theta])
        self.assertEqual(len(raw_cubes), 38)

        raw_cubes = iris.load_raw(self.dec_path, [self.level_30])
        self.assertEqual(len(raw_cubes), 4)

        raw_cubes = iris.load_raw(self.dec_path, [self.theta])
        self.assertEqual(len(raw_cubes), 38)
       

class TestBetween(tests.IrisTest):
    def run_test(self, function, numbers, results):
        for number, result in zip(numbers, results):
            self.assertEqual(function(number), result)
        
    def test_le_ge(self):
        function = iris.util.between(2, 4)
        numbers = [1, 2, 3, 4, 5]
        results = [False, True, True, True, False]
        self.run_test(function, numbers, results)
        
    def test_lt_gt(self):
        function = iris.util.between(2, 4, rh_inclusive=False, lh_inclusive=False)
        numbers = [1, 2, 3, 4, 5]
        results = [False, False, True, False, False]
        self.run_test(function, numbers, results)
        
    def test_le_gt(self):
        function = iris.util.between(2, 4, rh_inclusive=False)
        numbers = [1, 2, 3, 4, 5]
        results = [False, True, True, False, False]
        self.run_test(function, numbers, results)
        
    def test_lt_ge(self):
        function = iris.util.between(2, 4, lh_inclusive=False)
        numbers = [1, 2, 3, 4, 5]
        results = [False, False, True, True, False]
        self.run_test(function, numbers, results)


class TestEquality(tests.IrisTest):
    def _constraint_int(self, cls):
        a = cls(model=1)
        b = cls(model=1)
        c = cls(model=2)
        d = cls(wibble=1)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)

    def test_constraint_int(self):
        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._constraint_int(constraint)

    def _constraint_float(self, cls):
        a = cls(model=1.1)
        b = cls(model=1.1)
        c = cls(model=2.2)
        d = cls(wibble=1.1)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)

    def test_constraint_float(self):
        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._constraint_float(constraint)

    def _constraint_string(self, cls):
        a = cls(model='hello')
        b = cls(model='hello')
        c = cls(model=1)
        d = cls(wibble='hello')
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)

    def test_constraint_string(self):
        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._constraint_string(constraint)

    def _constraint_list(self, cls):
        a = cls(model=[10, 20])
        b = cls(model=[20, 10])
        c = cls(model=10)
        d = cls(wibble=[10, 20])
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)

    def test_constraint_list(self):
        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._constraint_list(constraint)

    def _constraint_tuple(self, cls):
        a = cls(model=(10, 20))
        b = cls(model=(20, 10))
        c = cls(model=10)
        d = cls(wibble=(10, 20))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)

    def test_constraint_tuple(self):
        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._constraint_tuple(constraint)

    def _constraint_set(self, cls):
        a = cls(model=set([10, 20]))
        b = cls(model=set([20, 10]))
        c = cls(model=10)
        d = cls(wibble=set([10, 20]))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)

    def test_constraint_set(self):
        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._constraint_set(constraint)

    def _constraint_dictionary(self, cls):
        a = cls(model={'one': 1, 'two': 2})
        b = cls(model={'two': 2, 'one': 1})
        c = cls(model=10)
        d = cls(wibble={'one': 1, 'two': 2})
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)

    def test_constraint_dictionary(self):
        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._constraint_dictionary(constraint)

    def _constraint_lambda(self, cls):
        common = lambda x: x > 10
        a = cls(model=lambda x: x > 10)
        b = cls(model=lambda x: x > 10)
        c = cls(model=10)
        d = cls(wibble=lambda x: x > 10)
        e = cls(model=common)
        f = cls(model=common)
        g = cls(wibble=common)
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)
        self.assertEqual(e, f)
        self.assertNotEqual(e, g)

    def test_constraint_lambda(self):
        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._constraint_lambda(constraint)

    def _constraint_func(self, cls):
        def func_1(arg):
            return arg > 10

        def func_2(arg):
            return arg > 10

        a = cls(model=func_1)
        b = cls(model=func_2)
        c = cls(model=10)
        d = cls(wibble=func_2)
        e = cls(model=func_1)
        f = cls(wibble=func_1)
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)
        self.assertEqual(a, e)
        self.assertNotEqual(a, f)

    def test_constraint_func(self):
        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._constraint_func(constraint)

    def test_constraint_cube_func(self):
        def func_1(cube):
            return True

        def func_2(cube):
            return True

        a = iris.Constraint(cube_func=func_1)
        b = iris.Constraint(cube_func=func_2)
        c = iris.Constraint(wibble=func_1)
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        d = iris.Constraint(cube_func=func_1)
        self.assertEqual(a, d)

    def test_constraint_combination(self):
        a = iris.Constraint(model=10)
        b = iris.Constraint(model=20)
        c = iris.Constraint(model=30)
        d = iris.Constraint(model=10)
        e = iris.Constraint(model=20)
        f = iris.Constraint(model=30)
        c1 = a & b
        c2 = b & a
        c3 = d & e
        self.assertNotEqual(c1, c2)
        self.assertEqual(c1, c3)
        c4 = a & b & c
        c5 = b & a & c
        c6 = a & c & b
        c7 = d & e & f
        self.assertNotEqual(c4, c5)
        self.assertNotEqual(c4, c6)
        self.assertEqual(c4, c7)


class TestPickle(tests.IrisTest):
    # It's not valid to pickle a function or lambda.

    def _pickle(self, cls, value):
        buf = cStringIO.StringIO()
        expected = cls(model=value)
        cPickle.dump(expected, buf, cPickle.HIGHEST_PROTOCOL)
        buf.seek(0)
        actual = cPickle.load(buf)
        self.assertEqual(expected, actual)

    def test_pickle_int(self):
        self._pickle(iris.Constraint, 10)
        self._pickle(iris.AttributeConstraint, 10)

    def test_pickle_float(self):
        self._pickle(iris.Constraint, 1.1)
        self._pickle(iris.AttributeConstraint, 1.1)

    def test_pickle_string(self):
        self._pickle(iris.Constraint, 'branston')
        self._pickle(iris.AttributeConstraint, 'chutney')

    def test_pickle_list(self):
        self._pickle(iris.Constraint, range(10))
        self._pickle(iris.AttributeConstraint, range(10))

    def test_pickle_tuple(self):
        t = tuple(range(10))
        self._pickle(iris.Constraint, t)
        self._pickle(iris.AttributeConstraint, t)

    def test_pickle_set(self):
        s = set(range(10))
        self._pickle(iris.Constraint, s)
        self._pickle(iris.AttributeConstraint, s)

    def test_pickle_dictionary(self):
        d = {'one': 1, 'two': 2, 'three': 3}
        self._pickle(iris.Constraint, d)
        self._pickle(iris.AttributeConstraint, d)

    def test_pickle_combination(self):
        a = iris.Constraint(a=1)
        b = iris.Constraint(b=2)
        c = iris.AttributeConstraint(c=3)
        expected = a & b & c
        buf = cStringIO.StringIO()
        cPickle.dump(expected, buf, cPickle.HIGHEST_PROTOCOL)
        buf.seek(0)
        actual = cPickle.load(buf)
        self.assertEqual(expected, actual)


class TestHash(tests.IrisTest):
    def _hash(self, cls, v1, v2, compare):
        a = cls(model=v1, depth=v1)
        b = cls(model=v2, depth=v2)
        compare(hash(a), hash(b))

    def test_hash_int(self):
        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._hash(constraint, 10, 10, self.assertEqual)
            self._hash(constraint, 10, 20, self.assertNotEqual)

    def test_hash_float(self):
        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._hash(constraint, 1.1, 1.1, self.assertEqual)
            self._hash(constraint, 1.1, 2.2, self.assertNotEqual)

    def test_hash_string(self):
        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._hash(constraint, 'hello', 'hello', self.assertEqual)
            self._hash(constraint, 'hello', 'goodbye', self.assertNotEqual)

    def test_hash_list(self):
        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._hash(constraint, range(10), range(10), self.assertEqual)
            self._hash(constraint, range(10), range(5), self.assertNotEqual)

    def test_hash_tuple(self):
        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._hash(constraint, tuple(range(10)), tuple(range(10)),
                       self.assertEqual)
            self._hash(constraint, tuple(range(10)), tuple(range(5)),
                       self.assertNotEqual)

    def test_hash_set(self):
        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._hash(constraint, set(range(10)), set(range(10)),
                       self.assertEqual)
            self._hash(constraint, set(range(10)), set(range(5)),
                       self.assertNotEqual)

    def test_hash_dictionary(self):
        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._hash(constraint, {'one': 1, 'two': 2},
                       {'two': 2, 'one': 1}, self.assertEqual)
            self._hash(constraint, {'one': 1, 'two': 2},
                       {'one': 'wibble', 'two': 'wobble'}, self.assertEqual)
            self._hash(constraint, {'one': 1, 'two': 2},
                       {'one': 1, 'too': 2}, self.assertNotEqual)

    def test_hash_lambda(self):
        common = lambda x: x > 1
        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._hash(constraint, lambda x: x > 1,
                       lambda x: x > 1, self.assertNotEqual)
            self._hash(constraint, common, common, self.assertEqual)

    def test_hash_func(self):
        def func_1(arg):
            return arg > 10

        def func_2(arg):
            return arg > 10

        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._hash(constraint, func_1, func_2, self.assertNotEqual)
            self._hash(constraint, func_1, func_1, self.assertEqual)

    def test_hash_cube_func(self):
        def func_1(cube):
            return True

        def func_2(cube):
            return True

        a = iris.Constraint(cube_func=func_1)
        b = iris.Constraint(cube_func=func_2)
        c = iris.Constraint(cube_func=func_1)
        self.assertNotEqual(hash(a), hash(b))
        self.assertEqual(hash(a), hash(c))

    def test_hash_combination(self):
        a = iris.Constraint(model=10)
        b = iris.Constraint(model=20)
        c = iris.Constraint(model=10)
        d = iris.Constraint(model=20)
        c1 = a & b
        c2 = b & a
        c3 = c & d
        self.assertNotEqual(hash(c1), hash(c2))
        self.assertEqual(hash(c1), hash(c3))

    def test_hash_lookup(self):
        a = iris.Constraint(model=10)
        b = iris.AttributeConstraint(model=10)
        c = a & b
        d = iris.Constraint(model=10)
        e = iris.AttributeConstraint(model=10)
        f = d & e
        lookup = {a: 'Constraint',
                  b: 'AttributeConstraint',
                  c: 'ConstraintCombination'}
        self.assertEqual(lookup[d], 'Constraint')
        self.assertEqual(lookup[e], 'AttributeConstraint')
        self.assertEqual(lookup[f], 'ConstraintCombination')


class TestCallable(tests.IrisTest):
    def _callable(self, cls):
        def func(arg):
            return True

        a = cls(model=lambda x: x > 1)
        b = cls(model=func)
        c = cls(model=10)
        d = cls(model=10, depth=func)
        e = cls(model=10, depth=lambda x: x > 1, level=20)
        self.assertTrue(a.callable())
        self.assertTrue(b.callable())
        self.assertFalse(c.callable())
        self.assertTrue(d.callable())
        self.assertTrue(e.callable())

    def test_callable(self):
        def func(arg):
            return True

        for constraint in [iris.Constraint, iris.AttributeConstraint]:
            self._callable(constraint)
        a = iris.Constraint(cube_func=None, model=10)
        b = iris.Constraint(cube_func=func)
        self.assertFalse(a.callable())
        self.assertTrue(b.callable())

    def test_callable_combination(self):
        def func(arg):
            return True

        a = iris.Constraint(model=10)
        b = iris.Constraint(model=lambda x: x > 1)
        c = iris.Constraint(model=20)
        d = iris.Constraint(model=func)
        c1 = a & b
        c2 = a & c
        c3 = a & d
        c4 = a & c & d
        c5 = a & b & c
        self.assertTrue(c1.callable())
        self.assertFalse(c2.callable())
        self.assertTrue(c3.callable())
        self.assertTrue(c4.callable())
        self.assertTrue(c5.callable())


if __name__ == "__main__":
    tests.main()
