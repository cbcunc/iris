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


# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import matplotlib.pyplot as plt
import numpy

import iris.analysis.trajectory
import iris.quickplot as qplt
import iris.tests.stock


class TestSimple(tests.IrisTest):
    def test_invalid_coord(self):
        cube = iris.tests.stock.realistic_4d()
        sample_points = [('altitude', [0, 10, 50])]
        with self.assertRaises(ValueError):
            iris.analysis.trajectory.interpolate(cube, sample_points, 'nearest')


class TestTrajectory(tests.IrisTest):
    def assertCML(self, cube, path, *args, **kwargs):
        try:
            coord = cube.coord('model_level_number')
            coord._TEST_COMPAT_force_explicit = True
            coord._TEST_COMPAT_override_axis = 'z'
        except iris.exceptions.CoordinateNotFoundError:
            pass
        try:
            coord = cube.coord('sigma')
            coord._TEST_COMPAT_override_axis = 'z'
        except iris.exceptions.CoordinateNotFoundError:
            pass
        try:
            coord = cube.coord('grid_latitude')
            coord._TEST_COMPAT_force_explicit = True
            coord._TEST_COMPAT_definitive = True
        except iris.exceptions.CoordinateNotFoundError:
            pass
        try:
            coord = cube.coord('time')
            coord._TEST_COMPAT_force_explicit = True
        except iris.exceptions.CoordinateNotFoundError:
            pass
        try:
            coord = cube.coord('surface_altitude')
            coord._TEST_COMPAT_points = False
        except iris.exceptions.CoordinateNotFoundError:
            pass
        super(TestTrajectory, self).assertCML(cube, path, *args, **kwargs)

    def test_trajectory_definition(self):
        # basic 2-seg line along x
        waypoints = [ {'lat':0, 'lon':0}, {'lat':0, 'lon':1}, {'lat':0, 'lon':2} ]
        trajectory = iris.analysis.trajectory.Trajectory(waypoints, sample_count=21)
        
        self.assertEqual(trajectory.length, 2.0)
        self.assertEqual(trajectory.sampled_points[19], {'lat': 0.0, 'lon': 1.9000000000000001})
        
        # 4-seg m-shape 
        waypoints = [ {'lat':0, 'lon':0}, {'lat':1, 'lon':1}, {'lat':0, 'lon':2}, {'lat':1, 'lon':3}, {'lat':0, 'lon':4} ]
        trajectory = iris.analysis.trajectory.Trajectory(waypoints, sample_count=33)
        
        self.assertEqual(trajectory.length, 5.6568542494923806)
        self.assertEqual(trajectory.sampled_points[31], {'lat': 0.12499999999999989, 'lon': 3.875})

    @iris.tests.skip_data
    def test_trajectory_extraction(self):

        # Load the COLPEX data => TZYX
        path = tests.get_data_path(['PP', 'COLPEX', 'theta_and_orog_subset.pp'])
        cube = iris.load_strict(path, 'air_potential_temperature')
        cube.coord('grid_latitude').bounds = None
        cube.coord('grid_longitude').bounds = None
        # TODO: Workaround until regrid can handle factories
        cube.remove_aux_factory(cube.aux_factories[0])
        cube.remove_coord('surface_altitude')
        self.assertCML(cube, ('trajectory', 'big_cube.cml'))
        
        # Pull out a single point
        single_point = cube.extract_by_trajectory([{'grid_latitude': -0.1188, 'grid_longitude': 359.57958984}])
        self.assertCML(single_point, ('trajectory', 'single_point.cml'))
        
        # Extract a simple, axis-aligned trajectory that is similar to an indexing operation.
        # (It's not exactly the same because the source cube doesn't have regular spacing.)
        waypoints = [
            {'grid_latitude': -0.1188, 'grid_longitude': 359.57958984},
            {'grid_latitude': -0.1188, 'grid_longitude': 359.66870117}
        ]
        trajectory = iris.analysis.trajectory.Trajectory(waypoints, sample_count=100)
        trajectory_cube = cube.extract_by_trajectory(trajectory)
        self.assertCML(trajectory_cube, ('trajectory', 'constant_latitude.cml'))

        # Sanity check the results against a simple slice
        plt.plot(cube[0, 0, 10, :].data)
        plt.plot(trajectory_cube[0, 0, :].data)
        self.check_graphic()
        
        # Extract a zig-zag trajectory
        waypoints = [
            {'grid_latitude': -0.1188, 'grid_longitude': 359.5886},
            {'grid_latitude': -0.0828, 'grid_longitude': 359.6606},
            {'grid_latitude': -0.0468, 'grid_longitude': 359.6246},
        ]
        trajectory = iris.analysis.trajectory.Trajectory(waypoints, sample_count=100)
        trajectory_cube = cube.extract_by_trajectory(trajectory)
        self.assertCML(trajectory_cube, ('trajectory', 'zigzag.cml'))

        # Sanity check the results against a simple slice
        x = cube.coord('grid_longitude').points
        y = cube.coord('grid_latitude').points
        plt.pcolormesh(x, y, cube[0, 0, :, :].data)
        x = trajectory_cube.coord('grid_longitude').points
        y = trajectory_cube.coord('grid_latitude').points
        plt.scatter(x, y, c=trajectory_cube[0, 0, :].data)
        self.check_graphic()

    @iris.tests.skip_data
    def test_tri_polar(self):
        # load data
        cubes = iris.load(tests.get_data_path(['NetCDF', 'ORCA2', 'votemper.nc']))
        cube = cubes[0]
        # The netCDF file has different data types for the points and
        # bounds of 'depth'. This wasn't previously supported, so we
        # emulate that old behaviour.
        cube.coord('depth').bounds = cube.coord('depth').bounds.astype(numpy.float32)

        # define a latitude trajectory (put coords in a different order to the cube, just to be awkward)
        latitudes = range(-90, 90, 2)
        longitudes = [-90]*len(latitudes)
        sample_points = [('longitude', longitudes), ('latitude', latitudes)]

        # extract
        sampled_cube = iris.analysis.trajectory.interpolate(cube, sample_points)
        coord = sampled_cube.coord('longitude')
        coord._TEST_COMPAT_override_axis = 'nav_lon'
        coord = sampled_cube.coord('latitude')
        coord._TEST_COMPAT_override_axis = 'nav_lat'
        coord = sampled_cube.coord('depth')
        coord._TEST_COMPAT_override_axis = 'z'
        self.assertCML(sampled_cube, ('trajectory', 'tri_polar_latitude_slice.cml'))

        # turn it upside down for the visualisation
        plot_cube = sampled_cube[0]
        plot_cube = plot_cube[::-1, :]

        plt.clf()
        plt.pcolormesh(plot_cube.data, vmin=cube.data.min(), vmax=cube.data.max())
        plt.colorbar()
        self.check_graphic()
        
        # Try to request linear interpolation.
        # Not allowed, as we have multi-dimensional coords.
        self.assertRaises(iris.exceptions.CoordinateMultiDimError, iris.analysis.trajectory.interpolate, cube, sample_points, method="linear")

        # Try to request unknown interpolation.
        self.assertRaises(ValueError, iris.analysis.trajectory.interpolate, cube, sample_points, method="linekar")


if __name__ == '__main__':
    tests.main()
