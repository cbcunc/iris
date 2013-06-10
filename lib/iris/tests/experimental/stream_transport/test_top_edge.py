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
Test the top_edge path finding module.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import iris
import iris.experimental.stream_transport.top_edge as top_edge
import iris.experimental.stream_transport.line_walk as line_walk


class TestPathFinder(tests.IrisTest):

    def setUp(self):
        # Test the path finding algorithm defined in the given module.
        self.cube = iris.load_cube(tests.get_data_path(('NetCDF', 'XXX', "XXX.nc")))
        
        # prepare a checker-board dataset to draw on
        self.checkers = np.zeros(cube.shape, dtype=int)
        self.checkers[::2, ::2] = 1
        self.checkers[1::2, 1::2] = 1
        self.checkers[0,0] = 8

    def ij_plot(self, col):
        plt.pcolormesh(self.checkers, cmap="binary")
        for i in range(len(input_lats)):
            for seg in results[i]:
                seg = np.array(seg)
                plt.plot(seg[:,1], seg[:,0], c=col, linewidth=2)

    def test_top_edge(self):
        input_lats = [60, 70, 80]
        results = []
        for lat in input_lats:
            result = module.find_path(self.cube, lat=lat)
            results.append(result)

        self.ij_plot("red")
        self.check_graphic()


if __name__ == "__main__":
    tests.main()
