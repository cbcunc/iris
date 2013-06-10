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
Test the stream trnasport module.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import iris
import iris.experimental.stream_transport as stream_transport


class TestStreamTransport():
    
    def load_data(self):
        input_data = {"t": None, "u": None, "v":None, "region_mask": None,
                      "dx": None, "dy": None, "dxu": None, "dzv": None}
        return input_data
    
    def test_net_transport(self):
        pass
    
    def test_stream_function(self):
        pass


if __name__ == "__main__":
    tests.main()
