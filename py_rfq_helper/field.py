import sys
from warp import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from field_from_two_term_potential import *

# FILENAME = "vecc_rfq_003_py.dat"
# FILENAME = "vecc_rfq_004_py.dat"

class FieldLoader(object):
    def __init__(self):

        self._filename = None

        self._z_length = 0.0
        self._ex = None
        self._ey = None
        self._ez = None
        self._nx = None
        self._ny = None
        self._nz = None
        self._dx = None
        self._dy = None
        self._dz = None
        self._xmin = 0.0
        self._xmax = 0.0
        self._ymin = 0.0
        self._ymax = 0.0
        self._zmin = 0.0
        self._zmax = 0.0

        self._vane_profile = np.array([])


    def load_field_from_file(self, filename=None):
        # Loads the field from a dat file.
        # Param: filename
        # Return: 
        # Code mostly taken from Warp website,
        # https://sites.google.com/a/lbl.gov/warp/home/how-to-s/saving-data/saving-variables
        
        if filename is None:
            print("No filename specified for field file. Closing.")
            exit(1)

        if filename is not None:
            self._filename = filename

        [x, y, z, e_x, e_y, e_z] = getdatafromtextfile(filename, nskip=1, dims=[6, None])

        self.parse_field(x, y, z, e_x, e_y, e_z)
        

    def load_field_from_cells(self, filename=None):
        # Loads and calculates field from PARMTEQ file using Two Term potential method

        if filename is None:
            print("No filename specified for cell file. Closing.")
            exit(1)
        else:
            self._filename = filename

        loadpath = filename

        fg = FieldGenerator(resolution=0.002)
        fg.set_bempp_mesh_size(0.002)

        fg.load_parameters_from_file(filename=loadpath)

        fg.add_cell(cell_type="TCS",
                    aperture=0.011255045027294745,
                    modulation=1.6686390559337798,
                  length=0.0427)
        # 0.10972618296477678

        fg.add_cell(cell_type="DCS",
                    aperture=0.015017826368066015,
                    modulation=1.0,
                    length=0.13)

        # Second Drift is necessary to keep cell count correct
        # fg.add_cell(cell_type="DCS",
        #             aperture=0.015,
        #             modulation=1.0,
        #             length=0.02)

        # fg.add_cell(cell_type="TCS",
        #             aperture=0.01,
        #             modulation=2.0,
        #             length=0.041,
        #             flip_z=True)

        # fg.add_cell(cell_type="TCS",
        #             aperture=0.01,
        #             modulation=2.0,
        #             length=0.041,
        #             shift_cell_no=True)

        # fg.set_plot_tcs_vanes(True)
        fg.set_calculate_vane_profile(True)
        fg.generate()

        # fg.plot_pot_of_z()
        # fg.plot_combo()
        # fg.plot_e_xy(z=1.33)
        # fg.plot_ex_of_z(x=0.05)
        # fg.plot_pot_xy(z=1.33)
        # fg.plot_ez_of_z()

        # fg.write_inventor_macro(save_folder=folder)

        x = fg._mesh_x.flatten()
        y = fg._mesh_y.flatten()
        z = fg._mesh_z.flatten()
        ex = fg._ex.flatten()
        ey = fg._ey.flatten()
        ez = fg._ez.flatten()

        self._vane_profile = fg._vane_profile_x

        self.parse_field(x, y, z, ex, ey, ez)



    def parse_field(self, x, y, z, e_x, e_y, e_z):
        # Loads and parses the field given arrays with x, y, z positions and 
        # corresponding ex, ey, ez arrays. 

        self._z_length = z.max() - z.min()

        dx = (x.max() - x.min()) / (len(unique(x)) - 1) 
        dy = (y.max() - y.min()) / (len(unique(y)) - 1) 
        dz = (z.max() - z.min()) / (len(unique(z)) - 1) 

        nx = nint((x.max() - x.min()) / dx)
        ny = nint((y.max() - y.min()) / dy)
        nz = nint((z.max() - z.min()) / dz)

        ex = fzeros((nx+1, ny+1, nz+1))
        ey = fzeros((nx+1, ny+1, nz+1))
        ez = fzeros((nx+1, ny+1, nz+1))

        ix = nint((x - x.min()) / dx)
        iy = nint((y - y.min()) / dy)
        iz = nint((z - z.min()) / dz)

        ii = ix + (nx+1)*iy + (nx+1)*(ny+1)*iz

        ex.ravel(order='F').put(ii,e_x)
        ey.ravel(order='F').put(ii,e_y)
        ez.ravel(order='F').put(ii,e_z)


        self._ex = ex
        self._ey = ey
        self._ez = ez
        self._dx = dx
        self._dy = dy
        self._dz = dz 
        self._nx = nx
        self._ny = ny
        self._nz = nz 
        self._xmin = x.min()
        self._xmax = x.max()
        self._ymin = y.min()
        self._ymax = y.max()
        self._zmin = z.min()
        self._zmax = z.max()