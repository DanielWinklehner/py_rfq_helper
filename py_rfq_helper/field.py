import sys
from warp import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
#from field_from_two_term_potential import *
from py_rfq_designer import *

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

        self._fg = None
        self._myrfq = None

        self._vane_profile = np.array([])

    def add_cell(self,
                 cell_type,
                 aperture,
                 modulation,
                 length,
                 flip_z=False,
                 shift_cell_no=False):

        self._fg.add_cell(cell_type, aperture, modulation, length, flip_z, shift_cell_no)

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
        

    def load_field_from_cells_tt(self, voltage, frequency, a_init, xy_limits, filename=None, resolution=0.002, ignore_rms=False):
        # Loads and calculates field from PARMTEQ file using Two Term potential method

        if filename is None:
            print("No filename specified for cell file. Closing.")
            exit(1)
        else:
            self._filename = filename

        loadpath = filename

        self._fg = FieldGenerator(resolution=resolution, xy_limits=xy_limits, ignore_rms=ignore_rms)

        self._fg._voltage   = voltage
        self._fg._frequency = frequency
        self._fg._a_init    = a_init

        self._fg.set_bempp_mesh_size(resolution)

        self._fg.load_parameters_from_file(filename=loadpath)


    def generate_field_from_cells_tt(self):
        self._fg.set_calculate_vane_profile(True)
        self._fg.generate()

        x = self._fg._mesh_x.flatten()
        y = self._fg._mesh_y.flatten()
        z = self._fg._mesh_z.flatten()
        ex = self._fg._ex.flatten()
        ey = self._fg._ey.flatten()
        ez = self._fg._ez.flatten()

        self._vane_profile = self._fg._vane_profile_x

        self.parse_field(x, y, z, ex, ey, ez)

    def load_field_from_cells_bempp(self, voltage, cyl_id, grid_res, pot_shift, add_endplates=True,filename=None):
        self._myrfq = PyRFQ(voltage=voltage, debug=True)

        # Load the base RFQ design from the parmteq file
        if myrfq.add_cells_from_file(filename=filename, ignore_rms=True) == 1:
            print("Something went wrong. Please check your file.")
            exit(1)


        myrfq.append_cell(cell_type="TCS",
                          aperture=0.007147,
                          modulation=1.6778,
                          length=0.033840)

        myrfq.append_cell(cell_type="DCS",
                          aperture=0.0095691183,
                          modulation=1.0,
                          length=0.1)


    def generate_field_from_cells_bempp(self, cyl_id, grid_res, pot_shift, add_endplates=True):
        # myrfq.set_bempp_parameter("add_endplates", True)
        # myrfq.set_bempp_parameter("cyl_id", 0.1)
        # myrfq.set_bempp_parameter("grid_res", 0.005)
        # myrfq.set_bempp_parameter("pot_shift", 3.0 * 22000.0)

        self._myrfq.set_bempp_parameter("add_endplates", add_endplates)
        self._myrfq.set_bempp_parameter("cyl_id", cyl_id)
        self._myrfq.set_bempp_parameter("grid_res", grid_res)
        self._myrfq.set_bempp_parameter("pot_shift", pot_shift)

        print("Generating vanes")
        ts = time.time()
        self._myrfq.generate_vanes()
        print("Generating vanes took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))

        print("Generating full mesh for BEMPP")
        ts = time.time()
        self._myrfq.generate_full_mesh()
        print("Meshing took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))

        print("Solving BEMPP problem")
        ts = time.time()
        self._myrfq.solve_bempp()
        print("Solving BEMPP took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))

        print("Calculating Potential")
        ts = time.time()
        myres = [0.002, 0.002, 0.002]
        limit = 0.02
        self._myrfq.calculate_potential(limits=((-limit, limit), (-limit, limit), (-0.1, 1.35)),
                                  res=myres,
                                  domain_decomp=(1, 1, 50),
                                  overlap=0)
        print("Potential took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))


    def parse_field(self, x, y, z, e_x, e_y, e_z):
        # Loads and parses the field given arrays with x, y, z positions and 
        # corresponding ex, ey, ez arrays. 

        self._z_length = z.max() - z.min()

        dx = (x.max() - x.min()) / (len(unique(x)) - 1) 
        dy = (y.max() - y.min()) / (len(unique(y)) - 1) 
        dz = (z.max() - z.min()) / (len(unique(z)) - 1) 

        print(unique(x))
        print("----------------------------------")
        print(unique(y))
        
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


