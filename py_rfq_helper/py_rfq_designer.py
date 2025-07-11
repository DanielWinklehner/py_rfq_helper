# py_rfq_designer.py
# Contains the FieldGenerator, PyRFQCell, PyRFQVane and PyRFQ classes.

# noinspection PyUnresolvedReferences
from warp import *
# from dans_pymodules import *
from .field_utils import *
import scipy.constants as const

# Check if we can connect to a display, if not disable all plotting and windowed stuff (like gmsh)
# TODO: This does not remotely cover all cases!
import os
if "DISPLAY" in os.environ.keys():
    x11disp = True
else:
    x11disp = False

try:
    from mpi4py import MPI
except ImportError:
    print("Could not import mpi4py!")
    MPI = None
    exit()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = MPI.Get_processor_name()

print("Process {} of {} on host {} started!".format(rank, size, host))

np.set_printoptions(threshold=10000)

# For now, everything involving the pymodules with be done on master proc (rank 0)
# if rank == 0:
#     from dans_pymodules import *

#     colors = MyColors()
# else:
#     colors = None

decimals = 12

__author__ = "Daniel Winklehner, Siddhartha Dechoudhury"
__doc__ = """Calculate RFQ fields from loaded cell parameters"""

# Initialize some global constants
amu = const.value("atomic mass constant energy equivalent in MeV")
echarge = const.value("elementary charge")
clight = const.value("speed of light in vacuum")

# Define the directions:
X = 0
Y = 1
Z = 2
XYZ = range(3)
AXES = {"X": 0, "Y": 1, "Z": 2}

# --- This is a nice implementation of a simple timer I found online -DW --- #
_tm = 0

def stopwatch(msg=''):
    tm = time.time()
    global _tm
    if _tm == 0:
        _tm = tm
        return
    print("%s: %.2f ms" % (msg, 1000.0 * (tm - _tm)))
    _tm = tm
# ------------------------------------------------------------------------- #

rot_map = {"yp": 0.0,
           "ym": 180.0,
           "xp": 270.0,
           "xm": 90.0}

# noinspection PyUnresolvedReferences
class PyRFQ(object):
    def __init__(self, filename, sim_sta, sim_end, sim_radius, voltage=None, debug=False):

        self._vanes = []
        self._cells = []
        self._cell_nos = []
        self._length = 0.0

        ###### From rfq helper class ######

        # User Parameters
        #     Passed in via constructor
        self.debug = debug  # flag for additional debugging output
        self.voltage = voltage  # vane voltage (currently unused in calculations)

        self.field_scaling_factor = 1  # factor to scale the e field magnitude by
        self.filename = filename  # field filename

        self.rfq_start = 0.0  # start of the physical rfq model
        self.rfq_end = None

        self.field_start = 0.0  # start of the ecternally generated e-field
        self.field_end = None

        self.sim_start = sim_sta  # start of the simulation
        self.sim_end = sim_end
        self.sim_radius = sim_radius

        # Variables for RFQ conductors
        self.simple_vanes = False  # whether we want simple rods as dummy vanes
        self.vane_radius = None  # radius of simple rods (m)
        self.vane_distance = None  # vane distance from axis (m)
        self.vane_sta = None  # vane/rod start (m)
        self.vane_end = None  # vane/rod end (m)
        self.vane_from_profile = False  # whether we want to generate a series of thin slices following the vane profile
        self.vane_profile = None  # N x 2 numpy array of x/y pairs for vane profile

        self.tank_sta = None  # start of the tank cylinder (m)
        self.tank_end = None  # end of the tank cylinder (m)
        self.tank_id = None  # tank inner diameter (m)

        # Note: tank from data overrides generating tank from cylinder and (optional) end plates
        self.tank_from_data = False  # whether we would like to generate the tank from a set of x/y pairs
        self.tank_data = None  # N x 2 numpy array of x/y pairs for inner tank surface

        # Optional, note: endplates always touch the beginning and end of the tank
        self.endplates = False
        # self.endplates_outer_diameter = 0.2
        self.entrance_plate_id = 0.1  # inner diameter of entrance aperture (m)
        self.exit_plate_id = 0.0  # inner diameter of entrance aperture (m)
        self.endplates_thickness = 0.02  # (m)

        # "Private" variables
        self._conductors    = None
        self._field         = FieldLoader()


    def __str__(self):
        text = "Summary of RFQ parameters goes here..."

        return text

    # def append_cell(self,
    #                 cell_type,
    #                 aperture,
    #                 modulation,
    #                 length,
    #                 flip_z=False,
    #                 shift_cell_no=False):
    #     assert cell_type in ["STA", "RMS", "NCS", "TCS", "DCS"], "cell_type must be one of STA, RMS, NCS, TCS, DCS!"
    #
    #     if len(self._cells) > 0:
    #         pc = self._cells[-1]
    #     else:
    #         pc = None
    #
    #     self._cells.append(PyRFQCell(cell_type=cell_type,
    #                                  aperture=aperture,
    #                                  modulation=modulation,
    #                                  length=length,
    #                                  flip_z=flip_z,
    #                                  shift_cell_no=shift_cell_no,
    #                                  prev_cell=pc,
    #                                  next_cell=None))
    #
    #     if len(self._cells) > 1:
    #         self._cells[-2].set_next_cell(self._cells[-1])
    #
    #     self._cell_nos = range(len(self._cells))
    #     self._length = np.sum([cell.length for cell in self._cells])
    #
    #     return 0

    # def add_cells_from_file(self, filename=None, ignore_rms=False):
    #     """
    #     Reads a file with cell parameters and generates the respective RFQCell objects
    #     :param filename:
    #     :param ignore_rms: Bool. If True, any radial matching cells in the file are ignored.
    #     :return:
    #     """
    #
    #     if filename is None:
    #         if rank == 0:
    #             # print("Process {} getting filename from dialog".format(rank))
    #             # from dans_pymodules import FileDialog
    #             fd = FileDialog()
    #             filename = fd.get_filename('open')
    #             data = {"fn": filename}
    #             # req = comm.isend({'fn':filename}, dest=1, tag=11)
    #             # req.wait()
    #         else:
    #             # req = comm.irecv(source=0, tag=11)
    #             data = None
    #             # print("Process {} received filename {}.".format(rank, data["fn"]))
    #
    #         data = comm.bcast(data, root=0)
    #         filename = data["fn"]
    #
    #     if filename is None:
    #         return 1
    #
    #     with open(filename, "r") as infile:
    #         if "Parmteqm" in infile.readline():
    #             # Detected Parmteqm file
    #             self.read_input_parmteq(filename, ignore_rms)
    #         else:
    #             # Assume only other case is VECC input file for now
    #             self.read_input_vecc(filename, ignore_rms)
    #
    #     return 0
    #
    # def read_input_parmteq(self, filename, ignore_rms):
    #     # Parameters: Filename, whether to ignore rms
    #     # Returns: None
    #     # Reads in cell data from a parmteq file
    #
    #     with open(filename, "r") as infile:
    #
    #         # Some user feedback:
    #         version = infile.readline().strip().split()[1].split(",")[0]
    #         print("Loading cells from Parmteqm v{} output file...".format(version))
    #
    #         # Find begin of cell information
    #         for line in infile:
    #             if "Cell" in line and "V" in line:
    #                 break
    #
    #         for line in infile:
    #             # Last line in cell data is repetition of header line
    #             if "Cell" in line and "V" in line:
    #                 break
    #
    #             # Cell number is a string (has key sometimes)
    #             items = line.strip().split()
    #             cell_no = items[0]
    #             params = [float(item) for item in items[1:]]
    #
    #             if len(items) == 10 and cell_no == "0":
    #                 # This is the start cell, only there to provide a starting aperture
    #                 if len(self._cells) == 0 and not ignore_rms:
    #                     # We use this only if there are no previous cells in the pyRFQ
    #                     # Else we ignore it...
    #                     self._cells.append(PyRFQCell(cell_type="STA",
    #                                                  aperture=params[6] * 0.01,
    #                                                  modulation=params[7],
    #                                                  length=0.0,
    #                                                  flip_z=False,
    #                                                  shift_cell_no=False,
    #                                                  prev_cell=None,
    #                                                  next_cell=None))
    #
    #                 continue
    #
    #             # For now we ignore "special" cells and add them manually
    #             if "T" in cell_no or "M" in cell_no or "F" in cell_no:
    #                 print("Ignored cell {}".format(cell_no))
    #                 continue
    #
    #             if params[7] == 1.0:
    #                 cell_type = "RMS"
    #                 if ignore_rms:
    #                     print("Ignored cell {}".format(cell_no))
    #                     continue
    #             else:
    #                 cell_type = "NCS"
    #
    #             if len(self._cells) > 0:
    #                 pc = self._cells[-1]
    #             else:
    #                 pc = None
    #
    #             self._cells.append(PyRFQCell(cell_type=cell_type,
    #                                          aperture=params[6] * 0.01,
    #                                          modulation=params[7],
    #                                          length=params[9] * 0.01,
    #                                          flip_z=False,
    #                                          shift_cell_no=False,
    #                                          prev_cell=pc,
    #                                          next_cell=None))
    #             if len(self._cells) > 1:
    #                 self._cells[-2].set_next_cell(self._cells[-1])
    #
    #     self._cell_nos = range(len(self._cells))
    #     self._length = np.sum([cell.length for cell in self._cells])

    # def generate_vanes(self):
    #
    #     assert len(self._cells) > 0, "No cells have been added, no vanes can be generated."
    #
    #     # There are four vanes (rods) in the RFQ
    #     # x = horizontal, y = vertical, with p, m denoting positive and negative axis directions
    #     # for vane_type in ["yp", "ym"]:
    #     for vane_type in ["yp"]:
    #         self._vanes.append(PyRFQVane(vane_type=vane_type,
    #                                      cells=self._cells,
    #                                      voltage=self._voltage + self._variables_bempp["pot_shift"],
    #                                      debug=self._debug))
    #
    #     # for vane_type in ["xp", "xm"]:
    #     for vane_type in ["xp"]:
    #         self._vanes.append(PyRFQVane(vane_type=vane_type,
    #                                      cells=self._cells,
    #                                      voltage=-self._voltage + self._variables_bempp["pot_shift"],
    #                                      debug=self._debug))
    #
    #     # Generate the two vanes in parallel:
    #     p = Pool()
    #     self._vanes = p.map(self.generate_vanes_worker, self._vanes)
    #
    #     return 0
    #
    # def generate_vanes_worker(self, vane):
    #
    #     dx_h = self._variables_bempp["grid_res"]  # TODO: Is there a reason to set them to different values?
    #
    #     vane.calculate_profile(fudge=True)
    #     vane.generate_gmsh_str(dx=dx_h, h=dx_h,
    #                            symmetry=False, mirror=True)
    #
    #     return vane
    #
    # def plot_vane_profile(self):
    #
    #     assert len(self._vanes) != 0, "No vanes calculated yet!"
    #
    #     _fig, _ax = plt.subplots()
    #
    #     for vane in self._vanes:
    #         if vane.vane_type == "xp":
    #             z, x = vane.get_profile(nz=10000)
    #             _ax.plot(z, x, color=colors[0], label="x-profile")
    #             print("X Vane starting point", z[0], x[0])
    #         if vane.vane_type == "yp":
    #             z, y = vane.get_profile(nz=10000)
    #             _ax.plot(z, -y, color=colors[1], label="y-profile")
    #             print("Y Vane starting point", z[0], y[0])
    #
    #     plt.xlabel("z (m)")
    #     plt.ylabel("x/y (m)")
    #
    #     plt.legend(loc=1)
    #
    #     plt.show()
    #
    # def print_cells(self):
    #     for number, cell in enumerate(self._cells):
    #         print("RFQ Cell {}: ".format(number + 1), cell)
    #
    #     return 0

    def install(self):
        # Parameters: None
        # Returns: None
        # Installs the field and conductors into Warp

        self.install_field()

        installconductor(self._conductors)
        scraper = ParticleScraper(self._conductors)


    def setup(self):
        # Parameters: None
        # Returns: None
        # Evaluates user parameters to ensure all dependencies are provided
        # Calculates and/or loads the field into the field class

        if not self.filename:
            print("Please provide a field file. Exiting")
            exit(1)
        if not self.rf_freq:
            print("The RF frequency (rf_freq) must be specified. Exiting")
            exit(1)

        if self.simple_vanes and not self.vane_from_profile:
            if not (self.vane_radius and self.vane_distance and self.vane_sta is not None and self.vane_end is not None):
                print("Please specify vane radius, distance, start, and end for the simple rod structure. Exiting.")
                exit(1)

        self.create_vanes()

        self._field.load_field_from_file(self.filename)


    def install_field(self):
        # import_field
        # Parameters: none
        # Returns: none
        # Loads the appropriate field into Warp simulation.

        def fieldscaling(time):
            val = np.cos(time * 2.0 * np.pi * self.rf_freq)
            # self._ray.append(val)
            return val * self.field_scaling_factor

        egrd = addnewegrddataset(ex=self._field._ex,
                                 ey=self._field._ey,
                                 ez=self._field._ez,
                                 dx=self._field._dx,
                                 dy=self._field._dy,
                                 zlength=self._field._z_length)

        # installs the field with the scaling function fieldscaling
        addnewegrd(id=egrd, zs=0,
                   xs=self._field._xmin, ys=self._field._ymin,
                   ze=self._field._z_length, func=fieldscaling)

    def create_vanes(self):
        # create_vanes
        # Parameters: None
        # Returns: None
        # Creates the conducting objects in the warp simulation.
        # Vanes and outer tube.

        # Variations of the tank:
        if self.tank_from_data:
            z = self.tank_data[:, 1]
            r = self.tank_data[:, 0]
            tank = ZSrfrvOut(zdata=z, rofzdata=r)
        elif self.tank_id and self.tank_sta and self.tank_end:
            tank = ZCylinderOut(0.5 * self.tank_id, self.tank_end - self.tank_sta,
                                zcent=0.5 * (self.tank_end + self.tank_sta))
            if self.endplates and self.endplates_thickness and self.entrance_plate_id and self.exit_plate_id:
                tank += ZCylinderOut(0.5 * self.entrance_plate_id, self.endplates_thickness,
                                     zcent=self.tank_sta - 0.5 * self.endplates_thickness)
                tank += ZCylinderOut(0.5 * self.exit_plate_id, self.endplates_thickness,
                                     zcent=self.tank_end + 0.5 * self.endplates_thickness)
        else:
            tank = ZCylinderOut(self.sim_radius, self.sim_end - self.sim_sta,
                                zcent=0.5 * (self.sim_end - self.sim_sta))

        all_conds = tank

        # Variations of the vanes/rods
        if self.vane_from_profile:
            print("Vane generation from profile not yet implemented. Exiting.")
            exit(1)
        elif self.simple_vanes and self.vane_sta and self.vane_end and self.vane_distance and self.vane_radius:
            length = self.vane_end - self.vane_sta
            zcent = self.vane_sta + 0.5 * length
            rod1 = ZCylinder(self.vane_radius, length, zcent=zcent, xcent=self.vane_distance)
            rod2 = ZCylinder(self.vane_radius, length, zcent=zcent, xcent=-self.vane_distance)
            rod3 = ZCylinder(self.vane_radius, length, zcent=zcent, ycent=self.vane_distance)
            rod4 = ZCylinder(self.vane_radius, length, zcent=zcent, ycent=-self.vane_distance)

            all_conds += rod1 + rod2 + rod3 + rod4

        self._conductors = all_conds

    def plot_efield(self):
        # Plots the e field along the z axis
        # Parameters: None
        # Returns: None

        plotegrd(component="z", iy=self._field._ny, ix=self._field._nx)
        fma()

        plotegrd(component="x", ix=self._field._nx, iy=self._field._ny)
        fma()

        plotegrd(component="y", iy=self._field._ny, ix=self._field._nx)
        fma()


if __name__ == "__main__":

    myrfq = PyRFQ(voltage=22000.0, debug=True)
