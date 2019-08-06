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
    import bempp.api
    from bempp.api.shapes.shapes import __generate_grid_from_geo_string as generate_from_string
except ImportError:
    print("Couldn't import BEMPP, no meshing or BEM field calculation will be possible.")
    bempp = None
    generate_from_string = None

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


# class FieldGenerator(object):
#     def __init__(self, resolution=0.001, xy_limits=None, ignore_rms=False,
#                  twoterm=True, eightterm=True):
#
#         self._colors = MyColors()
#
#         self._filename = None
#         self._parameters = []
#         self._nrms = None
#         self._max_cells = 0
#         self._plot_tcs_vanes = False
#
#         self._voltage = 50.0e3  # (V)
#         self._frequency = 32.8e6  # (Hz)
#         self._a_init = 0.038802  # (m)
#         self._resolution = resolution
#         self._bempp_mesh_size = resolution
#         self._xy_limits = xy_limits
#         self._total_length = 0.0  # (m)
#
#         self._nx = int((self._xy_limits[1] - self._xy_limits[0]) / self._resolution) + 1
#         self._ny = int((self._xy_limits[3] - self._xy_limits[2]) / self._resolution) + 1
#
#         self._mesh_x = None
#         self._mesh_y = None
#         self._mesh_z = None
#
#         self._z_linear = None
#
#         self._pot = None
#
#         self._ex = None
#         self._ey = None
#         self._ez = None
#
#         self._ex2 = None
#         self._ey2 = None
#         self._ez2 = None
#
#         self._calculate_vane_profile = False
#         self._vane_profile_x = None
#         self._vane_profile_y = None
#
#         self._ignore_rms = ignore_rms
#
#         self._vanes = []
#
#         self._cell_dtype = [("cell type", '|U16'),
#                             ("flip_z", bool),
#                             ("shift_cell_no", bool),
#                             ("cell no", int),
#                             ("energy", float),
#                             ("phase", float),
#                             ("aperture", float),
#                             ("modulation", float),
#                             ("focusing factor", float),
#                             ("cell length", float),
#                             ("cumulative length", float)]
#
#     def add_cell(self,
#                  cell_type,
#                  aperture,
#                  modulation,
#                  length,
#                  flip_z=False,
#                  shift_cell_no=False):
#
#         assert cell_type in ["STA", "RMS", "NCS", "TCS", "DCS"], "cell_type must be one of RMS, NCS, TCS, DCS!"
#
#         if (len(self._parameters) == 0):
#             cell_no = 1
#             cumulative_length = length
#         else:
#             cell_no = self._parameters[-1]["cell no"] + 1
#             cumulative_length = self._parameters[-1]["cumulative length"] + length
#
#         data = tuple([cell_type, flip_z, shift_cell_no, cell_no, 0.0, 0.0,
#                       aperture, modulation, 0.0, length, cumulative_length])
#
#         self._parameters = np.append(self._parameters, [np.array(data, dtype=self._cell_dtype)], axis=0)
#
#         return 0
#
#     def load_parameters_from_file(self, filename=None):
#         """
#         Load the cell parameters.
#         :param filename:
#         :return:
#         """
#
#         if filename is not None:
#             self._filename = filename
#
#         if self._filename is None:
#             # TODO: File Dialog!
#             print("No filename specified for parameter file. Closing.")
#             exit(1)
#
#         with open(filename, "r") as infile:
#             if "Parmteqm" in infile.readline():
#                 self.load_from_parmteq(filename)
#             else:
#                 self.load_from_vecc(filename)
#
#     def load_from_vecc(self, filename=None):
#         data = []
#
#         # noinspection PyTypeChecker
#         with open(self._filename, "r") as infile:
#
#             for line in infile:
#                 # noinspection PyTypeChecker
#                 data.append(tuple(["NCS", False, False]) + tuple([float(item) for item in line.strip().split()]))
#
#         self._parameters = np.array(data, dtype=self._cell_dtype)
#
#
#         # test = [x * y for x, y in zip(self._parameters["aperture"], self._parameters['modulation'])]
#         # test = [x + y for x, y in zip(test, self._parameters["aperture"])]
#         # test = [x / 2 for x in test]
#         # print(np.mean(test))
#
#         if self._nrms is None:
#             self._nrms = len(np.where(self._parameters["modulation"] == 1.0)[0])
#             print("I found {} cells with modulation 1 in the file..."
#                   "assuming this is the entrance Radial Matching Section (RMS)."
#                   " If this is incorrect, plase specify manually.".format(self._nrms))
#
#     def load_from_parmteq(self, filename=None):
#
#         with open(filename, "r") as infile:
#             data = []
#             version = infile.readline().strip().split()[1].split(",")[0]
#
#             for line in infile:
#                 if "Cell" in line and "V" in line:
#                     break
#
#             for line in infile:
#                 if "Cell" in line and "V" in line:
#                     break
#
#                 items = line.strip().split()
#                 cell_no = items[0]
#                 params = [float(item) for item in items[1:]]
#
#                 if len(items) == 10 and cell_no == "0":
#                     if len(self._parameters) == 0:
#                         self._parameters.append(np.array(("STA", False, False, 1, 0.0, 0.0,
#                                                      params[6] * 0.01,
#                                                      params[7], 0.0, 0.0, 0.0), dtype=self._cell_dtype))
#
#
#                     continue
#
#                 if "T" in cell_no or "M" in cell_no or "F" in cell_no:
#                     print("Ignored cell {}".format(cell_no))
#                     continue
#
#
#                 if params[7] == 1.0:
#                     cell_type = "RMS"
#                     if (self._ignore_rms):
#                         print("Ignored RMS cell!")
#                         continue
#
#                 else:
#                     cell_type = "NCS"
#
#                 self.add_cell(cell_type=cell_type,
#                               aperture=params[6] * 0.01,
#                               modulation=params[7],
#                               length=params[9] * 0.01,
#                               flip_z=False,
#                               shift_cell_no=False)
#
#         if self._nrms is None:
#             self._nrms = len(np.where(self._parameters["modulation"] == 1.0)[0])
#
#
#         return 0
#
#
#     def save_field_to_file(self, filename):
#
#         with open(filename, 'w') as outfile:
#             outfile.write("mesh_x, mesh_y, mesh_z, ex, ey, ez\n")
#             for x, y, z, ex, ey, ez in zip(self._mesh_x.flatten(), self._mesh_y.flatten(), self._mesh_z.flatten(),
#                                            self._ex.flatten(), self._ey.flatten(), self._ez.flatten()):
#                 outfile.write("{:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e}\n".format(x, y, z, ex, ey, ez))
#
#     def calculate_pot_rms(self, idx, rms_a):
#
#         x = self._mesh_x[idx]
#         y = self._mesh_y[idx]
#         z = self._mesh_z[idx]
#
#         self._pot[idx] = 0.5 * self._voltage * (x ** 2.0 - y ** 2.0) / rms_a(z) ** 2.0
#
#     def calculate_pot_drift(self, idx, aperture):
#         x = self._mesh_x[idx]
#         y = self._mesh_y[idx]
#         z = self._mesh_z[idx]
#
#         self._pot[idx] = 0.5 * self._voltage * (x ** 2.0 - y ** 2.0) / aperture ** 2.0
#
#         if self._calculate_vane_profile:
#             idx = np.where((np.min(z) <= self._z_linear) & (self._z_linear <= np.max(z)))
#
#             self._vane_profile_x[idx] = aperture
#             self._vane_profile_y[idx] = aperture
#
#         return 0
#
#     def calculate_profiles_2term(self, idx, cell_number):
#
#         cell_parameters = self._parameters[cell_number]
#         cell_start = cell_parameters["cumulative length"] - cell_parameters["cell length"]
#         z = self._mesh_z[idx] - cell_start  # z has to be adjusted such that the cell starts at 0.0
#
#         m = cell_parameters["modulation"]
#         a = cell_parameters["aperture"]
#
#         if 0 < cell_number:
#             ma_fudge_begin = 0.5 * (1.0 + self._parameters["aperture"][cell_number - 1] *
#                                     self._parameters["modulation"][cell_number - 1] / m / a)
#             a_fudge_begin = 0.5 * (1.0 + self._parameters["aperture"][cell_number - 1] / a)
#         else:
#             a_fudge_begin = ma_fudge_begin = 1.0
#
#         if (cell_number + 1) == len(self._parameters):
#             a_fudge_end = ma_fudge_end = 1.0
#         elif cell_number < len(self._parameters):
#             ma_fudge_end = 0.5 * (
#                 1.0 + self._parameters["aperture"][cell_number + 1] * self._parameters["modulation"][
#                     cell_number + 1] / m / a)
#             a_fudge_end = 0.5 * (1.0 + self._parameters["aperture"][cell_number + 1] / a)
#         else:
#             a_fudge_end = ma_fudge_end = 1.0
#
#         a_fudge = interp1d([0.0, cell_parameters["cell length"]], [a_fudge_begin, a_fudge_end])
#         ma_fudge = interp1d([0.0, cell_parameters["cell length"]], [ma_fudge_begin, ma_fudge_end])
#
#         kp = np.pi / cell_parameters["cell length"]
#         # mp = cell_parameters["modulation"]
#         # ap = cell_parameters["aperture"]
#
#         # denom = mp ** 2.0 * bessel1(0, kp * ap) + bessel1(0, mp * kp * ap)
#         # a10 = (mp ** 2.0 - 1.0) / denom
#         # r0 = ap / np.sqrt(1.0 - (mp ** 2.0 * bessel1(0, kp * ap) - bessel1(0, kp * ap)) / denom)
#
#         sign = (-1.0) ** (cell_parameters["cell no"] + 1)
#
#         idx2 = np.where((np.min(z) + cell_start <= self._z_linear) & (self._z_linear <= np.max(z) + cell_start))
#
#         def ap(zz):
#             return a * a_fudge(zz)
#
#         def mp(zz):
#             return m * ma_fudge(zz) / a_fudge(zz)
#
#         def a10(zz):
#             return (mp(zz) ** 2.0 - 1.0) / (mp(zz) ** 2.0 * bessel1(0, kp * ap(zz)) + bessel1(0, mp(zz) * kp * ap(zz)))
#
#         def r0(zz):
#             return ap(zz) / np.sqrt(1.0 - (mp(zz) ** 2.0 * bessel1(0, kp * ap(zz)) - bessel1(0, kp * ap(zz))) /
#                                     (mp(zz) ** 2.0 * bessel1(0, kp * ap(zz)) + bessel1(0, mp(zz) * kp * ap(zz))))
#
#         def vane_x(xx):
#             return + sign * (xx / r0(_z)) ** 2.0 + a10(_z) * bessel1(0.0, kp * xx) * np.cos(kp * _z) - sign
#
#         def vane_y(yy):
#             return - sign * (yy / r0(_z)) ** 2.0 + a10(_z) * bessel1(0.0, kp * yy) * np.cos(kp * _z) + sign
#
#         _vane_x = []
#         _vane_y = []
#
#         for _z in self._z_linear[idx2]:
#             _z -= cell_start
#             # noinspection PyTypeChecker
#             _vane_x.append(root(vane_x, ap(_z)).x[0])
#             # noinspection PyTypeChecker
#             _vane_y.append(root(vane_y, ap(_z)).x[0])
#
#         self._vane_profile_x[idx2] = _vane_x
#         self._vane_profile_y[idx2] = _vane_y
#
#         x = self._mesh_x[idx]
#         y = self._mesh_y[idx]
#         r = np.sqrt(x ** 2.0 + y ** 2.0)
#
#         self._pot[idx] = 0.5 * self._voltage * ((x ** 2.0 - y ** 2.0) / r0(z) ** 2.0
#                                                 + sign * a10(z) * bessel1(0, kp * r) * np.cos(kp * z))
#
#         return 0
#
#     def calculate_pot_2term(self, idx, cell_parameters):
#
#         x = self._mesh_x[idx]
#         y = self._mesh_y[idx]
#         cell_start = np.min(self._mesh_z[idx])
#         z = self._mesh_z[idx] - cell_start  # z has to go from 0 to cell_length
#
#         kp = np.pi / cell_parameters["cell length"]
#         mp = cell_parameters["modulation"]
#         ap = cell_parameters["aperture"]
#
#         denom = mp ** 2.0 * bessel1(0, kp * ap) + bessel1(0, mp * kp * ap)
#         a10 = (mp ** 2.0 - 1.0) / denom
#         r0 = ap / np.sqrt(1.0 - (mp ** 2.0 * bessel1(0, kp * ap) - bessel1(0, kp * ap)) / denom)
#         r = np.sqrt(x ** 2.0 + y ** 2.0)
#
#         sign = (-1.0) ** (cell_parameters["cell no"] + 1)
#
#         self._pot[idx] = 0.5 * self._voltage * ((x ** 2.0 - y ** 2.0) / r0 ** 2.0
#                                                 + sign * a10 * bessel1(0, kp * r) * np.cos(kp * z))
#
#         return 0
#
#     def calculate_e_2term(self, idx, cell_parameters):
#         x = self._mesh_x[idx]
#         y = self._mesh_y[idx]
#         z = self._mesh_z[idx] - np.min(self._mesh_z[idx])  # z has to go from 0 to cell_length
#
#         kp = np.pi / cell_parameters["cell length"]
#         mp = cell_parameters["modulation"]
#         ap = cell_parameters["aperture"]
#
#         denom = mp ** 2.0 * bessel1(0, kp * ap) + bessel1(0, mp * kp * ap)
#         a10 = (mp ** 2.0 - 1.0) / denom
#         xim = 1.0 - a10 * bessel1(0, kp * ap)
#         sign = (-1.0) ** (cell_parameters["cell no"])
#
#         r = np.sqrt(x ** 2.0 + y ** 2.0)
#
#         self._ex2[idx] = x * 0.5 * self._voltage * (+ 2.0 * xim / ap ** 2.0
#                                                     - sign * kp / r * a10 * bessel1(1.0, kp * r) * np.cos(kp * z))
#
#         self._ey2[idx] = y * 0.5 * self._voltage * (- 2.0 * xim / ap ** 2.0
#                                                     - sign * kp / r * a10 * bessel1(1.0, kp * r) * np.cos(kp * z))
#
#         self._ez2[idx] = 0.5 * self._voltage * sign * kp * a10 * bessel1(0.0, kp * r) * np.sin(kp * z)
#
#     @staticmethod
#     def calculate_transition_cell_length(cell_parameters):
#
#         k = np.pi / np.sqrt(3.0) / cell_parameters["cell length"]
#         m = cell_parameters["modulation"]
#         a = cell_parameters["aperture"]
#         r0 = 0.5 * (a + m * a)
#
#         def eta(kk):
#             return bessel1(0.0, kk * r0) / (3.0 * bessel1(0.0, 3.0 * kk * r0))
#
#         def func(kk):
#             return (bessel1(0.0, kk * m * a) + eta(kk) * bessel1(0.0, 3.0 * kk * m * a)) / \
#                    (bessel1(0.0, kk * a) + eta(kk) * bessel1(0.0, 3.0 * kk * a)) \
#                    + ((m * a / r0) ** 2.0 - 1.0) / ((a / r0) ** 2.0 - 1.0)
#
#         k = root(func, k).x[0]
#         tcs_length = np.pi / 2.0 / k
#         print("Transition cell has length {} which is {} * cell length, ".format(tcs_length,
#                                                                                  tcs_length / cell_parameters[
#                                                                                      "cell length"]), end="")
#         print("the remainder will be filled with a drift.")
#
#         assert tcs_length <= cell_parameters["cell length"], "Numerical determination of transition cell length " \
#                                                              "yielded value larger than cell length parameter!"
#
#         return np.pi / 2.0 / k
#
#     def calculate_pot_3term(self, idx, cell_parameters):
#
#         x = self._mesh_x[idx]
#         y = self._mesh_y[idx]
#         cell_start = np.min(self._mesh_z[idx])
#         z = self._mesh_z[idx] - cell_start  # z has to go from 0 to cell_length
#
#         k = np.pi / np.sqrt(3.0) / cell_parameters["cell length"]
#         m = cell_parameters["modulation"]
#         a = cell_parameters["aperture"]
#         r0 = 0.5 * (a + m * a)
#
#         print("Average radius of transition cell (a + ma) / 2 = {}".format(r0))
#
#         def eta(kk):
#             return bessel1(0.0, kk * r0) / (3.0 * bessel1(0.0, 3.0 * kk * r0))
#
#         def a10(kk):
#             return ((m * a / r0) ** 2.0 - 1.0) / (bessel1(0.0, kk * m * a) + eta(kk) * bessel1(0.0, 3.0 * kk * m * a))
#
#         def a30(kk):
#             return eta(kk) * a10(kk)
#
#         def func(kk):
#             return (bessel1(0.0, kk * m * a) + eta(kk) * bessel1(0.0, 3.0 * kk * m * a)) / \
#                    (bessel1(0.0, kk * a) + eta(kk) * bessel1(0.0, 3.0 * kk * a)) \
#                    + ((m * a / r0) ** 2.0 - 1.0) / ((a / r0) ** 2.0 - 1.0)
#
#         k = root(func, k).x[0]
#
#         if cell_parameters["shift_cell_no"]:
#             sign = (-1.0) ** (cell_parameters["cell no"] + 1)
#         else:
#             sign = (-1.0) ** (cell_parameters["cell no"])
#
#         r = np.sqrt(x ** 2.0 + y ** 2.0)
#
#         # print(np.cos(k * np.max(z)))
#         # print(np.cos(3.0 * k * np.max(z)))
#
#         self._pot[idx] = np.round(0.5 * self._voltage * (
#             + (x ** 2.0 - y ** 2.0) / r0 ** 2.0
#             - sign * a10(k) * bessel1(0.0, k * r) * np.cos(k * z)
#             - sign * a30(k) * bessel1(0.0, 3.0 * k * r) * np.cos(3.0 * k * z)
#         ), 5)
#
#         if self._plot_tcs_vanes or self._calculate_vane_profile:
#
#             idx = np.where((np.min(z) + cell_start <= self._z_linear) & (self._z_linear <= np.max(z) + cell_start))
#
#             def vane_x(xx):
#                 return - (xx / r0) ** 2.0 \
#                        + sign * a10(k) * bessel1(0.0, k * xx) * np.cos(k * _z) \
#                        + sign * a30(k) * bessel1(0.0, 3.0 * k * xx) * np.cos(3.0 * k * _z) + 1.0
#
#             def vane_y(yy):
#                 return + (yy / r0) ** 2.0 \
#                        + sign * a10(k) * bessel1(0.0, k * yy) * np.cos(k * _z) \
#                        + sign * a30(k) * bessel1(0.0, 3.0 * k * yy) * np.cos(3.0 * k * _z) - 1.0
#
#             def pot_on_axis():
#                 return 0.5 * self._voltage * (
#                     + a10(k) * (np.cos(k * _z))
#                     + a30(k) * (np.cos(3.0 * k * _z)))
#
#             _vane_x = []
#             _vane_y = []
#             _z_plot = []
#             _pot_plot = []
#
#             for _z in self._z_linear[idx]:
#                 _z -= cell_start
#                 # noinspection PyTypeChecker
#                 _vane_x.append(root(vane_x, r0).x[0])
#                 # noinspection PyTypeChecker
#                 _vane_y.append(root(vane_y, r0).x[0])
#                 _z_plot.append(_z)
#                 _pot_plot.append(pot_on_axis())
#
#             if self._plot_tcs_vanes:
#                 plt.plot(_z_plot, _vane_x, color=self._colors[0])
#                 plt.plot(_z_plot, _vane_y, color=self._colors[2])
#                 plt.scatter([0.0, 0.0, np.max(np.unique(z))], [a, m * a, r0], s=16, color=self._colors[1])
#
#                 plt.twinx()
#
#                 plt.plot(_z_plot, _pot_plot, color=self._colors[3])
#
#                 plt.show()
#
#             if self._calculate_vane_profile:
#                 if cell_parameters["flip_z"]:
#                     self._vane_profile_x[idx] = _vane_x[::-1]
#                     self._vane_profile_y[idx] = _vane_y[::-1]
#                 else:
#                     self._vane_profile_x[idx] = _vane_x
#                     self._vane_profile_y[idx] = _vane_y
#
#         return 0
#
#     @staticmethod
#     def pot_8term(coeff, _r, _t, _z, k):
#
#         a01, a03, a10, a12, a21, a23, a30, a32 = coeff
#
#         return + a01 * _r ** 2.0 * np.cos(2.0 * _t) \
#                + a03 * _r ** 6.0 * np.cos(6.0 * _t) \
#                + a10 * bessel1(0, k * _r) * np.cos(k * _z) \
#                + a12 * bessel1(4, k * _r) * np.cos(4.0 * _t) * np.cos(k * _z) \
#                + a21 * bessel1(2, 2.0 * k * _r) * np.cos(2.0 * _t) * np.cos(2.0 * k * _z) \
#                + a23 * bessel1(6, 2.0 * k * _r) * np.cos(6.0 * _t) * np.cos(2.0 * k * _z) \
#                + a30 * bessel1(0, 3.0 * k * _r) * np.cos(3.0 * k * _z) \
#                + a32 * bessel1(4, 3.0 * k * _r) * np.cos(4.0 * _t) * np.cos(3.0 * k * _z)
#
#     def calculate_pot_8term(self, idx, cell_parameters):
#
#         # --- Calculate the eight coefficients using a least squares fit: --- #
#         # First, generate a matrix of surface points
#         # Assume that cell starts with x = a and y = ma at z = 0
#         # lets start with 5 mm radius half-circle at z=0
#         samples = 1000
#
#         vane_radius = 0.005  # (m)
#
#         vane_opening_angle = 45.0  # (deg)
#         vane_opening_angle *= np.pi / 180.0
#
#         xvane_angles = np.linspace(np.pi - vane_opening_angle,
#                                    np.pi + vane_opening_angle,
#                                    samples)
#         yvane_angles = np.linspace(1.5 * np.pi - vane_opening_angle,
#                                    1.5 * np.pi + vane_opening_angle,
#                                    samples)
#
#         xvane_x = vane_radius * np.cos(xvane_angles)
#         xvane_y = vane_radius * np.sin(xvane_angles)
#         yvane_x = vane_radius * np.cos(yvane_angles)
#         yvane_y = vane_radius * np.sin(yvane_angles)
#
#         if cell_parameters["cell no"] % 2.0 == 0.0:
#
#             ap1 = cell_parameters["aperture"] * cell_parameters["modulation"]
#             ap2 = 0.5 * (ap1 + cell_parameters["aperture"])
#             ap3 = cell_parameters["aperture"]
#
#         else:
#
#             ap1 = cell_parameters["aperture"]
#             ap2 = 0.5 * (cell_parameters["modulation"] * ap1 + ap1)
#             ap3 = cell_parameters["modulation"] * ap1
#
#         r1 = np.sqrt((xvane_x + ap1 + vane_radius) ** 2.0 + xvane_y ** 2.0)
#         r2 = np.sqrt((xvane_x + ap2 + vane_radius) ** 2.0 + xvane_y ** 2.0)
#         r3 = np.sqrt((xvane_x + ap3 + vane_radius) ** 2.0 + xvane_y ** 2.0)
#
#         r4 = np.sqrt(yvane_x ** 2.0 + (yvane_y + ap3 + vane_radius) ** 2.0)
#         r5 = np.sqrt(yvane_x ** 2.0 + (yvane_y + ap2 + vane_radius) ** 2.0)
#         r6 = np.sqrt(yvane_x ** 2.0 + (yvane_y + ap1 + vane_radius) ** 2.0)
#
#         z1 = np.zeros(samples)
#         z2 = np.ones(samples) * 0.5 * cell_parameters["cell length"]
#         z3 = np.ones(samples) * cell_parameters["cell length"]
#
#         t1 = np.arctan2(xvane_y, xvane_x + ap1 + vane_radius)
#         t2 = np.arctan2(xvane_y, xvane_x + ap2 + vane_radius)
#         t3 = np.arctan2(xvane_y, xvane_x + ap3 + vane_radius)
#
#         t4 = np.arctan2(yvane_y + ap3 + vane_radius, yvane_x)
#         t5 = np.arctan2(yvane_y + ap2 + vane_radius, yvane_x)
#         t6 = np.arctan2(yvane_y + ap1 + vane_radius, yvane_x)
#
#         # plt.plot(r1 * np.cos(t1), r1 * np.sin(t1), linestyle='dashed', color='red', label="x1")
#         # plt.plot(r2 * np.cos(t2), r2 * np.sin(t2), linestyle='dotted', color='red', label="x2")
#         # plt.plot(r3 * np.cos(t3), r3 * np.sin(t3), color='red', label="x3")
#         # plt.plot(r4 * np.cos(t4), r4 * np.sin(t4), linestyle='dashed', color='blue', label="y1")
#         # plt.plot(r5 * np.cos(t5), r5 * np.sin(t5), linestyle='dotted', color='blue', label="y2")
#         # plt.plot(r6 * np.cos(t6), r6 * np.sin(t6), color='blue', label="y3")
#         # plt.legend()
#         # plt.show()
#         # exit()
#
#         # Finalize arrays containing the surface points in cylinder coordinates
#         r = np.array([r1, r2, r3, r4, r5, r6]).flatten()
#         t = np.array([t1, t2, t3, t4, t5, t6]).flatten()
#         z = np.array([z1, z2, z3, z1, z2, z3]).flatten()
#
#         # Generate an array of "results" (along the surface, all potentials are V/2)
#         y1 = np.ones(3 * samples) * +0.5 * self._voltage
#         y2 = np.ones(3 * samples) * -0.5 * self._voltage
#
#         y = np.array([y1, y2]).flatten()
#
#         k = np.pi / cell_parameters["cell length"]
#
#         matrix = []
#         for _r, _t, _z in zip(r, t, z):
#             matrix.append([_r ** 2.0 * np.cos(2.0 * _t),
#                            _r ** 6.0 * np.cos(6.0 * _t),
#                            bessel1(0, k * _r) * np.cos(k * _z),
#                            bessel1(4, k * _r) * np.cos(4.0 * _t) * np.cos(k * _z),
#                            bessel1(2, 2.0 * k * _r) * np.cos(2.0 * _t) * np.cos(2.0 * k * _z),
#                            bessel1(6, 2.0 * k * _r) * np.cos(6.0 * _t) * np.cos(2.0 * k * _z),
#                            bessel1(0, 3.0 * k * _r) * np.cos(3.0 * k * _z),
#                            bessel1(4, 3.0 * k * _r) * np.cos(4.0 * _t) * np.cos(3.0 * k * _z),
#                            ])
#
#         matrix = np.array(matrix)
#
#         # print(matrix)
#
#         # Call the least squares function and get the 8 coefficients
#         coeff = np.linalg.lstsq(matrix, y)[0]
#
#         x = self._mesh_x[idx]
#         y = self._mesh_y[idx]
#         z = self._mesh_z[idx]
#
#         z_min = np.min(z)
#         # z_half = 0.5 * (z_min + np.max(z))
#
#         z -= z_min
#
#         self._pot[idx] = self.pot_8term(coeff, np.sqrt(x ** 2.0 + y ** 2.0), np.arctan2(y, x), z, k)
#
#         return 0
#
#     def calculate_e_from_pot(self, idx, flip_z, ttf=0.0, phase=0.0):
#
#         if flip_z:
#             pot = self._pot[:, :, idx[2][-1]:idx[2][0] - 1:-1]
#         else:
#             pot = self._pot[:, :, idx[2][0]:idx[2][-1] + 1]
#
#         tt_factor = -np.cos((2.0 * np.pi * self._frequency * ttf) + (phase * np.pi / 180.0))
#
#         a, b, c = np.gradient(pot, self._resolution, edge_order=2)
#
#         self._ex[idx] = tt_factor * a.flatten()
#         self._ey[idx] = tt_factor * b.flatten()
#         self._ez[idx] = tt_factor * c.flatten()
#
#         return 0
#
#     def calculate_ex_rms(self, idx, rms_a, transit_time=0.0, phase=0.0):
#
#         x = self._mesh_x[idx]
#         z = self._mesh_z[idx]
#
#         self._ex[idx] = -(self._voltage * x / rms_a(z) ** 2.0) * (
#             np.cos((2.0 * np.pi * self._frequency * transit_time) + (phase * np.pi / 180.0)))
#
#         return 0
#
#     def calculate_ey_rms(self, idx, rms_a, transit_time=0.0, phase=0.0):
#
#         y = self._mesh_y[idx]
#         z = self._mesh_z[idx]
#
#         self._ey[idx] = (self._voltage * y / rms_a(z) ** 2.0) * (
#             np.cos((2.0 * np.pi * self._frequency * transit_time) + (phase * np.pi / 180.0)))
#
#         return 0
#
#     def calculate_dcs(self, cell_parameters):
#
#         # Find all z values that are within the cell
#         cell_idx = np.where((self._mesh_z <= cell_parameters["cumulative length"]) &
#                             (self._mesh_z >= (cell_parameters["cumulative length"] -
#                                               cell_parameters["cell length"])))
#
#         print("Calculating Drift Cell # {}, ".format(cell_parameters["cell no"]), end="")
#         print("from z = {} m to {} m".format(cell_parameters["cumulative length"] -
#                                              cell_parameters["cell length"],
#                                              cell_parameters["cumulative length"]))
#
#         self.calculate_pot_drift(cell_idx, cell_parameters["aperture"])
#         self.calculate_e_from_pot(cell_idx, cell_parameters["flip_z"])
#
#         return 0
#
#     def calculate_ncs(self, cell_parameters, cell_number=None):
#         print("Calcncs cell num: {}   passed:  {}".format(cell_parameters["cell no"], cell_number))
#         # Find all z values that are within the cell
#         cell_idx = np.where((self._mesh_z <= cell_parameters["cumulative length"]) &
#                             (self._mesh_z >= (cell_parameters["cumulative length"] -
#                                               cell_parameters["cell length"])))
#
#         print("Calculating Normal Cell # {}, ".format(cell_parameters["cell no"]), end="")
#         print("from z = {} m to {} m".format(cell_parameters["cumulative length"] -
#                                              cell_parameters["cell length"],
#                                              cell_parameters["cumulative length"]))
#
#         self.calculate_profiles_2term(cell_idx, cell_number)
#         # self.calculate_pot_2term(cell_idx, cell_parameters)
#         # self.calculate_pot_8term(cell_idx, cell_parameters)
#         self.calculate_e_from_pot(cell_idx, cell_parameters["flip_z"])
#         # self.calculate_e_2term(cell_idx, cell_parameters)
#
#     def calculate_rms_in(self):
#
#         # b_match = self._parameters[self._nrms + 1]["focusing factor"]
#         rms_length = self._parameters[self._nrms - 1]["cumulative length"]
#
#         rms_idx = np.where(self._mesh_z <= rms_length)
#
#         print("Calculating Radial Matching Section from z = {} m to {} m".format(0.0, rms_length))
#
#         # The RMS has a smooth decrease in aperture size from a_init to the size of the first RFQ cell
#         # Also, the first datapoint (at z = 0.0) is not included in the file, so we add it here.
#         rms_a = interp1d(np.append(np.array([0.0]), self._parameters["cumulative length"][:self._nrms]),
#                          np.append(np.array([self._a_init]), self._parameters["aperture"][:self._nrms]),
#                          kind="cubic")
#
#         self.calculate_pot_rms(rms_idx, rms_a)
#         # self.calculate_e_from_pot(rms_idx)
#         self.calculate_ex_rms(rms_idx, rms_a, 0.0, 0.0)
#         self.calculate_ey_rms(rms_idx, rms_a, 0.0, 0.0)
#
#         if self._calculate_vane_profile:
#             idx = np.where(self._z_linear <= rms_length)
#
#             self._vane_profile_x[idx] = rms_a(self._z_linear[idx])
#             self._vane_profile_y[idx] = rms_a(self._z_linear[idx])
#
#     def calculate_tcs(self, cell_parameters):
#         print("Calculating Transition Cell # {}, ".format(cell_parameters["cell no"]), end="")
#         print("from z = {} m to {} m".format(cell_parameters["cumulative length"] -
#                                              cell_parameters["cell length"],
#                                              cell_parameters["cumulative length"]))
#
#         tcs_length = self.calculate_transition_cell_length(cell_parameters)
#         tcs_begin = cell_parameters["cumulative length"] - cell_parameters["cell length"]
#
#         if cell_parameters["flip_z"]:
#
#             cell_idx = np.where((self._mesh_z >= tcs_begin) &
#                                 (self._mesh_z <= cell_parameters["cumulative length"] - tcs_length))
#
#             r0 = 0.5 * (cell_parameters["aperture"] + cell_parameters["modulation"] * cell_parameters["aperture"])
#
#             self.calculate_pot_drift(cell_idx, r0)
#
#             cell_idx = np.where((self._mesh_z >= cell_parameters["cumulative length"] - tcs_length) &
#                                 (self._mesh_z <= cell_parameters["cumulative length"]))
#
#             self.calculate_pot_3term(cell_idx, cell_parameters)
#
#         else:
#
#             cell_idx = np.where((self._mesh_z >= tcs_begin) &
#                                 (self._mesh_z <= (tcs_begin + tcs_length)))
#
#             self.calculate_pot_3term(cell_idx, cell_parameters)
#
#             cell_idx = np.where((self._mesh_z >= (tcs_begin + tcs_length)) &
#                                 (self._mesh_z <= cell_parameters["cumulative length"]))
#
#             r0 = 0.5 * (cell_parameters["aperture"] + cell_parameters["modulation"] * cell_parameters["aperture"])
#
#             self.calculate_pot_drift(cell_idx, r0)
#
#         cell_idx = np.where((self._mesh_z >= (cell_parameters["cumulative length"] -
#                                               cell_parameters["cell length"])) &
#                             (self._mesh_z <= cell_parameters["cumulative length"]))
#
#         self.calculate_e_from_pot(cell_idx, cell_parameters["flip_z"])
#
#     def generate(self):
#
#         # Find total number of cells
#         self._max_cells = self._parameters[-1]["cell no"]
#
#         # Generating the mesh points for the full length RFQ
#         self.generate_mesh()
#
#         # the RFQ has several sections that are calculated differently:
#         # *) Radial matching sections (RMS)
#         # *) Normal cell section for gentle buncher + accelerating section (NCS)
#         # *) Transition cell sections (TCS)
#         # *) Drift cell sections (DCS)
#
#         # Omit RMS if there are no cells with modulation 1
#         if self._nrms > 0:
#             print("within the self nrms!")
#         # Loop over all remaining cells starting after RMS
#         for cn, cell_parameters in enumerate(self._parameters[self._nrms:]):
#             cn += self._nrms
#             # Calculate the potential and field according to cell type
#             if cell_parameters["cell type"] == "NCS":
#                 self.calculate_ncs(cell_parameters, cell_number=cn)
#
#             elif cell_parameters["cell type"] == "TCS":
#                 self.calculate_tcs(cell_parameters)
#
#             elif cell_parameters["cell type"] == "DCS":
#                 self.calculate_dcs(cell_parameters)
#
#         return 0
#
#     def generate_mesh(self):
#
#         x_values = np.linspace(self._xy_limits[0], self._xy_limits[1], self._nx)
#         y_values = np.linspace(self._xy_limits[2], self._xy_limits[3], self._ny)
#
#         total_z_length = self._parameters["cumulative length"][-1] + self._resolution
#
#         z_values = np.arange(0.0, total_z_length, self._resolution)
#
#         if z_values[-1] > self._parameters["cumulative length"][-1]:
#             z_values = z_values[:-1]
#
#         self._total_length = np.max(z_values)
#
#         self._mesh_x, self._mesh_y, self._mesh_z = meshgrid(x_values, y_values, z_values, indexing='ij')
#
#         self._z_linear = np.sort(np.unique(self._mesh_z))
#
#         self._pot = np.zeros(self._mesh_x.shape)
#
#         self._ex = np.zeros(self._mesh_x.shape)
#         self._ey = np.zeros(self._mesh_x.shape)
#         self._ez = np.zeros(self._mesh_x.shape)
#
#         if self._calculate_vane_profile:
#             self._vane_profile_x = np.zeros(self._z_linear.shape)
#             self._vane_profile_y = np.zeros(self._z_linear.shape)
#
#         # self._ex2 = np.zeros(self._mesh_x.shape)
#         # self._ey2 = np.zeros(self._mesh_x.shape)
#         # self._ez2 = np.zeros(self._mesh_x.shape)
#
#         return 0
#
#     def plot_e_xy(self, z=0.0):
#
#         plot_idx = np.where(self._mesh_z == z)
#
#         plt.quiver(self._mesh_x[plot_idx].flatten(),
#                    self._mesh_y[plot_idx].flatten(),
#                    self._ex[plot_idx].flatten(),
#                    self._ey[plot_idx].flatten())
#
#         plt.xlabel("x (m)")
#         plt.ylabel("y (m)")
#
#         plt.show()
#
#     def plot_pot_xy(self, z=0.0):
#
#         # Find nearest mesh value to desired z
#         zvals = self._mesh_z.flatten()
#         z_close = zvals[np.abs(zvals - z).argmin()]
#
#         z_slice_idx = np.where(self._mesh_z[0, 0, :] == z_close)[0][0]
#
#         plt.contour(self._mesh_x[:, :, z_slice_idx],
#                     self._mesh_y[:, :, z_slice_idx],
#                     self._pot[:, :, z_slice_idx],
#                     40,
#                     cmap=plt.get_cmap("jet"))
#
#         plt.colorbar()
#
#         plt.xlabel("x (m)")
#         plt.ylabel("y (m)")
#         plt.gca().set_aspect('equal')
#
#         plt.show()
#
#     def plot_ex_of_z(self, x=0.0):
#
#         # Find nearest mesh value to desired x
#         xvals = self._mesh_x.flatten()
#         x_close = xvals[np.abs(xvals - x).argmin()]
#
#         plot_idx = np.where((self._mesh_x == x_close) & (self._mesh_y == 0.0))
#
#         plt.plot(self._mesh_z[plot_idx], self._ex[plot_idx])
#
#         plt.xlabel("z (m)")
#         plt.ylabel("Ex (V/m)")
#         plt.title("Ex(z) at x = {} m, y = 0 m".format(x_close))
#
#         plt.show()
#
#     def plot_ez_of_z(self):
#
#         plot_idx = np.where((self._mesh_x == 0.0) & (self._mesh_y == 0.0))
#
#         plt.plot(self._mesh_z[plot_idx], self._ez[plot_idx] / 100.0)
#
#         plt.xlim(0.0, 0.12)
#
#         plt.xlabel("z (m)")
#         plt.ylabel("Ez (V/m)")
#
#         plt.show()
#
#     def plot_pot_of_z(self, opera_data_fn=None):
#
#         plot_idx = np.where((self._mesh_x == 0.0) & (self._mesh_y == 0.0))
#
#         plt.plot(self._mesh_z[plot_idx], self._pot[plot_idx])
#
#         # plt.xlim(0.0, 0.12)
#
#         plt.xlabel("z (m)")
#         plt.ylabel("Potential (V)")
#         plt.title("U(z) at x = 0 m, y = 0 m")
#
#         if opera_data_fn is not None:
#             opera_data = []
#
#             # noinspection PyTypeChecker
#             with open(opera_data_fn, "r") as infile:
#
#                 for k in range(5):
#                     infile.readline()
#
#                 for line in infile:
#                     # noinspection PyTypeChecker
#                     opera_data.append(tuple([float(item) for item in line.strip().split()]))
#
#             mydtype = [("z", float),
#                        ("v", float),
#                        ("ez", float)]
#
#             opera_data = np.array(opera_data, dtype=mydtype)
#
#             plt.plot(0.01 * opera_data["z"] + 1.2070849776907893, -opera_data["v"])
#
#         plt.show()
#
#     def set_bempp_mesh_size(self, mesh_size):
#         print("voltage {}".format(self._voltage))
#         self._bempp_mesh_size = mesh_size
#
#     def set_calculate_vane_profile(self, calculate_vane_profile=True):
#         self._calculate_vane_profile = calculate_vane_profile
#
#     def set_plot_tcs_vanes(self, plot_tcs_vanes=True):
#         self._plot_tcs_vanes = plot_tcs_vanes
#
#     def plot_combo(self, opera_data_fn=None):
#
#         # plt.rc('text', usetex=True)
#         # plt.rc('font', family='serif')
#         # plt.rc('font', size=18)
#
#         # Find nearest mesh value to desired x
#         x_plot = 0.05
#         xvals = self._mesh_x.flatten()
#         x_close = xvals[np.abs(xvals - x_plot).argmin()]
#
#         plot_idx = np.where((self._mesh_x == x_close) & (self._mesh_y == 0.0))
#         plt.plot(self._mesh_z[plot_idx], self._ex[plot_idx] / 10, color=self._colors[2], label="$\mathrm{E}_\mathrm{x}$")
#         # plt.plot(self._mesh_z[plot_idx], self._ex2[plot_idx], linestyle='dashed', color='red')
#
#         # Find nearest mesh value to desired y
#         y_plot = 0.05
#         yvals = self._mesh_y.flatten()
#         y_close = yvals[np.abs(yvals - y_plot).argmin()]
#
#         plot_idx = np.where((self._mesh_y == y_close) & (self._mesh_x == 0.0))
#         plt.plot(self._mesh_z[plot_idx], self._ey[plot_idx] / 10, color=self._colors[1], label="$\mathrm{E}_\mathrm{y}$")
#         # plt.plot(self._mesh_z[plot_idx], self._ey2[plot_idx], linestyle='dashed', color='green')
#
#         plot_idx = np.where((self._mesh_x == 0.0) & (self._mesh_y == 0.0))
#         plt.plot(self._mesh_z[plot_idx], self._ez[plot_idx], color=self._colors[0], label="$\mathrm{E}_\mathrm{z}$")
#         # plt.plot(self._mesh_z[plot_idx], self._ez2[plot_idx], linestyle='dashed', color='blue')
#
#         plt.xlabel("Z (m)")
#         plt.ylabel("Electric Field (V/m)")
#         plt.xlim(0.0, None)
#         plt.ylim(-800000.0, 800000.0)
#
#         if opera_data_fn is not None:
#             opera_data = []
#
#             # noinspection PyTypeChecker
#             with open(opera_data_fn, "r") as infile:
#
#                 for k in range(5):
#                     infile.readline()
#
#                 for line in infile:
#                     # noinspection PyTypeChecker
#                     opera_data.append(tuple([float(item) for item in line.strip().split()]))
#
#             mydtype = [("z", float),
#                        ("v", float),
#                        ("ez", float)]
#
#             opera_data = np.array(opera_data, dtype=mydtype)
#
#             plt.plot(0.01 * opera_data["z"] + 1.2070849776907893, -opera_data["ez"] * 100.0, color=self._colors[3])
#
#         plt.legend(loc=2)
#
#         plt.show()
#
#     def plot_vane_profile(self):
#         print("==========\nPLOTTING VANE PROFILE\n==========")
#         if not self._calculate_vane_profile:
#             print("Vane profile was not calculated, use set_calculate_vane_profile() before generating.")
#
#             return 1
#
#         plt.plot(self._z_linear[:-1], self._vane_profile_x[:-1], color=self._colors[0])
#         plt.plot(self._z_linear[:-1], -self._vane_profile_y[:-1], color=self._colors[1])
#
#         interp_profile = interp1d(self._z_linear[:-1], self._vane_profile_x[:-1], kind='cubic')
#
#         z_plot = np.linspace(self._z_linear[1], self._z_linear[-2], 5000)
#
#         plt.plot(z_plot,
#                  interp_profile(z_plot),
#                  color=self._colors[2])
#
#         plt.xlim(min(self._z_linear), max(self._z_linear))
#         # plt.xlim(0.0, 0.12)
#         plt.ylim(-1.1 * max(self._vane_profile_x), 1.1 * max(self._vane_profile_x))
#
#         plt.show()
#
#         return 0
#
#     def mesh_vanes(self):
#
#         # We have already generated the vane profile, now we have to generate a mesh
#         # In BEMPP the mesh is generated from a set of vertices (given by arrays of z, y and z coordinates)
#         # and a set of elements that define how the vertices are connected.
#
#         # self._mesh_x = None
#         # self._mesh_y = None
#         # self._mesh_z = None
#         self._pot = None
#         self._ex = None
#         self._ey = None
#         self._ez = None
#
#         gc.collect()
#
#         stopwatch("start")
#
#         # If the user has not defined a separate bempp mesh size, the resolution will be used.
#         x_vane1 = PyRFQVane(vane_type="semi-circle",
#                        mesh_size=self._bempp_mesh_size,
#                        curvature=0.005,
#                        height=0.001,
#                        vane_z_data=self._z_linear,
#                        vane_y_data=self._vane_profile_x,
#                        rotation=90.0)
#
#         x_vane2 = PyRFQVane(vane_type="semi-circle",
#                        mesh_size=self._bempp_mesh_size,
#                        curvature=0.005,
#                        height=0.001,
#                        vane_z_data=self._z_linear,
#                        vane_y_data=self._vane_profile_x,
#                        rotation=-90.0)
#
#         y_vane1 = PyRFQVane(vane_type="semi-circle",
#                        mesh_size=self._bempp_mesh_size,
#                        curvature=0.005,
#                        height=0.001,
#                        vane_z_data=self._z_linear,
#                        vane_y_data=self._vane_profile_y,
#                        rotation=0.0)
#
#         y_vane2 = PyRFQVane(vane_type="semi-circle",
#                        mesh_size=self._bempp_mesh_size,
#                        curvature=0.005,
#                        height=0.001,
#                        vane_z_data=self._z_linear,
#                        vane_y_data=self._vane_profile_y,
#                        rotation=180.0)
#
#         x_vane1.generate_grid()
#         x_vane2.generate_grid()
#         y_vane1.generate_grid()
#         y_vane2.generate_grid()
#
#         complete_system = x_vane1 + x_vane2 + y_vane1 + y_vane2
#         # complete_system.plot_grid()
#
#         del x_vane1
#         del x_vane2
#         del y_vane1
#         del y_vane2
#
#         stopwatch('Vanes created')
#
#         # import bempp.api
#         #
#         # space = bempp.api.function_space(complete_system.get_grid(), "DP", 0)
#         # slp = bempp.api.operators.boundary.laplace.single_layer(space, space, space)
#         #
#         # def f(r, n, domain_index, result):
#         #     if abs(r[0]) > 0.011:
#         #         result[0] = 25000.0
#         #     else:
#         #         result[0] = -25000.0
#         #
#         # rhs = bempp.api.GridFunction(space, fun=f)
#         # sol, _ = bempp.api.linalg.gmres(slp, rhs)
#         #
#         # stopwatch('BEM problem solved')
#         #
#         # # sol.plot()
#         #
#         # # nvals = 251
#         # # z_vals = np.linspace(0.0, 0.250, nvals)
#         # # nvals = len(self._z_linear)
#         # # points = np.vstack([np.zeros(nvals), np.zeros(nvals), self._z_linear])
#         # points = np.vstack([self._mesh_x.flatten(),
#         #                     self._mesh_y.flatten(),
#         #                     self._mesh_z.flatten()])
#         #
#         # nearfield = bempp.api.operators.potential.laplace.single_layer(space, points)
#         # pot_discrete = nearfield * sol
#         #
#         # self._pot = pot_discrete[0].reshape(self._mesh_x.shape)
#         #
#         # stopwatch('potential calculated')
#         #
#         # self._ex, self._ey, self._ez = -np.gradient(self._pot, self._resolution, edge_order=2)
#         #
#         # stopwatch('field calculated')
#
#         return 0
#
#     def write_inventor_macro(self, save_folder=None):
#
#         if save_folder is None:
#             fd = FileDialog()
#             save_folder, _ = fd.get_filename('folder')
#
#         for direction in ["X", "Y"]:
#
#             # Generate text for Inventor macro
#             header_text = """Sub CreateRFQElectrode{}()
#                             Dim oApp As Application
#                             Set oApp = ThisApplication
#
#                             ' Get a reference to the TransientGeometry object.
#                             Dim tg As TransientGeometry
#                             Set tg = oApp.TransientGeometry
#
#                             Dim oPart As PartDocument
#                             Dim oCompDef As PartComponentDefinition
#                             Dim oSketch As Sketch3D
#                             Dim oSpline As SketchSplines3D
#                             Dim vertexCollection1 As ObjectCollection
#                             Dim oLine As SketchLines3D
#                             Dim number_of_points As Long
#                             Dim loft_section_index As Long
#                             Dim frequency As Integer: frequency = 10
#                             Dim oLoftDef As LoftDefinition
#                             Dim oLoftSections As ObjectCollection
#                             Dim spiral_electrode As LoftFeature
#                         """.format(direction)
#
#             electrode_text = """
#                                 Set oPart = oApp.Documents.Add(kPartDocumentObject, , True)
#
#                                 Set oCompDef = oPart.ComponentDefinition
#
#                             """
#             electrode_text += """
#                                 Set oSketch = oCompDef.Sketches3D.Add
#                                 Set oSpline = oSketch.SketchSplines3D
#                                 Set vertexCollection1 = oApp.TransientObjects.CreateObjectCollection(Null)
#
#                                 FileName = "{}"
#                                 fileNo = FreeFile 'Get first free file number
#
#                                 Open FileName For Input As #fileNo
#                                 Do While Not EOF(fileNo)
#
#                                     Dim strLine As String
#                                     Line Input #1, strLine
#
#                                     strLine = Trim$(strLine)
#
#                                     If strLine <> "" Then
#                                         ' Break the line up, using commas as the delimiter.
#                                         Dim astrPieces() As String
#                                         astrPieces = Split(strLine, ",")
#                                     End If
#
#                                     Call vertexCollection1.Add(tg.CreatePoint(astrPieces(0), astrPieces(1), astrPieces(2)))
#
#                                 Loop
#
#                                 Close #fileNo
#
#                                 Call oSpline.Add(vertexCollection1)
#
#                                 """.format(os.path.join(save_folder, "Transition_{}.txt".format(direction)))
#
#             footer_text = """
#                             oPart.UnitsOfMeasure.LengthUnits = kMillimeterLengthUnits
#
#                             ThisApplication.ActiveView.Fit
#
#                         End Sub
#                         """
#
#             with open(os.path.join(save_folder, "Transition_{}.ivb".format(direction)), "w") as outfile:
#
#                 outfile.write(header_text + electrode_text + footer_text)
#
#             z_start = 0.10972618296477678
#
#             with open(os.path.join(save_folder, "Transition_{}.txt".format(direction)), "w") as outfile:
#
#                 if direction == "X":
#
#                     # for i in range(len(self._z_linear)):
#                     for i in np.where(self._z_linear >= z_start)[0]:
#                         outfile.write("{:.6f}, {:.6f}, {:.6f}\n".format(
#                             self._vane_profile_x[i] * 100.0,  # For some weird reason Inventor uses cm as default...
#                             0.0,
#                             self._z_linear[i] * 100.0))
#
#                 else:
#
#                     # for i in range(len(self._z_linear)):
#                     for i in np.where(self._z_linear >= z_start)[0]:
#                         outfile.write("{:.6f}, {:.6f}, {:.6f}\n".format(
#                             0.0,
#                             self._vane_profile_y[i] * 100.0,  # For some weird reason Inventor uses cm as default...
#                             self._z_linear[i] * 100.0))
#
#         return 0


class PyRFQCell(object):
    def __init__(self,
                 cell_type,
                 aperture,
                 modulation,
                 length,
                 flip_z,
                 shift_cell_no,
                 prev_cell=None,
                 next_cell=None):

        assert cell_type in ["STA", "RMS", "NCS", "TCS", "DCS"], "cell_type must be one of RMS, NCS, TCS, DCS!"

        self._type = cell_type
        self._aperture = aperture
        self._modulation = modulation
        self._length = length
        self._flip_z = flip_z
        self._shift_cell_no = shift_cell_no
        self._profile_itp = None  # Interpolation of the cell profile
        self._prev_cell = prev_cell
        self._next_cell = next_cell

    def __str__(self):
        return "Type: '{}', Aperture: {:.6f}, Modulation: {:.4f}, " \
               "Length: {:.6f}, flip: {}, shift: {}".format(self._type,
                                                            self._aperture,
                                                            self._modulation,
                                                            self._length,
                                                            self._flip_z,
                                                            self._shift_cell_no)

    @property
    def length(self):
        return self._length

    @property
    def aperture(self):
        return self._aperture

    @property
    def avg_radius(self):
        return 0.5 * (self._aperture + self._modulation * self._aperture)

    @property
    def cell_type(self):
        return self._type

    @property
    def modulation(self):
        return self._modulation

    @property
    def prev_cell(self):
        return self._prev_cell

    @property
    def next_cell(self):
        return self._next_cell

    def calculate_transition_cell_length(self):

        le = self._length
        m = self._modulation
        a = self._aperture
        r0 = self.avg_radius
        k = np.pi / np.sqrt(3.0) / le

        def eta(kk):
            return bessel1(0.0, kk * r0) / (3.0 * bessel1(0.0, 3.0 * kk * r0))

        def func(kk):
            return (bessel1(0.0, kk * m * a) + eta(kk) * bessel1(0.0, 3.0 * kk * m * a)) / \
                   (bessel1(0.0, kk * a) + eta(kk) * bessel1(0.0, 3.0 * kk * a)) \
                   + ((m * a / r0) ** 2.0 - 1.0) / ((a / r0) ** 2.0 - 1.0)

        k = root(func, k).x[0]
        tcs_length = np.pi / 2.0 / k
        print("Transition cell has length {} which is {} * cell length, ".format(tcs_length, tcs_length / le), end="")
        print("the remainder will be filled with a drift.")

        assert tcs_length <= le, "Numerical determination of transition cell length " \
                                 "yielded value larger than cell length parameter!"

        return np.pi / 2.0 / k

    def set_prev_cell(self, prev_cell):
        assert isinstance(prev_cell, PyRFQCell), "You are trying to set a PyRFQCell with a non-cell object!"
        self._prev_cell = prev_cell

    def set_next_cell(self, next_cell):
        assert isinstance(next_cell, PyRFQCell), "You are trying to set a PyRFQCell with a non-cell object!"
        self._next_cell = next_cell

    def calculate_profile(self, cell_no, vane_type, fudge=False):
        print("cell_no: " + str(cell_no))
        assert vane_type in ["xp", "xm", "yp", "ym"], "Did not understand vane type {}".format(vane_type)

        if self._type == "STA":
            # Don't do anything for start cell
            return 0

        elif self._type == "DCS":
            self._profile_itp = interp1d([0.0, self._length],
                                         [self._aperture, self._aperture * self._modulation])
            return 0

        elif self._type == "RMS":

            # Find adjacent RMS cells and get their apertures
            cc = self
            pc = cc.prev_cell
            a = [cc.aperture]
            z = [cc.length]

            pc_is_rms = False
            if pc is not None:
                pc_is_rms = (pc.cell_type in ["RMS", "STA"])

            while pc_is_rms and cc.cell_type != "STA":

                a = [pc.aperture] + a
                z = [z[0] - cc.length] + z

                cc = pc
                pc = pc.prev_cell

                pc_is_rms = False
                if pc is not None:
                    pc_is_rms = (pc.cell_type in ["RMS", "STA"])

            cc = self
            nc = cc._next_cell

            if nc is not None:
                next_is_rms = (nc.cell_type == "RMS")

                while next_is_rms:

                    a.append(nc.aperture)
                    z.append(z[-1] + nc.length)

                    nc = nc.next_cell

                    next_is_rms = False
                    if nc is not None:
                        next_is_rms = (nc.cell_type == "RMS")

            self._profile_itp = interp1d(z, a, kind='cubic')

            return 0

        elif self._type == "TCS":

            le = self._length
            m = self._modulation
            a = self._aperture
            k = np.pi / np.sqrt(3.0) / le  # Initial guess
            r0 = 0.5 * (a + m * a)

            tcl = self.calculate_transition_cell_length()
            z = np.linspace(0.0, le, 200)

            idx = np.where(z <= tcl)

            vane = np.ones(z.shape) * r0

            print("Average radius of transition cell (a + ma) / 2 = {}".format(r0))

            def eta(kk):
                return bessel1(0.0, kk * r0) / (3.0 * bessel1(0.0, 3.0 * kk * r0))

            def a10(kk):
                return ((m * a / r0) ** 2.0 - 1.0) / (
                        bessel1(0.0, kk * m * a) + eta(kk) * bessel1(0.0, 3.0 * kk * m * a))

            def a30(kk):
                return eta(kk) * a10(kk)

            def func(kk):
                return (bessel1(0.0, kk * m * a) + eta(kk) * bessel1(0.0, 3.0 * kk * m * a)) / \
                       (bessel1(0.0, kk * a) + eta(kk) * bessel1(0.0, 3.0 * kk * a)) \
                       + ((m * a / r0) ** 2.0 - 1.0) / ((a / r0) ** 2.0 - 1.0)

            k = root(func, k).x[0]

            if self._shift_cell_no:
                sign = (-1.0) ** (cell_no + 1)
            else:
                sign = (-1.0) ** cell_no

            _vane = []

            if "x" in vane_type:

                def vane_x(xx):
                    return - (xx / r0) ** 2.0 \
                           + sign * a10(k) * bessel1(0.0, k * xx) * np.cos(k * _z) \
                           + sign * a30(k) * bessel1(0.0, 3.0 * k * xx) * np.cos(3.0 * k * _z) + 1.0

                for _z in z[idx]:
                    _vane.append(root(vane_x, r0).x[0])

            else:

                def vane_y(yy):
                    return + (yy / r0) ** 2.0 \
                           + sign * a10(k) * bessel1(0.0, k * yy) * np.cos(k * _z) \
                           + sign * a30(k) * bessel1(0.0, 3.0 * k * yy) * np.cos(3.0 * k * _z) - 1.0

                for _z in z[idx]:
                    _vane.append(root(vane_y, r0).x[0])

            if self._flip_z:
                _vane = _vane[::-1]
                vane[np.where(z >= le - tcl)] = _vane
            else:
                vane[idx] = _vane

            self._profile_itp = interp1d(z, vane, bounds_error=False, fill_value=0)

            return 0

        z = np.linspace(0.0, self._length, 100)

        a = self.aperture
        m = self.modulation

        pc = self._prev_cell
        if pc is not None and pc.cell_type in ["RMS", "DCS"]:
            pc = None
        nc = self._next_cell
        if nc is not None and nc.cell_type in ["RMS", "DCS"]:
            nc = None

        # print("Cell No {}, prev: {}, next: {}".format(cell_no, pc, nc))

        if pc is None or not fudge:
            a_fudge_begin = ma_fudge_begin = 1.0
        else:
            ma_fudge_begin = 0.5 * (1.0 + pc.aperture * pc.modulation / m / a)
            a_fudge_begin = 0.5 * (1.0 + pc.aperture / a)

        if nc is None or not fudge:
            a_fudge_end = ma_fudge_end = 1.0
        else:
            ma_fudge_end = 0.5 * (1.0 + nc.aperture * nc.modulation / m / a)
            a_fudge_end = 0.5 * (1.0 + nc.aperture / a)

        a_fudge = interp1d([0.0, self.length], [a_fudge_begin, a_fudge_end])
        ma_fudge = interp1d([0.0, self.length], [ma_fudge_begin, ma_fudge_end])
        kp = np.pi / self.length
        sign = (-1.0) ** (cell_no + 1)

        def ap(zz):
            return a * a_fudge(zz)

        def mp(zz):
            return m * ma_fudge(zz) / a_fudge(zz)

        def a10(zz):
            return (mp(zz) ** 2.0 - 1.0) / (mp(zz) ** 2.0 * bessel1(0, kp * ap(zz)) + bessel1(0, mp(zz) * kp * ap(zz)))

        def r0(zz):
            return ap(zz) / np.sqrt(1.0 - (mp(zz) ** 2.0 * bessel1(0, kp * ap(zz)) - bessel1(0, kp * ap(zz))) /
                                    (mp(zz) ** 2.0 * bessel1(0, kp * ap(zz)) + bessel1(0, mp(zz) * kp * ap(zz))))

        _vane = []

        if "x" in vane_type:

            def vane_x(xx):
                return + sign * (xx / r0(_z)) ** 2.0 + a10(_z) * bessel1(0.0, kp * xx) * np.cos(kp * _z) - sign

            for _z in z:
                _vane.append(root(vane_x, ap(_z)).x[0])

        else:

            def vane_y(yy):
                return - sign * (yy / r0(_z)) ** 2.0 + a10(_z) * bessel1(0.0, kp * yy) * np.cos(kp * _z) + sign

            for _z in z:
                _vane.append(root(vane_y, ap(_z)).x[0])

        self._profile_itp = interp1d(z, _vane)

        return 0

    def profile(self, z):

        # plt.figure()

        # domain = np.linspace(0.01828769716079613, 0.10972618296477678, 10)

        # plt.plot(domain, self._profile_itp(domain))
        # plt.show()

        return self._profile_itp(z)


class PyRFQVane(object):
    def __init__(self,
                 vane_type,
                 cells,
                 voltage,
                 mesh_size=0.001,
                 curvature=None,
                 height=0.001,
                 vane_z_data=None,
                 vane_y_data=None,
                 rotation=0.0,
                 debug=False):

        self._debug   = debug

        self._type    = vane_type
        self._cells   = cells
        self._voltage = voltage
        self._has_profile = False
        self._fudge = False

        self._length = np.sum([cell.length for cell in self._cells])  # type: float

        self._mesh_params = {"dx": 0.005,  # Mesh size (m)
                             "h": 0.005,  # gmsh meshing parameter (m)
                             "tip": "semi-circle",
                             "r_tip": 0.005,  # Radius of curvature of vane tip (m)
                             "h_block": 0.01,  # height of block sitting atop the curvature (m)
                             "symmetry": False,
                             "mirror": False,
                             "domain_idx": None,
                             "gmsh_str": None}

        self._mesh = None


        # Two Term potential
        self._curvature = curvature
        self._rotation  = rotation
        if height < 0.001:
            self._height = 0.001
        else:
            self._height = height

        self._vane_z_data = vane_z_data
        self._vane_y_data = vane_y_data
        self._mesh_size   = mesh_size
        self._grid = None

    @property
    def domain_idx(self):
        return self._mesh_params["domain_idx"]

    @property
    def has_profile(self):
        return self._has_profile

    @property
    def length(self):
        return self._length  # type: float

    @property
    def mesh(self):
        return self._mesh

    @property
    def vane_type(self):
        return self._type

    @property
    def vertices_elements(self):
        if self._mesh is not None:
            return self._mesh.leaf_view.vertices, self._mesh.leaf_view.elements
        else:
            return None, None

    @property
    def voltage(self):
        return self._voltage

    def get_parameter(self, key):

        if key in self._mesh_params.keys():
            return self._mesh_params[key]
        else:
            return None

    def set_voltage(self, voltage):
        self._voltage = voltage

    def set_domain_index(self, idx):
        self._mesh_params["domain_idx"] = idx

    def generate_gmsh_str(self, dx=None, h=None, symmetry=None, mirror=None):

        if symmetry is not None:
            self._mesh_params["symmetry"] = symmetry
        else:
            symmetry = self._mesh_params["symmetry"]

        if mirror is not None:
            self._mesh_params["mirror"] = mirror
        else:
            mirror = self._mesh_params["mirror"]

        assert not (symmetry is True and mirror is True), "Cannot have mirroring and symmetry at the same time!"

        if dx is not None:
            self._mesh_params["dx"] = dx
        else:
            dx = self._mesh_params["dx"]

        zmin = 0.0  # For now, RFQ vanes always start at 0.0, everything is relative
        zmax = self._length
        numz = np.round((zmax - zmin) / dx, 0) + 1
        sign = 1
        r_tip = self._mesh_params["r_tip"]
        h_block = self._mesh_params["h_block"]

        # Calculate z_data and vane profile:
        z, profile = self.get_profile(nz=numz)

        # Truncate for quicker results during debugging:
        # z = z[-200:]
        # profile = profile[-200:]

        ymax = r_tip + np.max(profile) + h_block

        # gmsh_str = "SetFactory('OpenCASCADE');\n"
        gmsh_str = """Geometry.NumSubEdges = 100; // nicer display of curve
                    Mesh.CharacteristicLengthMax = {};
                    h = {};

                    """.format(h, h)

        if symmetry:
            assert self._type not in ["ym", "xm"], "Sorry, mesh generation with symmetry only works for vanes " \
                                                   "located in positive axis directions (i.e. 'yp', 'xp'). "

            if "x" in self._type:
                sign = -1

        if "y" in self._type:
            self._mesh_params["domain_idx"] = 2
        else:
            self._mesh_params["domain_idx"] = 1

        new_pt = 1
        new_ln = 1

        surf_count = 1

        spline1_pts = [new_pt]

        # Center center spline (not actually a spline atm)
        for _z, _a in zip(z, profile):
            gmsh_str += "Point({}) = {{ {}, {}, {}, h }};\n".format(spline1_pts[-1],
                                                                    0.0, _a, _z)
            spline1_pts.append(spline1_pts[-1] + 1)

        new_pt = spline1_pts[-1]
        spline1_pts.pop(-1)

        spline1_lns = list(range(new_ln, new_ln + len(spline1_pts) - 1))
        new_ln = spline1_lns[-1] + 1

        gmsh_str += """
                    For i In {{{}:{}}}
                        Line(i{:+d}) = {{i-1, i}};
                    EndFor

                    """.format(spline1_pts[0] + 1, spline1_pts[-1], -(new_pt - new_ln))

        # Center outer spline (not actually a spline atm)
        spline2_pts = [new_pt]

        for _z, _a in zip(z, profile):
            gmsh_str += "Point({}) = {{ {}, {}, {}, h }};\n".format(spline2_pts[-1],
                                                                    r_tip, _a + r_tip, _z)
            spline2_pts.append(spline2_pts[-1] + 1)

        new_pt = spline2_pts[-1]
        spline2_pts.pop(-1)

        spline2_lns = list(range(new_ln, new_ln + len(spline2_pts) - 1))
        new_ln = spline2_lns[-1] + 1

        gmsh_str += """
                    For i In {{{}:{}}}
                        Line(i{:+d}) = {{i-1, i}};
                    EndFor

                    """.format(spline2_pts[0] + 1, spline2_pts[-1], -(new_pt - new_ln))

        # Four points on top
        top_pts = list(range(new_pt, new_pt + 4))
        new_pt = top_pts[-1] + 1

        gmsh_str += "Point({}) = {{ {}, {}, {}, h }};\n".format(top_pts[0], 0.0, ymax, z[-1])
        gmsh_str += "Point({}) = {{ {}, {}, {}, h }};\n".format(top_pts[1], 0.0, ymax, z[0])
        gmsh_str += "Point({}) = {{ {}, {}, {}, h }};\n".format(top_pts[2], r_tip, ymax, z[0])
        gmsh_str += "Point({}) = {{ {}, {}, {}, h }};\n".format(top_pts[3], r_tip, ymax, z[-1])
        gmsh_str += "\n"

        # Inner 3 surface lines and inner surface
        inner_lns = list(range(new_ln, new_ln + 3))
        new_ln = inner_lns[-1] + 1

        gmsh_str += "Line({}) = {{ {}, {} }};\n".format(inner_lns[0], spline1_pts[-1], top_pts[0])
        gmsh_str += "Line({}) = {{ {}, {} }};\n".format(inner_lns[1], top_pts[0], top_pts[1])
        gmsh_str += "Line({}) = {{ {}, {} }};\n\n".format(inner_lns[2], top_pts[1], spline1_pts[0])

        # gmsh_str += "ll = newll; " \
        #             "Line Loop (ll) = {{{}:{}, {}, {}, {}}}; " \
        #             "Plane Surface({}) = {{ll}};\n\n".format(sign * spline1_lns[0], sign * spline1_lns[-1],
        #                                                      sign * inner_lns[0], sign * inner_lns[1],
        #                                                      sign * inner_lns[2], surf_count)
        # surf_count += 1

        # Top 3 surface lines and top surface
        top_lns = list(range(new_ln, new_ln + 3))
        new_ln = top_lns[-1] + 1

        gmsh_str += "Line({}) = {{ {}, {} }};\n".format(top_lns[0], top_pts[1], top_pts[2])
        gmsh_str += "Line({}) = {{ {}, {} }};\n".format(top_lns[1], top_pts[2], top_pts[3])
        gmsh_str += "Line({}) = {{ {}, {} }};\n\n".format(top_lns[2], top_pts[3], top_pts[0])

        gmsh_str += "ll = newll; " \
                    "Line Loop (ll) = {{{}, {}, {}, {}}}; " \
                    "Plane Surface({}) = {{ll}};\n\n".format(-inner_lns[1] * sign, -top_lns[0] * sign,
                                                             -top_lns[1] * sign, -top_lns[2] * sign,
                                                             surf_count)
        surf_count += 1

        # Outer 2 lines and outer surface
        outer_lns = list(range(new_ln, new_ln + 2))
        new_ln = outer_lns[-1] + 1

        gmsh_str += "Line({}) = {{ {}, {} }};\n".format(outer_lns[0], spline2_pts[-1], top_pts[3])
        gmsh_str += "Line({}) = {{ {}, {} }};\n\n".format(outer_lns[1], top_pts[2], spline2_pts[0])

        gmsh_str += "ll = newll; " \
                    "Line Loop (ll) = {{{}:{}, {}, {}, {}}}; " \
                    "Plane Surface({}) = {{ll}};\n\n".format(-spline2_lns[0] * sign, -spline2_lns[-1] * sign,
                                                             -outer_lns[0] * sign, top_lns[1] * sign,
                                                             -outer_lns[1] * sign, surf_count)
        surf_count += 1

        # Center points for arcs
        arc_pts = [new_pt]

        for _z, _a in zip(z, profile):
            gmsh_str += "Point({}) = {{ {}, {}, {}, h }};\n".format(arc_pts[-1],
                                                                    0.0, _a + r_tip, _z)
            arc_pts.append(arc_pts[-1] + 1)

        new_pt = arc_pts[-1]
        arc_pts.pop(-1)

        arc_lns = list(range(new_ln, new_ln + len(arc_pts)))  # Cave: There are as many arcs as center points!
        new_ln = arc_lns[-1] + 1

        gmsh_str += """
                    For i In {{{}:{}}}
                        Circle(i{:+d}) = {{i - {}, i, i - {}}};
                    EndFor
                    """.format(arc_pts[0], arc_pts[-1], -(new_pt - new_ln), arc_pts[0] - spline1_pts[0], arc_pts[0] - spline2_pts[0])

                            # Line Loops and Surfaces for the Modulations
        gmsh_str += """
                    For i In {{{}:{}}}
                        ll = newll; Line Loop (ll) = {{{} * i, {} * (i {:+d}), {} * -(i + 1), {} * -(i {:+d})}}; Surface(i + {}) = {{ll}};
                    EndFor

                    """.format(arc_lns[0], arc_lns[-2],
                               sign, sign,
                               -(arc_lns[0] - spline2_lns[0]),
                               sign, sign,
                               -(arc_lns[0] - spline1_lns[0]),
                               surf_count - arc_lns[0])

        surf_count += len(arc_lns) - 1

        # Front and back surfaces
        gmsh_str += "ll = newll; " \
                    "Line Loop (ll) = {{{}, {}, {}, {}}}; " \
                    "Plane Surface({}) = {{ll}};\n\n".format(-arc_lns[0] * sign, outer_lns[1] * sign,
                                                             top_lns[0] * sign, -inner_lns[2] * sign,
                                                             surf_count)
        surf_count += 1

        gmsh_str += "ll = newll; " \
                    "Line Loop (ll) = {{{}, {}, {}, {}}}; " \
                    "Plane Surface({}) = {{ll}};\n\n".format(arc_lns[-1] * sign, outer_lns[0] * sign,
                                                             top_lns[2] * sign, -inner_lns[0] * sign,
                                                             surf_count)

        if not symmetry:
            # Mirror the half-vane on yz plane
            gmsh_str += "new_surfs[] = Symmetry {{1, 0, 0, 0}} " \
                        "{{Duplicata{{Surface {{1:{}}};}}}};\n".format(surf_count)

            if mirror:
                # Mirror the resulting vane on the xz plane (need to do it separately for both
                # shells to get surface normal right
                gmsh_str += "mir_vane1[] = Symmetry {{0, 1, 0, 0}} " \
                            "{{Duplicata{{Surface {{1:{}}};}}}};\n".format(surf_count)
                gmsh_str += "mir_vane2[] = Symmetry {{0, 1, 0, 0}} " \
                            "{{Duplicata{{Surface {{new_surfs[]}};}}}};\n".format(surf_count)

                # Add physical surface to identify this vane in gmsh
                gmsh_str += "Physical Surface({}) = {{1:{}, -new_surfs[], -mir_vane1[], mir_vane2[]}};\n\n".format(
                    self._mesh_params["domain_idx"], surf_count)

                # Rotate if necessary
                if self.vane_type == "xp":
                    gmsh_str += "Rotate {{{{0, 0, 1}}, {{0, 0, 0}}, {}}} " \
                                "{{Surface {{1:{}, new_surfs[], mir_vane1[], mir_vane2[]}};}}\n".format(-0.5 * np.pi,
                                                                                                        surf_count)
                elif self.vane_type == "xm":
                    gmsh_str += "Rotate {{{{0, 0, 1}}, {{0, 0, 0}}, {}}} " \
                                "{{Surface {{1:{}, new_surfs[], mir_vane1[], mir_vane2[]}};}}\n".format(0.5 * np.pi,
                                                                                                        surf_count)
                elif self.vane_type == "ym":
                    gmsh_str += "Rotate {{{{0, 0, 1}}, {{0, 0, 0}}, {}}} " \
                                "{{Surface {{1:{}, new_surfs[], mir_vane1[], mir_vane2[]};}}\n".format(np.pi,
                                                                                                       surf_count)

            else:
                # Add physical surface to identify this vane in gmsh (unmirrored)
                gmsh_str += "Physical Surface({}) = {{1:{}, -new_surfs[]}};\n\n".format(self._mesh_params["domain_idx"],
                                                                                        surf_count)

                # Rotate if necessary
                if self.vane_type == "xp":
                    gmsh_str += "Rotate {{{{0, 0, 1}}, {{0, 0, 0}}, {}}} " \
                                "{{Surface {{1:{}, new_surfs[]}};}}\n".format(-0.5 * np.pi, surf_count)
                elif self.vane_type == "xm":
                    gmsh_str += "Rotate {{{{0, 0, 1}}, {{0, 0, 0}}, {}}} " \
                                "{{Surface {{1:{}, new_surfs[]}};}}\n".format(0.5 * np.pi, surf_count)
                elif self.vane_type == "ym":
                    gmsh_str += "Rotate {{{{0, 0, 1}}, {{0, 0, 0}}, {}}} " \
                                "{{Surface {{1:{}, new_surfs[]}};}}\n".format(np.pi, surf_count)

        else:
            # Add physical surface to identify this vane in gmsh
            gmsh_str += "Physical Surface({}) = {{1:{}}};\n\n".format(self._mesh_params["domain_idx"],
                                                                      surf_count)
            # Create the Neumann BC surface
            axis_pts = list(range(new_pt, new_pt + 2))
            # new_pt = axis_pts[-1] + 1

            # The points on z axis
            gmsh_str += "Point({}) = {{ {}, {}, {}, h }};\n".format(axis_pts[0], 0.0, 0.0, z[0])
            gmsh_str += "Point({}) = {{ {}, {}, {}, h }};\n".format(axis_pts[1], 0.0, 0.0, z[-1])

            # NBC 3 lines and NBC surface
            nb_lns = list(range(new_ln, new_ln + 3))
            # new_ln = nb_lns[-1] + 1

            gmsh_str += "Line({}) = {{ {}, {} }};\n".format(nb_lns[0], spline1_pts[-1], axis_pts[1])
            gmsh_str += "Line({}) = {{ {}, {} }};\n".format(nb_lns[1], axis_pts[1], axis_pts[0])
            gmsh_str += "Line({}) = {{ {}, {} }};\n\n".format(nb_lns[2], axis_pts[0], spline1_pts[0])

            surf_count += 1

            gmsh_str += "ll = newll; " \
                        "Line Loop (ll) = {{{}:{}, {}, {}, {}}}; " \
                        "Plane Surface({}) = {{ll}};\n\n".format(spline1_lns[0] * sign, spline1_lns[-1] * sign,
                                                                 nb_lns[0] * sign, nb_lns[1] * sign, nb_lns[2] * sign,
                                                                 surf_count)

            # Add physical surface to identify the Neumann BC in gmsh
            gmsh_str += "Physical Surface({}) = {{{}}};\n\n".format(0, surf_count)

            # TODO: still need to mirror this on x=y plane for second vane!

        with open("test_{}.geo".format(self._type), "w") as _of:
            _of.write(gmsh_str)

        self._mesh_params["gmsh_str"] = gmsh_str

        return gmsh_str

    # def calculate_mesh(self, dx=None):
    #
    #     if dx is not None:
    #         self._mesh_params["dx"] = dx
    #     else:
    #         dx = self._mesh_params["dx"]
    #
    #     zmin = 0.0  # For now, RFQ vanes always start at 0.0, everything is relative
    #     zmax = self._length
    #     numz = np.round((zmax - zmin) / dx, 0) + 1
    #
    #     r_tip = self._mesh_params["r_tip"]
    #     h_block = self._mesh_params["h_block"]
    #
    #     # Calculate z_data and vane profile:
    #     z, vane = self.get_profile(nz=numz)
    #
    #     if self._mesh_params["tip"] == "semi-circle":
    #         # Calculate approximate angular resolution corresponding to desired mesh size
    #         num_phi = np.round(r_tip * np.pi / dx, 0)
    #         phi = np.pi / num_phi
    #
    #         print("With mesh_size {} m, we have {} points per semi-circle".format(dx, num_phi))
    #
    #         # We need two sets of phi values so that subsequent z positions form triangles rather than squares
    #         phi_set = [np.linspace(np.pi, 2.0 * np.pi, num_phi),
    #                    np.linspace(np.pi + 0.5 * phi, 2.0 * np.pi - 0.5 * phi, num_phi - 1)]
    #
    #         # maximum vertical extent:
    #         ymax = r_tip + np.max(vane) + h_block
    #         print("Maximum vertical extent: {} m".format(ymax))
    #         # TODO: Think about whether it's a good idea to use only half the mesh size on block part
    #         num_vert_pts = int(np.round((ymax - r_tip - np.min(vane)) / 2.0 / dx, 0))
    #         print("Number of points in vertical direction: {}".format(num_vert_pts))
    #         num_horz_pts = int(np.round(2.0 * r_tip / 2.0 / dx, 0)) - 1  # Not counting corners
    #         horz_step_size = (2.0 * r_tip) / (num_horz_pts + 1)
    #
    #         # Create point cloud for testing
    #         points = []
    #         for i, _z in enumerate(z):
    #
    #             # Create curved surface points:
    #             for _phi in phi_set[int(i % 2.0)]:
    #                 points.append((r_tip * np.cos(_phi),
    #                                r_tip * np.sin(_phi) + r_tip + vane[i],
    #                                _z))
    #
    #             # Create straight lines points:
    #             # Looking with z, we start at the right end of the curvature and go up-left-down
    #             vert_step = (ymax - r_tip - vane[i]) / num_vert_pts
    #
    #             # If we are in phi_set 0, we start one step up, in phi set 1 a half step up
    #             for j in range(num_vert_pts):
    #                 points.append((r_tip, points[-1][1] + vert_step, _z))
    #
    #             if i % 2.0 == 0.0:
    #                 points.pop(-1)
    #
    #             # Append the corner point
    #             points.append((r_tip, ymax, _z))
    #
    #             if i % 2.0 == 0.0:
    #                 for j in range(num_horz_pts + 1):
    #                     points.append((r_tip + 0.5 * horz_step_size - (j + 1) * horz_step_size, ymax, _z))
    #             else:
    #                 for j in range(num_horz_pts):
    #                     points.append((r_tip - (j + 1) * horz_step_size, ymax, _z))
    #
    #             # Append the corner point
    #             points.append((-r_tip, ymax, _z))
    #
    #             # Finish up by going down on the left side
    #             for j in range(num_vert_pts):
    #                 points.append((-r_tip, ymax - (j + 1) * vert_step + 0.5 * (i % 2.0) * vert_step, _z))
    #
    #             if i % 2.0 == 0.0:
    #                 points.pop(-1)
    #
    #         points = np.array(points, dtype=[("x", float), ("y", float), ("z", float)])
    #
    #         # Apply rotation
    #         rot = rot_map[self._type]
    #
    #         if rot != 0.0:
    #             angle = rot * np.pi / 180.0
    #
    #             x_old = points["x"].copy()
    #             y_old = points["y"].copy()
    #
    #             points["x"] = x_old * np.cos(angle) - y_old * np.sin(angle)
    #             points["y"] = x_old * np.sin(angle) + y_old * np.cos(angle)
    #
    #             del x_old
    #             del y_old
    #
    #         points_per_slice = len(np.where(points["z"] == z[0])[0])
    #         print("Points per slice = {}".format(points_per_slice))
    #
    #         # For each point, there need to be lines created to neighbors
    #         elements = []
    #         for i, point in enumerate(points[:-points_per_slice]):
    #             # Each element contains the indices of the bounding vertices
    #             slice_no = int(i / points_per_slice)
    #             # print(i, slice_no, int(slice_no % 2))
    #
    #             if slice_no % 2 == 0.0:
    #
    #                 if (i + 1) % points_per_slice != 0.0:
    #                     elements.append([i, i + 1, i + points_per_slice])
    #                 else:
    #                     elements.append(
    #                         [i, i + points_per_slice, i + 1 - points_per_slice])
    #
    #                 if i % points_per_slice != 0.0:
    #                     elements.append(
    #                         [i, i + points_per_slice - 1, i + points_per_slice])
    #                 else:
    #                     elements.append(
    #                         [i, i + points_per_slice, i + 2 * points_per_slice - 1])
    #
    #             else:
    #
    #                 if (i + 1) % points_per_slice != 0.0:
    #                     # Regular element
    #                     elements.append([i, i + 1, i + points_per_slice + 1])
    #                     elements.append([i, i + points_per_slice, i + points_per_slice + 1])
    #                 else:
    #                     # Last element of level
    #                     elements.append([i, i + 1 - points_per_slice, i + 1])
    #                     elements.append([i, i + 1, i + points_per_slice])
    #
    #         vertices = np.array([points["x"], points["y"], points["z"]])
    #         elements = np.array(elements).T
    #
    #         # Test bempp calculation for single vane
    #         if bempp is not None:
    #             # noinspection PyUnresolvedReferences
    #             self._mesh = bempp.api.grid.grid_from_element_data(vertices, elements)
    #
    #             # if self._debug:
    #             #     self._mesh.plot()
    #
    #         else:
    #             return 0
    #
    #     else:
    #         print("The vane type '{}' is not (yet) implemented. Aborting.".format(self._mesh_params["tip"]))
    #         return 1
    #     return 0

    def calculate_profile(self, fudge=None):

        if fudge is None:
            fudge = self._fudge

        for cell_no in range(len(self._cells)):
            self._cells[cell_no].calculate_profile(cell_no, self._type, fudge=fudge)

        self._has_profile = True

        return 0

    def get_profile(self, nz=1000):

        assert self._has_profile, "No profile has been generated!"

        z = np.round(np.linspace(0.0, self._length, nz), decimals)
        vane = np.zeros(z.shape)
        cum_len = 0.0

        for cell in self._cells:

            if cell.cell_type != "STA":
                idx = np.where((z >= cum_len) & (z <= cum_len + cell.length))

                vane[idx] = cell.profile(np.round(z[idx] - cum_len, decimals))

                cum_len += cell.length

        return z, vane

    def plot_mesh(self):

        if bempp is None or self._mesh is None:
            print("Either BEMPP couldn't be loaded or there is not yet a mesh generated!")
            return 1

        self._mesh.plot()

        return 0

    # two term functions from field_from_two_term_potential.py
    # def __add__(self, other):
    #
    #     assert self.has_grid() and other.has_grid(), \
    #         "Can only add two vanes with computed grids. Use generate_grid() first!"
    #
    #     # TODO: Collision detection???
    #
    #     vertices1, elements1 = self._grid.leaf_view.vertices, self._grid.leaf_view.elements
    #     no_vert_grid1 = vertices1.shape[1]
    #
    #     vertices2, elements2 = other.get_grid().leaf_view.vertices, other.get_grid().leaf_view.elements
    #
    #     vertices = np.append(vertices1, vertices2, axis=1)
    #     elements = np.append(elements1, elements2 + no_vert_grid1, axis=1)
    #
    #     # Adding together two Vane objects makes vane_z_data and vane_y_data obsolete
    #     new_vane = Vane(vane_type=self._vane_type,
    #                     mesh_size=self._mesh_size,
    #                     curvature=self._curvature,
    #                     height=self._height,
    #                     vane_z_data=None,
    #                     vane_y_data=None,
    #                     debug=self._debug)
    #
    #     import bempp.api
    #
    #     new_vane.set_grid(bempp.api.grid.grid_from_element_data(vertices, elements))
    #
    #     return new_vane

    # def generate_grid(self, mesh_size=None):
    #
    #     if mesh_size is not None:
    #         self._mesh_size = mesh_size
    #
    #     zmin = np.min(self._vane_z_data)
    #     zmax = np.max(self._vane_z_data)
    #     numz = np.round((zmax - zmin) / self._mesh_size, 0) + 1
    #
    #     if numz == len(self._vane_z_data):
    #
    #         vane_z_data = self._vane_z_data
    #         vane_y_data = self._vane_y_data
    #
    #     else:
    #         vane_interp = interp1d(self._vane_z_data, self._vane_y_data)
    #         vane_z_data = np.linspace(zmin, zmax, numz)
    #         vane_y_data = vane_interp(vane_z_data)
    #
    #     if self._vane_type == "semi-circle":
    #         # Calculate approximate angular resolution corresponding to desired mesh size
    #         num_phi = np.round(self._curvature * np.pi / self._mesh_size, 0)
    #         phi = np.pi / num_phi
    #
    #         print("With mesh_size {} m, we have {} points per semi-circle".format(self._mesh_size, num_phi))
    #
    #         # We need two sets of phi values so that subsequent z positions form triangles rather than squares
    #         phi_set = [np.linspace(np.pi, 2.0 * np.pi, num_phi),
    #                    np.linspace(np.pi + 0.5 * phi, 2.0 * np.pi - 0.5 * phi, num_phi - 1)]
    #
    #         # maximum vertical extent:
    #         ymax = self._curvature + np.max(vane_y_data) + self._height
    #         print("Maximum vertical extent: {} m".format(ymax))
    #         # TODO: Think about whether it's a good idea to use only half the mesh size on block part
    #         num_vert_pts = int(np.round((ymax - self._curvature - np.min(vane_y_data)) / 2.0 / self._mesh_size, 0))
    #         print("Number of points in vertical direction: {}".format(num_vert_pts))
    #         num_horz_pts = int(np.round(2.0 * self._curvature / 2.0 / self._mesh_size, 0)) - 1  # Not counting corners
    #         horz_step_size = (2.0 * self._curvature) / (num_horz_pts + 1)
    #
    #         # Create point cloud for testing
    #         points = []
    #         for i, _z in enumerate(vane_z_data):
    #
    #             # Create curved surface points:
    #             for _phi in phi_set[int(i % 2.0)]:
    #                 points.append((self._curvature * np.cos(_phi),
    #                                self._curvature * np.sin(_phi) + self._curvature + vane_y_data[i],
    #                                _z))
    #
    #             # Create straight lines points:
    #             # Looking with z, we start at the right end of the curvature and go up-left-down
    #             vert_step = (ymax - self._curvature - vane_y_data[i]) / num_vert_pts
    #
    #             # If we are in phi_set 0, we start one step up, in phi set 1 a half step up
    #             for j in range(num_vert_pts):
    #                 points.append((self._curvature, points[-1][1] + vert_step, _z))
    #
    #             if i % 2.0 == 0.0:
    #                 points.pop(-1)
    #
    #             # Append the corner point
    #             points.append((self._curvature, ymax, _z))
    #
    #             if i % 2.0 == 0.0:
    #                 for j in range(num_horz_pts + 1):
    #                     points.append((self._curvature + 0.5 * horz_step_size - (j + 1) * horz_step_size, ymax, _z))
    #             else:
    #                 for j in range(num_horz_pts):
    #                     points.append((self._curvature - (j + 1) * horz_step_size, ymax, _z))
    #
    #             # Append the corner point
    #             points.append((-self._curvature, ymax, _z))
    #
    #             # Finish up by going down on the left side
    #             for j in range(num_vert_pts):
    #                 points.append((-self._curvature, ymax - (j + 1) * vert_step + 0.5 * (i % 2.0) * vert_step, _z))
    #
    #             if i % 2.0 == 0.0:
    #                 points.pop(-1)
    #
    #         points = np.array(points, dtype=[("x", float), ("y", float), ("z", float)])
    #
    #         # Apply rotation
    #         if self._rotation != 0.0:
    #
    #             angle = self._rotation * np.pi / 180.0
    #
    #             x_old = points["x"].copy()
    #             y_old = points["y"].copy()
    #
    #             points["x"] = x_old * np.cos(angle) - y_old * np.sin(angle)
    #             points["y"] = x_old * np.sin(angle) + y_old * np.cos(angle)
    #
    #             del x_old
    #             del y_old
    #
    #         points_per_slice = len(np.where(points["z"] == vane_z_data[0])[0])
    #         print("Points per slice = {}".format(points_per_slice))
    #
    #         # For each point, there need to be lines created to neighbors
    #         elements = []
    #         for i, point in enumerate(points[:-points_per_slice]):
    #             # Each element contains the indices of the bounding vertices
    #             slice_no = int(i / points_per_slice)
    #             # print(i, slice_no, int(slice_no % 2))
    #
    #             if slice_no % 2 == 0.0:
    #
    #                 if (i + 1) % points_per_slice != 0.0:
    #                     elements.append([i, i + 1, i + points_per_slice])
    #                 else:
    #                     elements.append(
    #                         [i, i + points_per_slice, i + 1 - points_per_slice])
    #
    #                 if i % points_per_slice != 0.0:
    #                     elements.append(
    #                         [i, i + points_per_slice - 1, i + points_per_slice])
    #                 else:
    #                     elements.append(
    #                         [i, i + points_per_slice, i + 2 * points_per_slice - 1])
    #
    #             else:
    #
    #                 if (i + 1) % points_per_slice != 0.0:
    #                     # Regular element
    #                     elements.append([i, i + 1, i + points_per_slice + 1])
    #                     elements.append([i, i + points_per_slice, i + points_per_slice + 1])
    #                 else:
    #                     # Last element of level
    #                     elements.append([i, i + 1 - points_per_slice, i + 1])
    #                     elements.append([i, i + 1, i + points_per_slice])
    #
    #         # get first slice indices
    #         # idx = np.where(points[2] == vane_z_data[0])
    #         # plt.scatter(points[0][idx], points[1][idx], color="blue")
    #
    #         # get second slice indices
    #         # idx = np.where(points[2] == vane_z_data[1])
    #         # plt.scatter(points[0][idx], points[1][idx], color="red")
    #         #
    #         # plt.gca().set_aspect('equal')
    #
    #         # from mpl_toolkits.mplot3d import Axes3D
    #         # fig = plt.figure()
    #         # ax = fig.add_subplot(111, projection='3d')
    #         # for element in elements:
    #         #     triangle = np.append(points[element], points[element[0]])
    #         #     ax.plot(triangle["x"], triangle["y"], triangle["z"])
    #         # # for i, point in enumerate(points):
    #         # #     ax.scatter(point["x"], point["y"], point["z"], marker=r"${}$".format(i), s=49)
    #         # ax.set_xlabel("x")
    #         # ax.set_ylabel("y")
    #         # ax.set_aspect("equal")
    #         # plt.show()
    #         #
    #         # exit()
    #
    #         vertices = np.array([points["x"], points["y"], points["z"]])
    #         elements = np.array(elements).T
    #
    #         # Test bem++ calculation for single vane
    #         try:
    #
    #             import bempp.api
    #
    #         except ImportError as e:
    #
    #             print("Couldn't find module bempp. This only works in Linux environments!")
    #             print("Error was: {}".format(e))
    #             exit(1)
    #
    #         self._grid = bempp.api.grid.grid_from_element_data(vertices, elements)
    #
    #         if self._debug:
    #
    #             self._grid.plot()
    #
    #     else:
    #         print("The vane type '{}' is not (yet) implemented. Aborting.".format(self._vane_type))
    #         return 1

    def get_grid(self):
        return self._grid

    def has_grid(self):
        return self._grid is not None

    def plot_grid(self):

        if self._grid is None:
            print("No grid generated or loaded yet")
            return 1

        self._grid.plot()

        return 0

    def rotate_vane(self, angle):

        if self._grid is None:
            print("Please generate a grid first.")
            return 1

        # TODO: Implement rotation after generation of vanes
        print("Rotation of existing grid by {} rad...not yet implemented...".format(angle))

        return 0

    def set_grid(self, grid):

        # TODO: Assert if grid is bempp compatible grid object!

        self._grid = grid


# noinspection PyUnresolvedReferences
class PyRFQ(object):
    def __init__(self, voltage=None, filename=None, from_cells=False,
                 twoterm=True, boundarymethod=False, debug=False):

        self._debug = debug
        self._voltage = voltage
        self._vanes = []
        self._cells = []
        self._cell_nos = []
        self._length = 0.0
        self._full_mesh = None

        self._variables_bempp = {"solution": None,
                                 "f_space": None,
                                 "operator": None,
                                 "grid_fun": None,
                                 "grid_res": 0.005,  # grid resolution in (m)
                                 "ef_itp": None,  # type: Field
                                 "ef_phi": None,  # type: np.ndarray
                                 "pot_shift": 0.0,  # Shift all potentials by this value (and the solution back)
                                                    # This can help with jitter on z axis where pot ~ 0 otherwise
                                 # TODO: Should put pot in its own class that also holds dx, nx, etc.
                                 "add_cyl": False,  # Do we want to add a grounded cylinder to the BEMPP problem
                                 "add_endplates": False,  # Or just grounded end plates
                                 "cyl_id": 0.2,  # Inner diameter of surrounding cylinder
                                 "cyl_gap": 0.01  # gap between vanes and cylinder TODO: Maybe make this asymmetric?
                                 }

        self._variables_inventor = {"vane_type": "hybrid",
                                    "vane_radius": 0.005,  # m
                                    "vane_height": 0.05,  # m
                                    "vane_height_type": 'absolute',
                                    "nz": 500
                                    }


        ###### From rfq helper class ######

        # User Parameters
        #     Passed in via constructor
        self._from_cells     = from_cells      # run and calculate from parmteq cells
        self._filename       = filename        
        self._twoterm        = twoterm         # Use Two Term potential method to calculate field
        self._boundarymethod = boundarymethod  # Use BEMPP to calculate field


        #     Must be set outside of object creation
        self.simple_rods    = True  # cylindrical vanes
        self.vane_radius    = None  # radius of simple rods
        self.vane_distance  = None  # vane distance from axis
        self.rf_freq        = None  # RF frequency
        self.zstart         = 0.0   # start of the RFQ
        self.sim_start      = 0.0   # start of the simulation
        self.sim_end_buffer = 0.0   # added distance not part of RFQ but part of simulation
        self.resolution     = 0.002
        
        #optional
        self.endplates      = False
        self.endplates_outer_diameter = 0.2
        self.endplates_inner_diameter = 0.1
        self.endplates_distance_from_vanes = 0.1
        self.endplates_thickness = 0.1


        self.xy_limits     = None   # X and Y limits in the field calculation
        self.z_limits      = None   # Z limits for the field calculation
        self.ignore_rms    = False

        # Two term variables
        self.tt_a_init     = None   # initial aperture size for two term potential calculation

        # Bempp variables
        self.add_endplates  = True
        self.cyl_id         = None
        self.grid_res_bempp = None
        self.pot_shift      = None

        # "Private" variables
        self._conductors    = None
        self._field         = FieldLoader()
        self._sim_end       = 0.0
        self._length        = 0.0
        self._fieldzmax     = 0.0

        # Debugging
        self._ray           = [] #debugging

    def __str__(self):
        text = "\nPyRFQ object with {} cells and length {:.4f} m. Vane voltage = {} V\n".format(self._cell_nos[-1],
                                                                                                self._length,
                                                                                                self._voltage)
        text += "Cells:\n"
        for i, cell in enumerate(self._cells):
            text += "Cell {}: ".format(i) + cell.__str__() + "\n"

        return text

    def set_bempp_parameter(self, keyword=None, value=None):

        if keyword is None or value is None:
            print("In 'set_bempp_parameter': Either keyword or value were not specified.")
            return 1

        if keyword not in self._variables_bempp.keys():
            print("In 'set_bempp_parameter': Unrecognized keyword '{}'.".format(keyword))
            return 1

        self._variables_bempp[keyword] = value

        return 0

    def get_bempp_parameter(self, keyword=None):

        if keyword is None or value is None:
            print("In 'set_bempp_parameter': Either keyword or value were not specified.")
            return 1

        if keyword not in self._variables_bempp.keys():
            print("In 'set_bempp_parameter': Unrecognized keyword '{}'.".format(keyword))
            return 1

        return self._variables_bempp[keyword]

    def append_cell(self,
                    cell_type,
                    aperture,
                    modulation,
                    length,
                    flip_z=False,
                    shift_cell_no=False):
        assert cell_type in ["STA", "RMS", "NCS", "TCS", "DCS"], "cell_type must be one of STA, RMS, NCS, TCS, DCS!"

        if len(self._cells) > 0:
            pc = self._cells[-1]
        else:
            pc = None

        self._cells.append(PyRFQCell(cell_type=cell_type,
                                     aperture=aperture,
                                     modulation=modulation,
                                     length=length,
                                     flip_z=flip_z,
                                     shift_cell_no=shift_cell_no,
                                     prev_cell=pc,
                                     next_cell=None))

        if len(self._cells) > 1:
            self._cells[-2].set_next_cell(self._cells[-1])

        self._cell_nos = range(len(self._cells))
        self._length = np.sum([cell.length for cell in self._cells])

        return 0

    def add_cells_from_file(self, filename=None, ignore_rms=False):
        """
        Reads a file with cell parameters and generates the respective RFQCell objects
        :param filename:
        :param ignore_rms: Bool. If True, any radial matching cells in the file are ignored.
        :return:
        """

        if filename is None:
            if rank == 0:
                # print("Process {} getting filename from dialog".format(rank))
                # from dans_pymodules import FileDialog
                fd = FileDialog()
                filename = fd.get_filename('open')
                data = {"fn": filename}
                # req = comm.isend({'fn':filename}, dest=1, tag=11)
                # req.wait()
            else:
                # req = comm.irecv(source=0, tag=11)
                data = None
                # print("Process {} received filename {}.".format(rank, data["fn"]))

            data = comm.bcast(data, root=0)
            filename = data["fn"]

        if filename is None:
            return 1

        with open(filename, "r") as infile:
            if "Parmteqm" in infile.readline():
                # Detected Parmteqm file
                self.read_input_parmteq(filename, ignore_rms)
            else:
                # Assume only other case is VECC input file for now
                self.read_input_vecc(filename, ignore_rms)

        return 0

    def read_input_parmteq(self, filename, ignore_rms):
        # Parameters: Filename, whether to ignore rms
        # Returns: None
        # Reads in cell data from a parmteq file

        with open(filename, "r") as infile:

            # Some user feedback:
            version = infile.readline().strip().split()[1].split(",")[0]
            print("Loading cells from Parmteqm v{} output file...".format(version))

            # Find begin of cell information
            for line in infile:
                if "Cell" in line and "V" in line:
                    break

            for line in infile:
                # Last line in cell data is repetition of header line
                if "Cell" in line and "V" in line:
                    break

                # Cell number is a string (has key sometimes)
                items = line.strip().split()
                cell_no = items[0]
                params = [float(item) for item in items[1:]]

                if len(items) == 10 and cell_no == "0":
                    # This is the start cell, only there to provide a starting aperture
                    if len(self._cells) == 0 and not ignore_rms:
                        # We use this only if there are no previous cells in the pyRFQ
                        # Else we ignore it...
                        self._cells.append(PyRFQCell(cell_type="STA",
                                                     aperture=params[6] * 0.01,
                                                     modulation=params[7],
                                                     length=0.0,
                                                     flip_z=False,
                                                     shift_cell_no=False,
                                                     prev_cell=None,
                                                     next_cell=None))

                    continue

                # For now we ignore "special" cells and add them manually
                if "T" in cell_no or "M" in cell_no or "F" in cell_no:
                    print("Ignored cell {}".format(cell_no))
                    continue

                if params[7] == 1.0:
                    cell_type = "RMS"
                    if ignore_rms:
                        print("Ignored cell {}".format(cell_no))
                        continue
                else:
                    cell_type = "NCS"

                if len(self._cells) > 0:
                    pc = self._cells[-1]
                else:
                    pc = None

                self._cells.append(PyRFQCell(cell_type=cell_type,
                                             aperture=params[6] * 0.01,
                                             modulation=params[7],
                                             length=params[9] * 0.01,
                                             flip_z=False,
                                             shift_cell_no=False,
                                             prev_cell=pc,
                                             next_cell=None))
                if len(self._cells) > 1:
                    self._cells[-2].set_next_cell(self._cells[-1])

        self._cell_nos = range(len(self._cells))
        self._length = np.sum([cell.length for cell in self._cells])

    def read_input_vecc(self, filename, ignore_rms):
        # Parameters: Filename and whether to ignore rms
        # Returns: None
        # Reads cell data in from vecc file

        with open(filename, "r") as infile:

            for line in infile:

                params = [float(item) for item in line.strip().split()]

                if params[4] == 1.0:
                    cell_type = "RMS"
                    if ignore_rms:
                        continue
                else:
                    cell_type = "NCS"

                if len(self._cells) > 0:
                    pc = self._cells[-1]
                else:
                    pc = None

                self._cells.append(PyRFQCell(cell_type=cell_type,
                                             aperture=params[3],
                                             modulation=params[4],
                                             length=params[6],
                                             flip_z=False,
                                             shift_cell_no=False,
                                             prev_cell=pc,
                                             next_cell=None))
                if len(self._cells) > 1:
                    self._cells[-2].set_next_cell(self._cells[-1])

        self._cell_nos = range(len(self._cells))
        self._length = np.sum([cell.length for cell in self._cells])

    def calculate_efield(self):
        # TODO: missing parameters (_d)

        assert self._variables_bempp["ef_phi"] is not None, \
            "Please calculate the potential first!"

        ex, ey, ez = np.gradient(self._variables_bempp["ef_phi"],
                                 _d[X], _d[Y], _d[Z])

        if rank == 0:

            _field = Field("RFQ E-Field",
                           dim=3,
                           field={"x": RegularGridInterpolator(points=_r, values=-ex,
                                                               bounds_error=False, fill_value=0.0),
                                  "y": RegularGridInterpolator(points=_r, values=-ey,
                                                               bounds_error=False, fill_value=0.0),
                                  "z": RegularGridInterpolator(points=_r, values=-ez,
                                                               bounds_error=False, fill_value=0.0)
                                  })

            mpi_data = {"efield": _field}
        else:
            mpi_data = None

        mpi_data = comm.bcast(mpi_data, root=0)

        self._variables_bempp["ef_itp"] = mpi_data["efield"]

        return 0

    def calculate_potential(self,
                            limits=((None, None), (None, None), (None, None)),
                            res=(0.002, 0.002, 0.002),
                            domain_decomp=(4, 4, 4),
                            overlap=0):
        """
        Calculates the E-Field from the BEMPP solution using the user defined cube or
        the cube corresponding to the cyclindrical outer boundary.

        TODO: This function is not very MPI aware and could be optimized!
        TODO: BEMPP uses all available processors on the node to calculate the potential.
        TODO: But if we run on multiple nodes, we could partition the domains.

        :param limits: tuple, list or np.ndarray of shape (3, 2)
                       containing xmin, xmax, ymin, ymax, zmin, zmax
                       use None to use the individual limit from the electrode system.
        :param res: resolution of the 3D mesh
        :param domain_decomp: how many subdomains to use for calculation in the three directions x, y, z
                              Note: it can significantly increase computation speed to use more subdomains,
                              up to a point...
        :param overlap: overlap of the subdomains in cell numbers, does not have effect at the moment.
                        Note: There is a minimum overlap of one cell at overlap = 0
        :return:
        """

        limits = np.array(limits)

        if limits.shape != (3, 2):
            print("Wrong shape of limits: {}. "
                  "Must be ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = (3, 2).".format(limits.shape))
            return 1

        sol = self._variables_bempp["solution"]
        fsp = self._variables_bempp["f_space"]

        if sol is None:
            print("Please solve with BEMPP before calculating the E-Field")
            return 1

        _ts = time.time()

        # noinspection PyUnresolvedReferences
        all_vert = self._full_mesh.leaf_view.vertices

        # get limits from electrodes
        limits_elec = np.array([[np.min(all_vert[i, :]), np.max(all_vert[i, :])] for i in XYZ])

        # replace None limits with electrode limits
        limits[np.where(limits is None)] = limits_elec[np.where(limits is None)]

        res = np.array([res]).ravel()
        _n = np.array((limits[:, 1] - limits[:, 0]) / res, int) + 1

        # Recalculate resolution to match integer n's
        _d = (limits[:, 1] - limits[:, 0]) / _n

        # Generate a full mesh to be indexed later
        _r = np.array([np.linspace(limits[i, 0], limits[i, 1], _n[i]) for i in XYZ])
        mesh = np.meshgrid(_r[X], _r[Y], _r[Z], indexing='ij')  # type: np.ndarray

        # Initialize potential array
        pot = np.zeros(mesh[0].shape)

        # Index borders (can be float)
        borders = np.array([np.linspace(0, _n[i], domain_decomp[i] + 1) for i in XYZ])
        # Indices (must be int)
        start_idxs = np.array([np.array(borders[i][:-1], int) - overlap for i in XYZ])
        end_idxs = np.array([np.array(borders[i][1:], int) + overlap for i in XYZ])

        for i in XYZ:
            start_idxs[i][0] = 0
            end_idxs[i][-1] = int(borders[i][-1])

        # Print out domain information
        if self._debug:
            print("Potential Calculation. "
                  "Grid spacings: ({:.4f}, {:.4f}, {:.4f}), number of meshes: {}".format(_d[0], _d[1], _d[2], _n))
            print("Number of Subdomains: {}, "
                  "Domain decomposition {}:".format(np.product(domain_decomp), domain_decomp))

            for i, dirs in enumerate(["x", "y", "z"]):
                print("{}: Indices {} to {}".format(dirs, start_idxs[i], end_idxs[i] - 1))

        # Iterate over all the dimensions, calculate the subset of e-field
        domain_idx = 1
        for x1, x2 in zip(start_idxs[X], end_idxs[X]):
            for y1, y2 in zip(start_idxs[Y], end_idxs[Y]):
                for z1, z2 in zip(start_idxs[Z], end_idxs[Z]):

                    print("[{}] Domain {}/{}, "
                          "Index Limits: x = ({}, {}), "
                          "y = ({}, {}), "
                          "z = ({}, {})".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - _ts))),
                                                domain_idx,
                                                np.product(domain_decomp),
                                                x1, x2 - 1, y1, y2 - 1, z1, z2 - 1))

                    grid_pts = np.vstack([_mesh[x1:x2, y1:y2, z1:z2].ravel() for _mesh in mesh])

                    sl_pot = bempp.api.operators.potential.laplace.single_layer(fsp, grid_pts)
                    _pot = sl_pot * sol
                    pot[x1:x2, y1:y2, z1:z2] = _pot.reshape([x2 - x1, y2 - y1, z2 - z1])

                    domain_idx += 1

                    del grid_pts
                    del sl_pot
                    del _pot

        self._variables_bempp["ef_phi"] = pot -  3.0 * self._voltage

        return 0

    def plot_combo(self, xypos=0.000, xyscale=1.0, zlim=None):

        assert self._variables_bempp["ef_itp"] is not None, "No E-Field calculated yet!"

        numpts = 5000

        if zlim is None:
            zmin = np.min(self._variables_bempp["rf_itp"]._field["z"].grid[2])  # TODO: Field() should have limits
            zmax = np.min(self._variables_bempp["rf_itp"]._field["z"].grid[2])
        else:
            zmin, zmax = zlim

        # Bz of z at x = y = 0
        x = np.zeros(numpts)
        y = np.zeros(numpts)
        z = np.linspace(zmin, zmax, numpts)

        points = np.vstack([x, y, z]).T

        _, _, ez = self._variables_bempp["ef_itp"](points)

        plt.plot(z, ez, color=colors[0], label="$E_z$")

        # Bx of z at x = 0.005,  y = 0
        x = np.ones(numpts) * xypos

        points = np.vstack([x, y, z]).T

        ex, _, _ = self._variables_bempp["ef_itp"](points)

        plt.plot(z, xyscale * ex,
                 color=colors[1],
                 label="$\mathrm{{E}}_\mathrm{{x}}$ at x = {} m".format(xypos))

        # By of z at x = 0.0,  y = 0.005
        x = np.zeros(numpts)
        y = np.ones(numpts) * xypos

        points = np.vstack([x, y, z]).T

        _, ey, _ = self._variables_bempp["ef_itp"](points)

        plt.plot(z, xyscale * ey,
                 color=colors[2],
                 label="$\mathrm{{E}}_\mathrm{{y}}$ at y = {} m".format(xypos))

        plt.xlabel("z (m)")
        plt.ylabel("Field (V/m)")

        plt.legend(loc=2)

        plt.show()

    def get_phi(self):

        return self._variables_bempp["ef_phi"]

    def generate_full_mesh(self):

        assert self._vanes is not None, "No vanes generated yet, cannot mesh..."

        # Initialize empty arrays of the correct shape (3 x n)
        vertices = np.zeros([3, 0])
        elements = np.zeros([3, 0])
        vertex_counter = 0
        domains = np.zeros([0], int)

        # For now, do this only on the first node
        if rank == 0:

            for _vane in self._vanes:
                # noinspection PyCallingNonCallable
                mesh = generate_from_string(_vane.get_parameter("gmsh_str"))

                _vertices = mesh.leaf_view.vertices
                _elements = mesh.leaf_view.elements
                _domain_ids = mesh.leaf_view.domain_indices

                vertices = np.concatenate((vertices, _vertices), axis=1)
                elements = np.concatenate((elements, _elements + vertex_counter), axis=1)
                domains = np.concatenate((domains, _domain_ids), axis=0)

                # Increase the running counters
                vertex_counter += _vertices.shape[1]

            if self._variables_bempp["add_cyl"]:
                zmin = 0.0 - self._variables_bempp["cyl_gap"]
                zmax = self._length + self._variables_bempp["cyl_gap"]
                rmax = self._variables_bempp["cyl_id"] / 2.0

                cyl_gmsh_str = """Geometry.NumSubEdges = 100; // nicer display of curve
                                Mesh.CharacteristicLengthMax = {};
                                h = {};
                                rmax = {};
                                zmin = {};
                                zmax = {};
                                len = zmax - zmin;

                                """.format(0.025, 0.025, rmax, zmin, zmax)  # TODO: Make this a variable (mesh size)
                cyl_gmsh_str += """Point(1) = { 0, 0, zmin, h };

                                Point(2) = {rmax,0,zmin,h};
                                Point(3) = {0,rmax,zmin,h};
                                Point(4) = {-rmax,0,zmin,h};
                                Point(5) = {0,-rmax,zmin,h};

                                Circle(1) = {2,1,3};
                                Circle(2) = {3,1,4};
                                Circle(3) = {4,1,5};
                                Circle(4) = {5,1,2};

                                Line Loop(5) = {1,2,3,4};
                                Plane Surface(6) = {5};

                                out[] = Extrude{0, 0, len} { Surface {6}; };

                                Physical Surface(100) = {6, out[]};

                                """
                # noinspection PyCallingNonCallable
                if self._debug:
                    with open("cyl_str.geo", "w") as _of:
                        _of.write(cyl_gmsh_str)

                mesh = generate_from_string(cyl_gmsh_str)

                _vertices = mesh.leaf_view.vertices
                _elements = mesh.leaf_view.elements
                _domain_ids = mesh.leaf_view.domain_indices

                vertices = np.concatenate((vertices, _vertices), axis=1)
                elements = np.concatenate((elements, _elements + vertex_counter), axis=1)
                domains = np.concatenate((domains, _domain_ids), axis=0)

            elif self._variables_bempp["add_endplates"]:

                zmin = 0.0 - self._variables_bempp["cyl_gap"]
                zmax = self._length + self._variables_bempp["cyl_gap"]
                rmax = self._variables_bempp["cyl_id"] / 2.0

                cyl_gmsh_str = """Geometry.NumSubEdges = 100; // nicer display of curve
                                Mesh.CharacteristicLengthMax = {};
                                h = {};
                                rmax = {};
                                zmin = {};
                                zmax = {};
                                len = zmax - zmin;
                                            """.format(0.005, 0.005, rmax, zmin, zmax)  # TODO: Make this a variable (mesh size)
                cyl_gmsh_str += """
                                Point(1) = { 0, 0, zmin, h };
                                Point(2) = {rmax,0,zmin,h};
                                Point(3) = {0,rmax,zmin,h};
                                Point(4) = {-rmax,0,zmin,h};
                                Point(5) = {0,-rmax,zmin,h};

                                Circle(1) = {2,1,3};
                                Circle(2) = {3,1,4};
                                Circle(3) = {4,1,5};
                                Circle(4) = {5,1,2};

                                Line Loop(5) = {1,2,3,4};
                                Plane Surface(6) = {5};

                                entrance_plate[] = Extrude{0, 0, -0.005} { Surface {6}; };

                                Point(106) = { 0, 0, zmax, h };
                                Point(107) = {rmax,0,zmax,h};
                                Point(108) = {0,rmax,zmax,h};
                                Point(109) = {-rmax,0,zmax,h};
                                Point(110) = {0,-rmax,zmax,h};

                                Circle(107) = {107,106,108};
                                Circle(108) = {108,106,109};
                                Circle(109) = {109,106,110};
                                Circle(110) = {110,106,107};

                                Line Loop(111) = {-107,-108,-109,-110};
                                Plane Surface(112) = {111};

                                exit_plate[] = Extrude{0, 0, 0.005} { Surface {112}; };

                                Physical Surface(100) = {6, 112, -entrance_plate[], -exit_plate[]};
                                """
                # noinspection PyCallingNonCallable
                if self._debug:
                    with open("cyl_str.geo", "w") as _of:
                        _of.write(cyl_gmsh_str)

                mesh = generate_from_string(cyl_gmsh_str)

                _vertices = mesh.leaf_view.vertices
                _elements = mesh.leaf_view.elements
                _domain_ids = mesh.leaf_view.domain_indices

                vertices = np.concatenate((vertices, _vertices), axis=1)
                elements = np.concatenate((elements, _elements + vertex_counter), axis=1)
                domains = np.concatenate((domains, _domain_ids), axis=0)

            mpi_data = {"vert": vertices,
                        "elem": elements,
                        "doma": domains}

        else:

            mpi_data = None

        mpi_data = comm.bcast(mpi_data, root=0)

        self._full_mesh = bempp.api.grid.grid_from_element_data(mpi_data["vert"],
                                                                mpi_data["elem"],
                                                                mpi_data["doma"])

        if self._debug:
            if rank == 0:
                self._full_mesh.plot()

        return 0

    def solve_bempp(self):

        if self._full_mesh is None:
            print("Please generate a mesh before solving with BEMPP!")
            return 1

        dp0_space = bempp.api.function_space(self._full_mesh, "DP", 0)
        slp = bempp.api.operators.boundary.laplace.single_layer(dp0_space, dp0_space, dp0_space)

        domain_mapping = {100: self._variables_bempp["pot_shift"]}  # 100 is ground
        for vane in self._vanes:
            domain_mapping[vane.domain_idx] = vane.voltage

        def f(*args):
            domain_index = args[2]
            result = args[3]
            result[0] = domain_mapping[domain_index]

        dirichlet_fun = bempp.api.GridFunction(dp0_space, fun=f)

        if self._debug:
            if rank == 0:
                dirichlet_fun.plot()
                bempp.api.export(grid_function=dirichlet_fun, file_name="dirichlet_func.msh")

        # Solve
        sol, info = bempp.api.linalg.gmres(slp, dirichlet_fun, tol=1e-5, use_strong_form=True)

        self._variables_bempp["solution"] = sol
        self._variables_bempp["f_space"] = dp0_space

        # self._variables_bempp["operator"] = slp
        # self._variables_bempp["grid_fun"] = dirichlet_fun

        # Quick test: plot potential across RFQ center
        # numpoints = 201
        # _x = np.linspace(-0.04, 0.04, numpoints)
        # _y = np.linspace(-0.04, 0.04, numpoints)
        # mesh = np.meshgrid(_x, _y, 0.5, indexing='ij')  # type: np.ndarray
        # grid_pts = np.vstack([_mesh.ravel() for _mesh in mesh])
        # sl_pot = bempp.api.operators.potential.laplace.single_layer(dp0_space, grid_pts)
        # _pot = sl_pot * sol
        # _pot = _pot.reshape([numpoints, numpoints])

        # Apply mask where electrodes are (TEMP!!!)
        # import numpy.ma as ma
        # epsilon = 0.012 * self._voltage
        # masked_pot = ma.masked_array(_pot, mask=(np.abs(_pot) >= (self._voltage-epsilon)))

        # dx = 0.08 / (numpoints - 1)

        # ex, ey = np.gradient(masked_pot, dx, dx)

        # plt.imshow(masked_pot.T, extent=(-0.04, 0.04, -0.04, 0.04))
        # plt.xlabel("x (m)")
        # plt.ylabel("y (m)")
        # plt.title("Potential (V)")
        # plt.colorbar()
        # plt.show()

        # plt.imshow(ex.T, extent=(-0.04, 0.04, -0.04, 0.04))
        # plt.xlabel("x (m)")
        # plt.ylabel("y (m)")
        # plt.title("E_x (V/m)")
        # plt.colorbar()
        # plt.show()

        # plt.imshow(ey.T, extent=(-0.04, 0.04, -0.04, 0.04))
        # plt.xlabel("x (m)")
        # plt.ylabel("y (m)")
        # plt.title("E_y (V/m)")
        # plt.colorbar()
        # plt.show()

        # plt.quiver(mesh[0].flatten(), mesh[1].flatten(), ex.flatten(), ey.flatten(), scale=None)
        # plt.show()

        # exit()

        return 0

    def generate_vanes(self):

        assert len(self._cells) > 0, "No cells have been added, no vanes can be generated."

        # There are four vanes (rods) in the RFQ
        # x = horizontal, y = vertical, with p, m denoting positive and negative axis directions
        # for vane_type in ["yp", "ym"]:
        for vane_type in ["yp"]:
            self._vanes.append(PyRFQVane(vane_type=vane_type,
                                         cells=self._cells,
                                         voltage=self._voltage + self._variables_bempp["pot_shift"],
                                         debug=self._debug))

        # for vane_type in ["xp", "xm"]:
        for vane_type in ["xp"]:
            self._vanes.append(PyRFQVane(vane_type=vane_type,
                                         cells=self._cells,
                                         voltage=-self._voltage + self._variables_bempp["pot_shift"],
                                         debug=self._debug))

        # Generate the two vanes in parallel:
        p = Pool()
        self._vanes = p.map(self.generate_vanes_worker, self._vanes)

        return 0

    def generate_vanes_worker(self, vane):

        dx_h = self._variables_bempp["grid_res"]  # TODO: Is there a reason to set them to different values?

        vane.calculate_profile(fudge=True)
        vane.generate_gmsh_str(dx=dx_h, h=dx_h,
                               symmetry=False, mirror=True)

        return vane

    def plot_vane_profile(self):

        assert len(self._vanes) != 0, "No vanes calculated yet!"

        _fig, _ax = plt.subplots()

        for vane in self._vanes:
            if vane.vane_type == "xp":
                z, x = vane.get_profile(nz=10000)
                _ax.plot(z, x, color=colors[0], label="x-profile")
                print("X Vane starting point", z[0], x[0])
            if vane.vane_type == "yp":
                z, y = vane.get_profile(nz=10000)
                _ax.plot(z, -y, color=colors[1], label="y-profile")
                print("Y Vane starting point", z[0], y[0])

        plt.xlabel("z (m)")
        plt.ylabel("x/y (m)")

        plt.legend(loc=1)

        plt.show()

    def print_cells(self):
        for number, cell in enumerate(self._cells):
            print("RFQ Cell {}: ".format(number + 1), cell)

        return 0

    def write_inventor_macro(self,
                             save_folder=None,
                             **kwargs):

        """
        This function writes out the vane profiles for X and Y and Inventor VBA macros that can
        be run immediately to generate 3D solid models in Autodesk Inventor (c).
        kwargs:
        vane_type: one of 'rod', 'vane', default is 'vane' TODO: Only 'vane' implemented as of now.
        vane_radius: radius of curvature of circular vanes, default is 0.005 m TODO: add hyperbolic vanes
        vane_height: height of a single vane either from the minimum point (vane_height_type = 'relative')
                            or from the center of the RFQ (vane_height_type = 'absolute')
                            default is 0.05 m
        vane_height_type: see above. default is absolute
        nz: number of points to use for spline in z direction. default is 500.
        :param save_folder: If None, a prompt is opened
        :return:
        """

        # TODO: assert that height and absolute/relative combination work out geometrically
        # TODO: with the amplitude ofthe modulations (i.e. no degenerate geometry)

        for key, value in kwargs.items():
            assert key in self._variables_inventor.keys(), "write_inventor_macro: Unrecognized kwarg '{}'".format(key)
            self._variables_inventor[key] = value

        assert self._variables_inventor["vane_type"] != "rod", "vane_type == 'rod' not implemented yet. Aborting"

        if save_folder is None:

            fd = FileDialog()
            save_folder, _ = fd.get_filename('folder')

            if save_folder is None:
                return 0

        for direction in ["X", "Y"]:

            # Generate text for Inventor macro
            header_text = """Sub CreateRFQElectrode{}()
                            Dim oApp As Application
                            Set oApp = ThisApplication

                            ' Get a reference to the TransientGeometry object.
                            Dim tg As TransientGeometry
                            Set tg = oApp.TransientGeometry

                            Dim oPart As PartDocument
                            Dim oCompDef As PartComponentDefinition
                            Dim oSketch3D As Sketch3D
                            Dim oSpline As SketchSpline3D
                            Dim vertexCollection1 As ObjectCollection

                        """.format(direction)

            electrode_text = """
                            Set oPart = oApp.Documents.Add(kPartDocumentObject, , True)
                            Set oCompDef = oPart.ComponentDefinition
                            Set oSketch3D = oCompDef.Sketches3D.Add
                            Set vertexCollection1 = oApp.TransientObjects.CreateObjectCollection(Null)

                            FileName = "{}"
                            fileNo = FreeFile 'Get first free file number

                            Dim minHeight As Double
                            minHeight = 10000  'cm, large number

                            Open FileName For Input As #fileNo
                            Do While Not EOF(fileNo)

                                Dim strLine As String
                                Line Input #1, strLine

                                strLine = Trim$(strLine)

                                If strLine <> "" Then
                                    ' Break the line up, using commas as the delimiter.
                                    Dim astrPieces() As String
                                    astrPieces = Split(strLine, ",")
                                End If

                                Call vertexCollection1.Add(tg.CreatePoint(astrPieces(0), astrPieces(1), astrPieces(2)))

                                ' For X vane this is idx 0, for y vane it is idx 1
                                If CDbl(astrPieces({})) < minHeight Then
                                    minHeight = CDbl(astrPieces({}))
                                End If

                            Loop

                            Close #fileNo

                            Set oSpline = oSketch3D.SketchSplines3D.Add(vertexCollection1)

                            """.format(os.path.join(save_folder, "Vane_{}.txt".format(direction)), AXES[direction], AXES[direction])

            sweep_text = """
                                ' Now make a sketch to be swept
                                ' Start with a work plane
                                Dim oWP As WorkPlane
                                Set oWP = oCompDef.WorkPlanes.AddByNormalToCurve(oSpline, oSpline.StartSketchPoint)
                                
                                ' Add a 2D sketch
                                Dim oSketch2D As PlanarSketch
                                Set oSketch2D = oCompDef.Sketches.Add(oWP)
                            """
            if direction == "X":
                sweep_text += """
                                ' Make sure the orientation of the sketch is correct
                                ' We want the sketch x axis oriented with the lab y axis for X vane
                                oSketch2D.AxisEntity = oCompDef.WorkAxes.Item(2)
                            """
            else:
                sweep_text += """
                                ' Make sure the orientation of the sketch is correct
                                ' We want the sketch x axis oriented with the lab y axis for X vane
                                oSketch2D.AxisEntity = oCompDef.WorkAxes.Item(1)
                                ' Also, we need to flip the axis for Y vanes
                                oSketch2D.NaturalAxisDirection = False
                            """
            sweep_text += """
                                ' Draw the half circle and block
                                Dim radius As Double
                                Dim height As Double
                                
                                radius = {}  'cm
                                height = {}  'cm
                                
                                Dim oOrigin As SketchEntity
                                Set oOrigin = oSketch2D.AddByProjectingEntity(oSpline.StartSketchPoint)
                                """.format(self._variables_inventor["vane_radius"] * 100.0,
                                       self._variables_inventor["vane_height"] * 100.0)

            sweep_text += """
                                Dim oCenter As Point2d
                                Set oCenter = tg.CreatePoint2d(oOrigin.Geometry.X, oOrigin.Geometry.Y - radius)
                                
                                Dim oCirc1 As Point2d
                                Set oCirc1 = tg.CreatePoint2d(oOrigin.Geometry.X - radius, oOrigin.Geometry.Y - radius)
                                
                                Dim oCirc2 As Point2d
                                Set oCirc2 = tg.CreatePoint2d(oOrigin.Geometry.X + radius, oOrigin.Geometry.Y - radius)
                                
                                Dim arc As SketchArc
                                Set arc = oSketch2D.SketchArcs.AddByThreePoints(oCirc1, oOrigin.Geometry, oCirc2)
                                
                                """
            sweep_text += """
                                Dim line1 As SketchLine
                                Set line1 = oSketch2D.SketchLines.AddByTwoPoints(arc.EndSketchPoint, arc.StartSketchPoint)

                                ' Create a Path
                                Dim oPath As Path
                                Set oPath = oCompDef.Features.CreatePath(oSpline)
                                
                                ' Create a profile.
                                Dim oProfile As Profile
                                Set oProfile = oSketch2D.Profiles.AddForSolid
                                
                                ' Create the sweep feature.
                                Dim oSweep As SweepFeature
                                Set oSweep = oCompDef.Features.SweepFeatures.AddUsingPath(oProfile, oPath, kJoinOperation)
                                """

                                        # Small modification depending on absolute or relative vane height:
            if self._variables_inventor["vane_height_type"] == 'relative':
                sweep_text += """
                                ' Create another work plane above the vane
                                Dim oWP2 As WorkPlane
                                Set oWP2 = oCompDef.WorkPlanes.AddByPlaneAndOffset(oCompDef.WorkPlanes.Item({}), minHeight + height)
                                """.format(AXES[direction] + 1)  # X is 0 and Y is 1, but the correct plane indices are 1 and 2
            else:
                sweep_text += """
                                ' Create another work plane above the vane
                                Dim oWP2 As WorkPlane
                                Set oWP2 = oCompDef.WorkPlanes.AddByPlaneAndOffset(oCompDef.WorkPlanes.Item({}), height)
                                """.format(AXES[direction] + 1)  # X is 0 and Y is 1, but the correct plane indices are 1 and 2

            sweep_text += """
                                ' Start a sketch
                                Set oSketch2D = oCompDef.Sketches.Add(oWP2)
                                
                                ' Project the bottom face of the sweep
                                ' (start and end face might be tilted and contribute)
                                ' at this point I don't know how Inventor orders the faces, 2 is my best guess but
                                ' might be different occasionally... -DW
                                Dim oEdge As Edge
                                For Each oEdge In oSweep.SideFaces.Item(2).Edges
                                    Call oSketch2D.AddByProjectingEntity(oEdge)
                                Next

                                ' Create a profile.
                                Set oProfile = oSketch2D.Profiles.AddForSolid

                                ' Extrude
                                Dim oExtDef As ExtrudeDefinition
                                Dim oExt As ExtrudeFeature
                                Set oExtDef = oCompDef.Features.ExtrudeFeatures.CreateExtrudeDefinition(oProfile, kJoinOperation)
                                Call oExtDef.SetToNextExtent(kNegativeExtentDirection, oSweep.SurfaceBody)
                                Set oExt = oCompDef.Features.ExtrudeFeatures.Add(oExtDef)

                                ' Repeat but cutting in the up-direction
                                ' Extrude
                                Set oExtDef = oCompDef.Features.ExtrudeFeatures.CreateExtrudeDefinition(oProfile, kCutOperation)
                                Call oExtDef.SetThroughAllExtent(kPositiveExtentDirection)
                                Set oExt = oCompDef.Features.ExtrudeFeatures.Add(oExtDef)
                                """

            footer_text =   """
                                oPart.UnitsOfMeasure.LengthUnits = kMillimeterLengthUnits

                                ThisApplication.ActiveView.Fit

                            End Sub
                            """

            # Write the Autodesk Inventor VBA macros:
            with open(os.path.join(save_folder, "Vane_{}.ivb".format(direction)), "w") as outfile:

                outfile.write(header_text + electrode_text + sweep_text + footer_text)

            # Write the vane profile files:
            with open(os.path.join(save_folder, "Vane_{}.txt".format(direction)), "w") as outfile:

                if direction == "X":
                    for vane in self._vanes:
                        if vane.vane_type == "xp":
                            z, x = vane.get_profile(nz=self._variables_inventor["nz"])
                            min_x = np.min(x)
                            max_x = np.max(x)
                            z_start = np.min(z)
                            z_end = np.max(z)

                    for _x, _z in zip(x, z):
                        outfile.write("{:.6f}, {:.6f}, {:.6f}\n".format(
                            _x * 100.0,  # For some weird reason Inventor uses cm as default...
                            0.0,
                            _z * 100.0))

                else:
                    for vane in self._vanes:
                        if vane.vane_type == "yp":
                            z, y = vane.get_profile(nz=self._variables_inventor["nz"])
                            min_y = np.min(y)
                            max_y = np.max(y)

                    for _y, _z in zip(y, z):
                        outfile.write("{:.6f}, {:.6f}, {:.6f}\n".format(
                            0.0,
                            _y * 100.0,  # For some weird reason Inventor uses cm as default...
                            _z * 100.0))

            # Write an info file with some useful information:
            with open(os.path.join(save_folder, "Info.txt"), "w") as outfile:

                datestr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                outfile.write("Inventor Macros and Profile generated on {}\n\n".format(datestr))

                outfile.write("Parameters:\n")
                for key, value in self._variables_inventor.items():
                    outfile.write("{}: {}\n".format(key, value))

                if self._variables_inventor["vane_height_type"] == 'absolute':
                    max_extent_x = max_extent_y = self._variables_inventor["vane_height"]
                else:
                    max_extent_x = self._variables_inventor["vane_height"] + min_x
                    max_extent_y = self._variables_inventor["vane_height"] + min_y

                outfile.write("\nOther useful values:\n")
                outfile.write("Maximum Extent in X: {} m\n".format(max_extent_x))
                outfile.write("Maximum Extent in Y: {} m\n".format(max_extent_y))
                outfile.write("Z Start: {} m\n".format(z_start))
                outfile.write("Z End: {} m\n".format(z_end))

        return 0

    def install(self):
        # Parameters: None
        # Returns: None
        # Installs the field and conductors into Warp
        if self._from_cells:
            if self._twoterm:
                self._field.generate_field_from_cells_tt()
            elif (self._boundarymethod):
                self.generate_field_bempp()

        

        self._sim_end = self._field._zmax + self.sim_end_buffer

        self._fieldzmax  = self._field._zmax
        self.import_field()
        
        self.create_vanes()

    def generate_field_bempp(self):
        self.set_bempp_parameter("add_endplates", self.add_endplates)
        self.set_bempp_parameter("cyl_id", self.cyl_id)
        self.set_bempp_parameter("grid_res", self.grid_res_bempp)
        self.set_bempp_parameter("pot_shift", self.pot_shift)

        print("Generating vanes")
        ts = time.time()
        self.generate_vanes()
        print("Generating vanes took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))

        print("Generating full mesh for BEMPP")
        ts = time.time()
        self.generate_full_mesh()
        print("Meshing took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))

        print("Solving BEMPP problem")
        ts = time.time()
        self.solve_bempp()
        print("Solving BEMPP took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))

        print("Calculating Potential")
        ts = time.time()
        myres = [self.resolution, self.resolution, self.resolution]

        self.calculate_potential(limits=((self.xy_limits[0], self.xy_limits[1]),
                                         (self.xy_limits[2], self.xy_limits[3]),
                                         (self.z_limits[0], self.z_limits[1])),
                                  res=myres,
                                  domain_decomp=(1, 1, 50),
                                  overlap=0)
        print("Potential took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))

        ##########################
        # TODO: PARSE FIELD HERE #
        ##########################
        exit(1)

    def setup(self):
        # Parameters: None
        # Returns: None
        # Evaluates user parameters to ensure all dependencies are provided
        # Calculates and/or loads the field into the field class

        if (not self._filename):
            print("Please provide a file. Exiting")
            exit(1)
        if (not self.vane_radius) and (self.simple_rods):
            print("Please specify the vane radius (vane_radius) for the simple rod structure. Exiting.")
            exit(1)
        if (not self.vane_distance):
            print("Please specify the vane distance (vane_distance) from axis. Exiting.")
            exit(1)
        if (not self.rf_freq):
            print("The RF frequency (rf_freq) must be specified. Exiting")
            exit(1)
        if (self._from_cells and self._twoterm):
            if self._twoterm:
                print("Resolution is {}".format(self.resolution))
                if (not self.xy_limits) or (np.shape(self.xy_limits) != (4,)):
                    print("Please set XY limits (xy_limits) in the form of a list [xmin, xmax, ymin, ymax]")
                    exit(1)
                if (not self._voltage):
                    print("Please set vane voltage (voltage) for two term potential calculation")
                    exit(1)
                elif (not self.tt_a_init):
                    print("Please set initial aperture (tt_a_init) for two term potential calculation")
                    exit(1)
            if self._boundarymethod:
                if (not self.cyl_id):
                    print("Please set [asdfasdfasdf] (cyl_id) for boundary method calculation")
                    exit(1)
                if (not self.grid_res_bempp):
                    print("Please set boundary method grid resolution (grid_res_bempp)")
                    exit(1)
                if (not self.pot_shift):
                    print("Please set potential shift (pot_shift) for boundary method.")
                    exit(1)
                if (not self.z_limits):
                    print("Please provide z limits (z_limits) for BEMPP calculation.")
                    exit(1)

        if self._from_cells:
            if (self._twoterm):
                self._field.load_field_from_cells_tt(self._voltage, 
                                                  self.rf_freq,
                                                  self.tt_a_init,
                                                  self.xy_limits,
                                                  self._filename, 
                                                  resolution=self.resolution,
                                                  ignore_rms=self.ignore_rms)
            elif (self._boundarymethod):
                if (self.add_cells_from_file(filename=self._filename, ignore_rms=self.ignore_rms)==1):
                    print("Something went wrong. Please check your file.")
                    exit(1)

        else:
            self._field.load_field_from_file(self._filename)

    def import_field(self):
        # import_field
        # Parameters: none
        # Returns: none
        # Loads the appropriate field into Warp simulation.

        def fieldscaling(time):
            val = np.cos(time * 2 * np.pi * self.rf_freq)
            self._ray.append(val)
            return val*2 

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

        length = self._field._z_length
        self._length = length
        zcent  = (self._field._z_length / 2.0) + abs(self.zstart)

        outer_shell = ZCylinderOut(self.vane_distance + 0.02, (self._sim_end - self.sim_start), zcent=(self._sim_end + self.sim_start)/2)
        total_conductors = outer_shell

        if (self.simple_rods):
            rod1 = ZCylinder(self.vane_radius, length, zcent=zcent, xcent=self.vane_distance)
            rod2 = ZCylinder(self.vane_radius, length, zcent=zcent, xcent=-self.vane_distance) 
            rod3 = ZCylinder(self.vane_radius, length, zcent=zcent, ycent=self.vane_distance)
            rod4 = ZCylinder(self.vane_radius, length, zcent=zcent, ycent=-self.vane_distance)

            total_conductors += rod1 + rod2 + rod3 + rod4

        if (self.endplates):
            firstendplate = ZCylinderOut(self.endplates_inner_diameter, self.endplates_thickness, zcent=self._field._zmin-self.endplates_distance_from_vanes)
            endendplate = ZCylinderOut(self.endplates_inner_diameter, self.endplates_thickness, zcent=self._field._zmax+self.endplates_distance_from_vanes)
            total_conductors += firstendplate + endendplate

        installconductor(total_conductors)
        scraper = ParticleScraper(total_conductors)

        self._conductors = total_conductors

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

        # plotegrd(component="x", iz=50)
        # fma()
        #plotegrd(component="y", iz=50)
        #fma()
    
    def add_cell(self,
                 cell_type,
                 aperture,
                 modulation,
                 length,
                 flip_z=False,
                 shift_cell_no=False):

        if (self._twoterm):
            self._field.add_cell(cell_type, aperture, modulation, length, flip_z, shift_cell_no)
        elif (self._boundarymethod):
            self.append_cell(cell_type, aperture, modulation, length, flip_z=flip_z, shift_cell_no=shift_cell_no)


if __name__ == "__main__":

    myrfq = PyRFQ(voltage=22000.0, debug=True)

    # myrfq.append_cell(cell_type="STA",
    #                   aperture=0.15,
    #                   modulation=1.0,
    #                   length=0.0)

    # Load the base RFQ design from the parmteq file
    if myrfq.add_cells_from_file(filename="input/PARMTEQOUT.TXT", ignore_rms=True) == 1:
        exit()

    myrfq.append_cell(cell_type="TCS",
                      aperture=0.007147,
                      modulation=1.6778,
                      length=0.033840)

    myrfq.append_cell(cell_type="DCS",
                      aperture=0.0095691183,
                      modulation=1.0,
                      length=0.1)

    myrfq.set_bempp_parameter("add_endplates", True)
    myrfq.set_bempp_parameter("cyl_id", 0.1)
    myrfq.set_bempp_parameter("grid_res", 0.005)
    myrfq.set_bempp_parameter("pot_shift", 3.0 * 22000.0)

    print(myrfq)

    print("Generating vanes")
    ts = time.time()
    myrfq.generate_vanes()
    print("Generating vanes took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))

    # if rank == 0:
    #     myrfq.plot_vane_profile()
    #     myrfq.write_inventor_macro(vane_type='vane',
    #                                vane_radius=0.005,
    #                                vane_height=0.15,
    #                                vane_height_type='absolute',
    #                                nz=600)
    # exit()

    print("Generating full mesh for BEMPP")
    ts = time.time()
    myrfq.generate_full_mesh()
    print("Meshing took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))

    # input("Hit enter to continue...")

    print("Solving BEMPP problem")
    ts = time.time()
    myrfq.solve_bempp()
    print("Solving BEMPP took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))

    # input("Hit enter to continue...")

    print("Calculating Potential")
    ts = time.time()
    myres = [0.002, 0.002, 0.002]
    limit = 0.02
    myrfq.calculate_potential(limits=((-limit, limit), (-limit, limit), (-0.1, 1.35)),
                              res=myres,
                              domain_decomp=(1, 1, 50),
                              overlap=0)
    print("Potential took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))

    myrfq.calculate_efield()

    # import pickle

    # with open("ef_phi.field", "wb") as outfile:
    #     pickle.dump(myrfq.get_phi(), outfile)

    # import numpy.ma as ma
    #
    # epsilon = 0.012 * 22000
    # masked_pot = ma.masked_array(mypot, mask=(np.abs(mypot) >= (22000 - epsilon)))
    #
    # plt.imshow(masked_pot[int(0.5 * pot.shape[0]), :, :].T, extent=(-0.1, 1.35, -0.015, 0.015))
    # plt.xlabel("x (m)")
    # plt.ylabel("y (m)")
    # plt.title("Potential (V)")
    # plt.colorbar()
    # plt.show()
    #
    # exit()
    #
    # if rank == 0:
    #     myrfq.plot_combo(xypos=0.005, xyscale=1.0, zlim=(-0.1, 1.35))
    
    import pickle
    
    with open("efield_out.field", "wb") as outfile:
        pickle.dump(myrfq._variables_bempp["ef_itp"], outfile)

    # myrfq.plot_vane_profile()
