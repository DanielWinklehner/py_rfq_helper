import numpy as np
import scipy.constants as const
from multiprocessing import Pool
from scipy.interpolate import interp1d
# from scipy import meshgrid
from scipy.special import iv as bessel1
from scipy.optimize import root
# import pickle
# import scipy.constants as const
# import numpy as np
# import platform
# import matplotlib.pyplot as plt
# import gc
import datetime
import time

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
if rank == 0:
    from dans_pymodules import *

    colors = MyColors()
else:
    colors = None

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
        self._aperture = np.round(aperture, decimals)
        self._modulation = np.round(modulation, decimals)
        self._length = np.round(length, decimals)
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
                 debug=False):

        self._debug = debug

        self._type = vane_type
        self._cells = cells
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

    def calculate_mesh(self, dx=None):

        if dx is not None:
            self._mesh_params["dx"] = dx
        else:
            dx = self._mesh_params["dx"]

        zmin = 0.0  # For now, RFQ vanes always start at 0.0, everything is relative
        zmax = self._length
        numz = np.round((zmax - zmin) / dx, 0) + 1

        r_tip = self._mesh_params["r_tip"]
        h_block = self._mesh_params["h_block"]

        # Calculate z_data and vane profile:
        z, vane = self.get_profile(nz=numz)

        if self._mesh_params["tip"] == "semi-circle":
            # Calculate approximate angular resolution corresponding to desired mesh size
            num_phi = np.round(r_tip * np.pi / dx, 0)
            phi = np.pi / num_phi

            print("With mesh_size {} m, we have {} points per semi-circle".format(dx, num_phi))

            # We need two sets of phi values so that subsequent z positions form triangles rather than squares
            phi_set = [np.linspace(np.pi, 2.0 * np.pi, num_phi),
                       np.linspace(np.pi + 0.5 * phi, 2.0 * np.pi - 0.5 * phi, num_phi - 1)]

            # maximum vertical extent:
            ymax = r_tip + np.max(vane) + h_block
            print("Maximum vertical extent: {} m".format(ymax))
            # TODO: Think about whether it's a good idea to use only half the mesh size on block part
            num_vert_pts = int(np.round((ymax - r_tip - np.min(vane)) / 2.0 / dx, 0))
            print("Number of points in vertical direction: {}".format(num_vert_pts))
            num_horz_pts = int(np.round(2.0 * r_tip / 2.0 / dx, 0)) - 1  # Not counting corners
            horz_step_size = (2.0 * r_tip) / (num_horz_pts + 1)

            # Create point cloud for testing
            points = []
            for i, _z in enumerate(z):

                # Create curved surface points:
                for _phi in phi_set[int(i % 2.0)]:
                    points.append((r_tip * np.cos(_phi),
                                   r_tip * np.sin(_phi) + r_tip + vane[i],
                                   _z))

                # Create straight lines points:
                # Looking with z, we start at the right end of the curvature and go up-left-down
                vert_step = (ymax - r_tip - vane[i]) / num_vert_pts

                # If we are in phi_set 0, we start one step up, in phi set 1 a half step up
                for j in range(num_vert_pts):
                    points.append((r_tip, points[-1][1] + vert_step, _z))

                if i % 2.0 == 0.0:
                    points.pop(-1)

                # Append the corner point
                points.append((r_tip, ymax, _z))

                if i % 2.0 == 0.0:
                    for j in range(num_horz_pts + 1):
                        points.append((r_tip + 0.5 * horz_step_size - (j + 1) * horz_step_size, ymax, _z))
                else:
                    for j in range(num_horz_pts):
                        points.append((r_tip - (j + 1) * horz_step_size, ymax, _z))

                # Append the corner point
                points.append((-r_tip, ymax, _z))

                # Finish up by going down on the left side
                for j in range(num_vert_pts):
                    points.append((-r_tip, ymax - (j + 1) * vert_step + 0.5 * (i % 2.0) * vert_step, _z))

                if i % 2.0 == 0.0:
                    points.pop(-1)

            points = np.array(points, dtype=[("x", float), ("y", float), ("z", float)])

            # Apply rotation
            rot = rot_map[self._type]

            if rot != 0.0:
                angle = rot * np.pi / 180.0

                x_old = points["x"].copy()
                y_old = points["y"].copy()

                points["x"] = x_old * np.cos(angle) - y_old * np.sin(angle)
                points["y"] = x_old * np.sin(angle) + y_old * np.cos(angle)

                del x_old
                del y_old

            points_per_slice = len(np.where(points["z"] == z[0])[0])
            print("Points per slice = {}".format(points_per_slice))

            # For each point, there need to be lines created to neighbors
            elements = []
            for i, point in enumerate(points[:-points_per_slice]):
                # Each element contains the indices of the bounding vertices
                slice_no = int(i / points_per_slice)
                # print(i, slice_no, int(slice_no % 2))

                if slice_no % 2 == 0.0:

                    if (i + 1) % points_per_slice != 0.0:
                        elements.append([i, i + 1, i + points_per_slice])
                    else:
                        elements.append(
                            [i, i + points_per_slice, i + 1 - points_per_slice])

                    if i % points_per_slice != 0.0:
                        elements.append(
                            [i, i + points_per_slice - 1, i + points_per_slice])
                    else:
                        elements.append(
                            [i, i + points_per_slice, i + 2 * points_per_slice - 1])

                else:

                    if (i + 1) % points_per_slice != 0.0:
                        # Regular element
                        elements.append([i, i + 1, i + points_per_slice + 1])
                        elements.append([i, i + points_per_slice, i + points_per_slice + 1])
                    else:
                        # Last element of level
                        elements.append([i, i + 1 - points_per_slice, i + 1])
                        elements.append([i, i + 1, i + points_per_slice])

            vertices = np.array([points["x"], points["y"], points["z"]])
            elements = np.array(elements).T

            # Test bempp calculation for single vane
            if bempp is not None:
                # noinspection PyUnresolvedReferences
                self._mesh = bempp.api.grid.grid_from_element_data(vertices, elements)

                # if self._debug:
                #     self._mesh.plot()

            else:
                return 0

        else:
            print("The vane type '{}' is not (yet) implemented. Aborting.".format(self._mesh_params["tip"]))
            return 1
        return 0

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
        # count = 0
        for cell in self._cells:

            if cell.cell_type != "STA":
                _z_end = np.round(cum_len + cell.length, decimals)
                idx = np.where((z >= cum_len) & (z <= _z_end))

                # print("")
                # print("Cell # {}".format(count))
                # print("Cell extent: {} to {}".format(cum_len, _z_end))
                # print("z_lab = [{};{}]".format(z[idx][0], z[idx][-1]))
                # print("z_loc = [{};{}]".format(z[idx][0] - cum_len, z[idx][-1] - cum_len))

                vane[idx] = cell.profile(np.round(z[idx] - cum_len, decimals))

                cum_len = np.round(cum_len + cell.length, decimals)
                # count += 1

        return z, vane

    def plot_mesh(self):

        if bempp is None or self._mesh is None:
            print("Either BEMPP couldn't be loaded or there is not yet a mesh generated!")
            return 1

        self._mesh.plot()

        return 0


# noinspection PyUnresolvedReferences
class PyRFQ(object):
    def __init__(self, voltage, debug=False):

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

        return 0

    def read_input_vecc(self, filename, ignore_rms):

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

        return 0

    def calculate_efield(self):

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

            footer_text = """
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


if __name__ == "__main__":

    myrfq = PyRFQ(voltage=22000.0, debug=True)

    # myrfq.append_cell(cell_type="STA",
    #                   aperture=0.15,
    #                   modulation=1.0,
    #                   length=0.0)
    myrfq.append_cell(cell_type="DCS",
                      aperture=0.009289,
                      modulation=1.0,
                      length=0.04858)

    # Load the base RFQ design from the parmteq file
    if myrfq.add_cells_from_file(ignore_rms=True) == 1:
        exit()

    myrfq.append_cell(cell_type="TCS",
                      aperture=0.006837,
                      modulation=1.6778,
                      length=0.032727)

    myrfq.append_cell(cell_type="DCS",
                      aperture=0.0091540593,
                      modulation=1.0,
                      length=0.14)

    # myrfq.append_cell(cell_type="STA",
    #                   aperture=0.0095691183,
    #                   modulation=1.0,
    #                   length=0.0)
    #
    # myrfq.append_cell(cell_type="RMS",
    #                   aperture=0.010944,
    #                   modulation=1.0,
    #                   length=0.018339)
    #
    # myrfq.append_cell(cell_type="RMS",
    #                   aperture=0.016344,
    #                   modulation=1.0,
    #                   length=0.018339)
    #
    # myrfq.append_cell(cell_type="RMS",
    #                   aperture=0.041051,
    #                   modulation=1.0,
    #                   length=0.018339)
    #
    # myrfq.append_cell(cell_type="RMS",
    #                   aperture=0.150000,
    #                   modulation=1.0,
    #                   length=0.018339)

    # myrfq.add_cells_from_file(filename="/mnt/c/Users/Daniel Winklehner/Dropbox (MIT)/Code/Python/"
    #                                    "py_rfq_designer/py_rfq_designer/Parm_50_63cells.dat")

    # Add some more cells for transition, drift and re-bunching
    # myrfq.append_cell(cell_type="TCS",
    #                   aperture=0.011255045027294745,
    #                   modulation=1.6686390559337798,
    #                   length=0.0427)

    # myrfq.append_cell(cell_type="DCS",
    #                   aperture=0.015017826368066015,
    #                   modulation=0.9988129861386651,
    #                   length=0.13)
    #
    # myrfq.append_cell(cell_type="TCS",
    #                   aperture=0.01,
    #                   modulation=2.0,
    #                   length=0.041,
    #                   flip_z=True)
    #
    # myrfq.append_cell(cell_type="TCS",
    #                   aperture=0.01,
    #                   modulation=2.0,
    #                   length=0.041,
    #                   shift_cell_no=True)
    #
    # myrfq.append_cell(cell_type="RMS",
    #                   aperture=0.015,
    #                   modulation=1.0,
    #                   length=0.01828769716079613)
    #
    # myrfq.append_cell(cell_type="RMS",
    #                   aperture=0.015192005013805875,
    #                   modulation=1.0,
    #                   length=0.01828769716079613)
    #
    # myrfq.append_cell(cell_type="RMS",
    #                   aperture=0.01604070586606224,
    #                   modulation=1.0,
    #                   length=0.01828769716079613)
    #
    # myrfq.append_cell(cell_type="RMS",
    #                   aperture=0.01774325365260171,
    #                   modulation=1.0,
    #                   length=0.01828769716079613)
    #
    # myrfq.append_cell(cell_type="RMS",
    #                   aperture=0.021077138259769042,
    #                   modulation=1.0,
    #                   length=0.01828769716079613)
    #
    # myrfq.append_cell(cell_type="RMS",
    #                   aperture=0.02919376887767351,
    #                   modulation=0.038802 / 0.02919376887767351,
    #                   length=0.01828769716079613)

    myrfq.set_bempp_parameter("add_endplates", True)
    myrfq.set_bempp_parameter("cyl_id", 0.1)
    myrfq.set_bempp_parameter("grid_res", 0.005)
    myrfq.set_bempp_parameter("pot_shift", 3.0 * 22000.0)

    print(myrfq)

    print("Generating vanes")
    ts = time.time()
    myrfq.generate_vanes()
    print("Generating vanes took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))

    if rank == 0:
        myrfq.plot_vane_profile()
        myrfq.write_inventor_macro(vane_type='vane',
                                   vane_radius=0.0093,
                                   vane_height=0.03,
                                   vane_height_type='absolute',
                                   nz=600)
    exit()

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

    import pickle

    with open("ef_phi.field", "wb") as outfile:
        pickle.dump(myrfq.get_phi(), outfile)

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
    #
    # import pickle
    #
    # with open("efield_out.field", "wb") as outfile:
    #     pickle.dump(myrfq._variables_bempp["ef_itp"], outfile)

    # myrfq.plot_vane_profile()
