import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import numpy.ma as ma
import scipy.constants as const
from multiprocessing import Pool
from scipy.interpolate import interp1d
from dans_pymodules import Vector2D
import matplotlib.pyplot as plt
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
import copy
import os
import sys
import shutil

from matplotlib.patches import Arc as Arc

load_previous = False

# Check if we can connect to a display, if not disable all plotting and windowed stuff (like gmsh)
# TODO: This does not remotely cover all cases!
if "DISPLAY" in os.environ.keys():
    x11disp = True
else:
    x11disp = False

# --- Try importing BEMPP
HAVE_BEMPP = False
try:
    import bempp.api
    from bempp.api.shapes.shapes import __generate_grid_from_geo_string as generate_from_string

    HAVE_BEMPP = True
except ImportError:
    print("Couldn't import BEMPP, no meshing or BEM field calculation will be possible.")
    bempp = None
    generate_from_string = None

# --- Try importing mpi4py, if it fails, we fall back to single processor
try:

    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    HOST = MPI.Get_processor_name()

    print("Process {} of {} on host {} started!".format(RANK + 1, SIZE, HOST))
    sys.stdout.flush()

except ImportError:

    MPI = None
    COMM = None
    RANK = 0
    SIZE = 1
    import socket

    HOST = socket.gethostname()

    print("Could not import mpi4py, falling back to single core (and python multiprocessing in some instances)!")

# --- Try importing pythonocc-core
HAVE_OCC = False
try:

    from OCC.Extend.DataExchange import read_stl_file
    from OCC.Display.SimpleGui import init_display
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeTorus, BRepPrimAPI_MakeSweep
    from OCC.Core.BRepTools import breptools_Write
    from OCC.Core.BRepBndLib import brepbndlib_Add
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.gp import gp_Pnt, gp_Pnt2d
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.TopAbs import TopAbs_ON, TopAbs_OUT, TopAbs_IN
    from OCC.Core.GeomAPI import GeomAPI_Interpolate, GeomAPI_PointsToBSpline
    from OCC.Core.Geom import Geom_BSplineCurve
    from OCC.Core.Geom2d import Geom2d_BSplineCurve
    from OCC.Core.TColgp import TColgp_HArray1OfPnt, TColgp_Array1OfPnt
    from OCC.Core.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal
    from OCC.Core.GeomAbs import GeomAbs_C1, GeomAbs_C2, GeomAbs_G1
    from OCC.Core.Geom2dAPI import Geom2dAPI_Interpolate, Geom2dAPI_PointsToBSpline
    from OCC.Core.TColgp import TColgp_HArray1OfPnt2d, TColgp_Array1OfPnt2d
    from OCCUtils.Common import *

    from py_electrodes import ElectrodeObject

    HAVE_OCC = True

except ImportError:

    ElectrodeObject = None
    print("Something went wrong during OCC import. No CAD support possible!")

USE_MULTIPROC = True  # In case we are not using mpi or only using 1 processor, fall back on multiprocessing
GMSH_EXE = "/home/daniel/src/gmsh-4.0.6-Linux64/bin/gmsh"
# GMSH_EXE = "E:/gmsh4/gmsh.exe"
HAVE_TEMP_FOLDER = False
np.set_printoptions(threshold=10000)

HAVE_GMSH = True
# Quick test if gmsh path is correct
if not Path(GMSH_EXE).is_file():
    print("Gmsh path seems to be wrong! No meshing will be possible!")
    HAVE_GMSH = False

# For now, everything involving the pymodules with be done on master proc (RANK 0)
if RANK == 0:

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

# Define the axis directions and vane rotations:
X = 0
Y = 1
Z = 2

XYZ = range(3)

AXES = {"X": 0, "Y": 1, "Z": 2}

rot_map = {"yp": 0.0,
           "ym": 180.0,
           "xp": 270.0,
           "xm": 90.0}


class Polygon2D(object):
    """
    Simple class to handle polygon operations such as point in polygon or
    orientation of rotation (cw or ccw), area, etc.
    """

    def add_point(self, p=None):
        """
        Append a point to the polygon
        """

        if p is not None:

            if isinstance(p, tuple) and len(p) == 2:

                self.poly.append(p)

            else:
                print("Error in add_point of Polygon: p is not a 2-tuple!")

        else:
            print("Error in add_point of Polygon: No p given!")

        return 0

    def add_polygon(self, poly=None):
        """
        Append a polygon object to the end of this polygon
        """

        if poly is not None:

            if isinstance(poly, Polygon2D):
                self.poly.extend(poly.poly)
            # if isinstance(poly.poly, list) and len(poly.poly) > 0:
            #
            #     if isinstance(poly.poly[0], tuple) and len(poly.poly[0]) == 2:
            #         self.poly.extend(poly.poly)

        return 0

    def area(self):
        """
        Calculates the area of the polygon. only works if there are no crossings

        Taken from http://paulbourke.net, algorithm written by Paul Bourke, 1998

        If area is positive -> polygon is given clockwise
        If area is negative -> polygon is given counter clockwise
        """

        area = 0
        poly = self.poly
        npts = len(poly)
        j = npts - 1
        i = 0

        for _ in poly:
            p1 = poly[i]
            p2 = poly[j]
            area += (p1[0] * p2[1])
            area -= p1[1] * p2[0]
            j = i
            i += 1

        area /= 2

        return area

    def centroid(self):
        """
        Calculate the centroid of the polygon

        Taken from http://paulbourke.net, algorithm written by Paul Bourke, 1998
        """
        poly = self.poly
        npts = len(poly)
        x = 0
        y = 0
        j = npts - 1
        i = 0

        for _ in poly:
            p1 = poly[i]
            p2 = poly[j]
            f = p1[0] * p2[1] - p2[0] * p1[1]
            x += (p1[0] + p2[0]) * f
            y += (p1[1] + p2[1]) * f
            j = i
            i += 1

        f = self.area() * 6

        return x / f, y / f

    def clockwise(self):
        """
        Returns True if the polygon points are ordered clockwise

        If area is positive -> polygon is given clockwise
        If area is negative -> polygon is given counter clockwise
        """

        if self.area() > 0:
            return True
        else:
            return False

    def closed(self):
        """
        Checks whether the polygon is closed (i.e first point == last point)
        """

        if self.poly[0] == self.poly[-1]:

            return True

        else:

            return False

    def nvertices(self):
        """
        Returns the number of vertices in the polygon
        """

        return len(self.poly)

    def point_in_poly(self, p=None):
        """
        Check if a point p (tuple of x,y) is inside the polygon
        This is called the "ray casting method": If a ray cast from p crosses
        the polygon an even number of times, it's outside, otherwise inside

        From: http://www.ariel.com.au/a/python-point-int-poly.html

        Note:   Points directly on the edge or identical with a vertex are not
                        considered "inside" the polygon!
        """

        if p is None:
            return None

        poly = self.poly
        x = p[0]
        y = p[1]
        n = len(poly)
        inside = False

        p1x, p1y = poly[0]

        for i in range(n + 1):

            p2x, p2y = poly[i % n]

            if y > min(p1y, p2y):

                if y <= max(p1y, p2y):

                    if x <= max(p1x, p2x):

                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x

                        if p1x == p2x or x <= xinters:
                            inside = not inside

            p1x, p1y = p2x, p2y

        return inside

    def remove_last(self):
        """
        Remove the last tuple in the ploygon
        """

        self.poly.pop(-1)

        return 0

    def reverse(self):
        """
        Reverses the ordering of the polygon (from cw to ccw or vice versa)
        """

        temp_poly = []
        nv = self.nvertices()

        for i in range(self.nvertices() - 1, -1, -1):
            temp_poly.append(self.poly[i])

        self.poly = temp_poly

        return temp_poly

    def rotate(self, index):
        """
        rotates the polygon, so that the point with index 'index' before now has
        index 0
        """

        if index > self.nvertices() - 1:
            return 1

        for i in range(index):
            self.poly.append(self.poly.pop(0))

        return 0

    def __init__(self, poly=None):
        """
        construct a polygon object
        If poly is not specified, an empty polygon is created
        if poly is specified, it has to be a list of 2-tuples!
        """
        self.poly = []

        if poly is not None:

            if isinstance(poly, list) and len(poly) > 0:

                if isinstance(poly[0], tuple) and len(poly[0]) == 2:
                    self.poly = poly

    def __getitem__(self, index):

        return self.poly[index]

    def __setitem__(self, index, value):

        if isinstance(value, tuple) and len(value) == 2:
            self.poly[index] = value


class PyRFQCell(object):

    def __init__(self,
                 cell_type,
                 prev_cell=None,
                 next_cell=None,
                 debug=False,
                 **kwargs):
        """
        :param cell_type:
                STA: Start cell without length (necessary at beginning of RMS if there are no previous cells)
                RMS: Radial Matching Section.
                NCS: Normal Cell. A regular RFQ cell
                TCS: Transition Cell.
                DCS: Drift Cell. No modulation.
                TRC: Trapezoidal cell (experimental, for re-bunching only!).
        :param prev_cell:
        :param next_cell:
        :param debug:

        Keyword Arguments (mostly from Parmteq Output File):
        V:       Intervane voltage in V
        Wsyn:    Energy of the synchronous particle in MeV
        Sig0T:   Transverse zero-current phase advance in degrees per period
        Sig0L:   Longitudinal zero-current phase advance in degrees per period
        A10:     Acceleration term [first theta-independent term in expansion]
        Phi:     Synchronous phase in degrees
        a:       Minimum radial aperture in m
        m:       Modulation (dimensionless)
        B:       Focusing parameter (dimensionless) B = q V lambda^2/(m c^2 r0^2)
        L:       Cell length in cm
        A0:      Quadrupole term [first z-independent term in expansion]
        RFdef:   RF defocusing term
        Oct:     Octupole term
        A1:      Duodecapole term [second z-independent term in expansion]
        """

        assert cell_type in ["start", "rms", "regular",
                             "transition", "transition_auto", "drift", "trapezoidal"], \
            "cell_type not recognized!"

        self._type = cell_type

        self._params = {"voltage": None,
                        "Wsyn": None,
                        "Sig0T": None,
                        "Sig0L": None,
                        "A10": None,
                        "Phi": None,
                        "a": None,
                        "m": None,
                        "B": None,
                        "L": None,
                        "A0": None,
                        "RFdef": None,
                        "Oct": None,
                        "A1": None,
                        "flip_z": False,
                        "shift_cell_no": False,
                        "fillet_radius": None
                        }

        self._prev_cell = prev_cell
        self._next_cell = next_cell

        self._debug = debug

        for key, item in self._params.items():
            if key in kwargs.keys():
                self._params[key] = kwargs[key]

        if self.initialize() != 0:
            print("Cell failed self-check! Aborting.")
            exit(1)

        self._profile_itp = None  # Interpolation of the cell profile

    def __str__(self):
        return "Type: '{}', Aperture: {:.6f}, Modulation: {:.4f}, " \
               "Length: {:.6f}, flip: {}, shift: {}".format(self._type,
                                                            self._params["a"],
                                                            self._params["m"],
                                                            self._params["L"],
                                                            self._params["flip_z"],
                                                            self._params["shift_cell_no"])

    @property
    def length(self):
        return self._params["L"]

    @property
    def aperture(self):
        return self._params["a"]

    @property
    def avg_radius(self):
        return 0.5 * (self._params["a"] + self._params["m"] * self._params["a"])

    @property
    def cell_type(self):
        return self._type

    @property
    def modulation(self):
        return self._params["m"]

    @property
    def prev_cell(self):
        return self._prev_cell

    @property
    def next_cell(self):
        return self._next_cell

    def calculate_transition_cell_length(self):

        le = self._params["L"]
        m = self._params["m"]
        a = self._params["a"]
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

        assert tcs_length <= le, "Numerical determination of transition cell length " \
                                 "yielded value larger than cell length parameter!"

        if tcs_length > le:
            print("the remainder will be filled with a drift.")

        return tcs_length

    def initialize(self):
        # TODO: Refactor this maybe? seems overly complicated...
        # Here we check the different cell types for consistency and minimum necessary parameters
        if self._type in ["transition", "transition_auto"]:
            assert self.prev_cell is not None, "A transition cell needs a preceeeding cell."
            assert self.prev_cell.cell_type == "regular", "Currently a transition cell must follow a regular cell."

        # Aperture:
        assert self._params["a"] is not None, "No aperture given for {} cell".format(self._type)

        if self._params["a"] == 'auto':

            assert self._type in ["drift", "trapezoidal", "transition", "transition_auto"], \
                "Unsupported cell type '{}' for auto-aperture".format(self._type)

            assert self.prev_cell is not None, "Need a preceeding cell for auto aperture!"

            if self.prev_cell.cell_type in ["transition", "transition_auto"]:
                self._params["a"] = self.prev_cell.avg_radius
            else:
                self._params["a"] = self.prev_cell.aperture

        self._params["a"] = np.round(self._params["a"], decimals)

        # Modulation:
        if self._type in ["start", "rms", "drift"]:
            self._params["m"] = 1.0

        assert self._params["m"] is not None, "No modulation given for {} cell".format(self._type)

        if self._params["m"] == 'auto':
            assert self._type in ["transition", "transition_auto"], \
                "Only transition cell can have 'auto' modulation at the moment!"

            self._params["m"] = self.prev_cell.modulation

        self._params["m"] = np.round(self._params["m"], decimals)

        # Length:
        if self._type == "start":
            self._params["L"] = 0.0

        assert self._params["L"] is not None, "No length given for {} cell".format(self._type)

        if self._params["L"] == "auto":
            assert self._type == "transition_auto", "Only transition_auto cells allow auto-length!"

            self._params["L"] = self.prev_cell.length  # use preceeding cell length L for calculation of L'
            self._params["L"] = self.calculate_transition_cell_length()

        self._params["L"] = np.round(self._params["L"], decimals)

        if self._type == "trapezoidal":
            assert self._params["fillet_radius"] is not None, "For 'TRC' cell a fillet radius must be given!"

        return 0

    def set_prev_cell(self, prev_cell):
        assert isinstance(prev_cell, PyRFQCell), "You are trying to set a PyRFQCell with a non-cell object!"
        self._prev_cell = prev_cell

    def set_next_cell(self, next_cell):
        assert isinstance(next_cell, PyRFQCell), "You are trying to set a PyRFQCell with a non-cell object!"
        self._next_cell = next_cell

    def calculate_profile_rms(self, vane_type, cell_no):
        # Assemble RMS section by finding adjacent RMS cells and get their apertures
        cc = self
        pc = cc.prev_cell

        rms_cells = [cc]
        shift = 0.0

        while pc is not None and pc.cell_type == "rms":
            rms_cells = [pc] + rms_cells
            shift += pc.length

            cc = pc
            pc = cc.prev_cell

        cc = self
        nc = cc._next_cell

        while nc is not None and nc.cell_type == "rms":
            rms_cells = rms_cells + [nc]
            cc = nc
            nc = cc.next_cell

        # Check for starting cell
        assert rms_cells[0].prev_cell is not None, "Cannot assemble RMS section without a preceding cell! " \
                                                   "At the beginning ofthe RFQ consider using a start (STA) cell."

        a = [0.5 * rms_cells[0].prev_cell.aperture * (1.0 + rms_cells[0].prev_cell.modulation)]
        z = [0.0]

        for _cell in rms_cells:
            a.append(_cell.aperture)
            z.append(z[-1] + _cell.length)

        self._profile_itp = interp1d(np.array(z) - shift, np.array(a), kind='cubic')

        return 0

    def calculate_profile_transition(self, vane_type, cell_no):

        le = self._params["L"]
        m = self._params["m"]
        a = self._params["a"]
        k = np.pi / np.sqrt(3.0) / le  # Initial guess
        r0 = 0.5 * (a + m * a)

        if self.cell_type == "transition_auto":
            tcl = le
        else:
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

        if self._params["shift_cell_no"]:
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

        if self._params["flip_z"]:
            _vane = _vane[::-1]
            vane[np.where(z >= le - tcl)] = _vane
        else:
            vane[idx] = _vane

        self._profile_itp = interp1d(z, vane, bounds_error=False, fill_value=0)

        return 0

    def calculate_profile_trapezoidal(self, vane_type, cell_no):
        # TODO: This is a rough test of a trapezoidal cell: _/-\_
        # TODO: tilted parts are as long as roof and start and end (cell_length/5)
        fillet_radius = self._params["fillet_radius"]  # m

        def intersection(_p1, _v1, _p2, _v2):
            s = (_v2[1] * (_p2[0] - _p1[0]) + _v2[0] * (_p1[1] - _p2[1])) / (_v1[0] * _v2[1] - _v1[1] * _v2[0])
            return _p1 + s * _v1

        def arc_to_poly(z1, r1, z2, r2, r_curv, invert):
            """
            transform an arc into a polygon
            """
            polygon = Polygon2D()

            cur = 1
            if invert:
                cur = -1

            dp = np.sqrt((z2 - z1) ** 2 + (r2 - r1) ** 2)

            if r_curv < 0.5 * dp:
                return None

            dx = np.sqrt(abs((0.5 * dp) ** 2.0 - r_curv ** 2.0))
            zc = (z1 + z2) * 0.5 - cur * dx * (r1 - r2) / dp
            rc = (r1 + r2) * 0.5 + cur * dx * (z1 - z2) / dp

            if round(z1 - zc, 8) == 0:
                if r1 > rc:
                    p1 = 90
                else:
                    p1 = 270
            else:
                p1 = np.arctan((r1 - rc) / (z1 - zc)) / np.pi * 180.0
                if z1 < zc:
                    p1 += 180
            if p1 < 0:
                p1 += 360

            if round(z2 - zc, 8) == 0:
                if r2 > rc:
                    p2 = 90
                else:
                    p2 = 270
            else:
                p2 = np.arctan((r2 - rc) / (z2 - zc)) / np.pi * 180.0
                if z2 < zc:
                    p2 += 180
            if p2 < 0:
                p2 += 360

            diff = p2 - p1
            if diff < 0:
                diff += 360
            if diff > 180:
                p3 = p1
                p1 = p2
                p2 = p3

            num_vert = 10  # No need for too many, just spline guide points

            if p2 < p1:
                dp = float((p2 + 360.0 - p1) / (float(num_vert) - 1.0))

            else:
                dp = float((p2 - p1) / (float(num_vert) - 1.0))

            for j in range(num_vert):
                phi = np.deg2rad(p1 + dp * j)
                z_temp = zc + (r_curv * np.cos(phi))
                r_temp = rc + (r_curv * np.sin(phi))
                polygon.add_point((z_temp, r_temp))

            if not invert:
                polygon.reverse()

            return polygon, p1, p2

        # Flip for y vane
        flip_r = ("y" in vane_type) ^ self._params["shift_cell_no"]

        # 6 vertices for 5 segments of the trapezoidal cell
        _z = np.linspace(0, self._params["L"], 6, endpoint=True)

        if flip_r:
            _r = np.array([self._params["a"],
                           self._params["a"],
                           self._params["a"] * (2.0 - self._params["m"]),
                           self._params["a"] * (2.0 - self._params["m"]),
                           self._params["a"],
                           self._params["a"]
                           ])
        else:
            _r = np.array([self._params["a"],
                           self._params["a"],
                           self._params["a"] * self._params["m"],
                           self._params["a"] * self._params["m"],
                           self._params["a"],
                           self._params["a"]
                           ])

        # Now we replace the inner vertices with fillets
        _vertices = np.array(list(zip(_z, _r)))
        _new_verts = Polygon2D([tuple(_vertices[0])])

        for i in range(4):

            temp_poly = Polygon2D([tuple(_vertices[0 + i]), tuple(_vertices[1 + i]), tuple(_vertices[i + 2])])
            clockwise = temp_poly.clockwise()

            # Calculate maximum radius for fillet
            _v1 = Vector2D(p0=_vertices[i + 1], p1=_vertices[i + 0])
            _v2 = Vector2D(p0=_vertices[i + 1], p1=_vertices[i + 2])

            if clockwise:
                p_in_line1 = Vector2D(_vertices[i + 1]) + _v1.rotate_ccw().normalize() * fillet_radius  # belongs to v1
                p_in_line2 = Vector2D(_vertices[i + 1]) + _v2.rotate_cw().normalize() * fillet_radius  # belongs to v2
            else:
                p_in_line1 = Vector2D(_vertices[i + 1]) + _v1.rotate_cw().normalize() * fillet_radius  # belongs to v1
                p_in_line2 = Vector2D(_vertices[i + 1]) + _v2.rotate_ccw().normalize() * fillet_radius  # belongs to v2

            m_center = intersection(p_in_line1, _v1, p_in_line2, _v2)
            v_new1 = intersection(Vector2D(_vertices[i + 1]), _v1.normalize(), m_center, _v1.rotate_cw().normalize())
            v_new2 = intersection(Vector2D(_vertices[i + 1]), _v2.normalize(), m_center, _v2.rotate_cw().normalize())

            arcpoly, ps, pe = arc_to_poly(v_new1[0], v_new1[1],
                                          v_new2[0], v_new2[1],
                                          fillet_radius,
                                          not clockwise)

            _new_verts.add_polygon(arcpoly)

        _new_verts.add_point(tuple(_vertices[-1]))
        _new_verts = np.array(_new_verts[:])

        self._profile_itp = interp1d(_new_verts[:, 0], _new_verts[:, 1])

        return 0

    def calculate_profile(self, cell_no, vane_type, fudge=False):

        print("cell_no: " + str(cell_no))

        assert vane_type in ["xp", "xm", "yp", "ym"], "Did not understand vane type {}".format(vane_type)

        if self._type == "start":
            # Don't do anything for start cell
            return 0

        elif self._type == "trapezoidal":

            assert self._prev_cell.cell_type == "drift", "Rebunching cell must follow a drift cell (DCS)!"
            self.calculate_profile_trapezoidal(vane_type, cell_no)

            return 0

        elif self._type == "drift":

            self._profile_itp = interp1d([0.0, self._params["L"]],
                                         [self._params["a"], self._params["a"] * self._params["m"]])

            return 0

        elif self._type == "rms":

            self.calculate_profile_rms(vane_type, cell_no)

            return 0

        elif self._type in ["transition", "transition_auto"]:

            self.calculate_profile_transition(vane_type, cell_no)

            return 0

        # Else: regular cell:
        z = np.linspace(0.0, self._params["L"], 100)

        a = self.aperture
        m = self.modulation

        pc = self._prev_cell
        if pc is not None and pc.cell_type in ["rms", "drift"]:
            pc = None
        nc = self._next_cell
        if nc is not None and nc.cell_type in ["rms", "drift"]:
            nc = None

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
        return self._profile_itp(z)


class PyRFQElectrode(object):
    def __init__(self,
                 name,
                 parent,
                 zmin,
                 zlen,
                 voltage,
                 reverse_normals=False,
                 h=0.025,
                 debug=False):

        self._name = name
        self._parent = parent
        self._domain_idx = None
        self._voltage = voltage
        self._debug = debug

        self._zmin = zmin
        self._zlen = zlen

        self._geo_str = None
        self._occ_obj = None
        self._occ_npart = 1

        self._mesh_fn = None

        self._reverse_normals = reverse_normals
        self._h = h
        self._refine_steps = 0

    @property
    def domain_idx(self):
        return self._domain_idx

    @property
    def mesh_fn(self):
        return self._mesh_fn

    @property
    def name(self):
        return self._name

    @property
    def parent(self):
        return self._parent

    @property
    def voltage(self):
        return self._voltage

    @abstractmethod
    def generate_geo_str(self, *args, **kwargs):
        pass

    def generate_gmsh_files(self):

        tmp_dir = self._parent.temp_dir

        if tmp_dir is not None:

            geo_fn = os.path.join(tmp_dir, "{}.geo".format(self.name))
            msh_fn = os.path.splitext(geo_fn)[0] + ".msh"
            stl_fn = os.path.splitext(geo_fn)[0] + ".stl"
            brep_fn = os.path.splitext(geo_fn)[0] + ".brep"
            refine_fn = os.path.join(tmp_dir, "refine_{}.geo".format(self.name))

            gmsh_success = 0

            with open(geo_fn, "w") as _of:
                _of.write(self._geo_str)

            command = "{} \"{}\" -0 -o \"{}\" -format brep".format(GMSH_EXE, geo_fn, brep_fn)
            if self._debug:
                print("Running", command)
                sys.stdout.flush()
            gmsh_success += os.system(command)

            refine_str = """
Merge "{}";
Mesh.SecondOrderLinear = 0;
RefineMesh;
""".format(msh_fn)

            with open(refine_fn, "w") as _of:
                _of.write(refine_str)

            # TODO: Could we use higher order (i.e. curved) meshes? -DW
            # For now, we need to save in msh2 format for BEMPP compability
            command = "{} \"{}\" -2 -o \"{}\" -format msh2".format(GMSH_EXE, geo_fn, msh_fn)
            if self._debug:
                print("Running", command)
                sys.stdout.flush()
            gmsh_success += os.system(command)

            for i in range(self._refine_steps):
                command = "{} \"{}\" -0 -o \"{}\" -format msh2".format(GMSH_EXE, refine_fn, msh_fn)
                if self._debug:
                    print("Running", command)
                    sys.stdout.flush()
                gmsh_success += os.system(command)

            # --- TODO: For testing: save stl mesh file also
            command = "{} \"{}\" -0 -o \"{}\" -format stl".format(GMSH_EXE, msh_fn, stl_fn)
            if self._debug:
                print("Running", command)
                sys.stdout.flush()
            gmsh_success += os.system(command)
            # --- #

            if gmsh_success != 0:  # or not os.path.isfile("shape.stl"):
                print("Something went wrong with gmsh, be sure you defined "
                      "the correct path at the beginning of the file!")
                return 1

            self._mesh_fn = msh_fn

        return 0

    def generate_occ(self):

        if HAVE_OCC:

            tmp_dir = self._parent.temp_dir
            brep_fn = os.path.join(tmp_dir, "{}.brep".format(self._name))

            self._occ_obj = ElectrodeObject()
            self._occ_obj.load_from_brep(brep_fn)
            self._occ_obj.partition_z(self._occ_npart)

            return 0

        else:

            print("Couldn't load PythonOCC-Core earlier, cannot create OpenCasCade object!")

            return 1

    def points_inside(self, _points):
        """
        Function that calculates whether the point(s) is/are inside the vane or not.
        Currently this only works with pythonocc-core installed and can be very slow
        for a large number of points.
        :param _points: any shape (N, 3) structure holding the points to check. Can be a list of tuples,
                       a list of lists, a numpy array of points (N, 3)...
                       Alternatively: a single point with three coordinates (list, tuple or numpy array)
        :return: boolean numpy array of True or False depending on whether the points are inside or
                 outside (on the surface is counted as inside!)
        """

        if self._occ_obj is not None:

            return self._occ_obj.points_inside(_points)

        else:

            return 1


class PyRFQAnnulus(PyRFQElectrode):

    def __init__(self,
                 name,
                 parent,
                 zmin,
                 zlen,
                 voltage=0.0,
                 debug=False,
                 reverse_normals=False,
                 h=0.05,
                 plate_dia=1.0,
                 aperture_dia=0.0):

        super().__init__(name,
                         parent,
                         zmin,
                         zlen,
                         voltage,
                         reverse_normals,
                         h,
                         debug)

        self._aperture_dia = aperture_dia
        self._plate_dia = plate_dia
        self._domain_idx = 100

    def generate_geo_str(self):

        zlen = self._zlen
        r_plate = self._plate_dia / 2.0
        r_ap = self._aperture_dia / 2.0
        zmin = self._zmin
        reverse_normals = self._reverse_normals
        h = self._h

        self._geo_str = """SetFactory("OpenCASCADE");
Geometry.NumSubEdges = 100; // nicer display of curve
Mesh.CharacteristicLengthMax = {};
        """.format(h)
        self._geo_str += "// Create Plate \n"
        self._geo_str += "Cylinder(1) = {{ 0, 0, {}, 0, 0, {}, {}, 2 * Pi }};\n".format(zmin,
                                                                                        zlen,
                                                                                        r_plate)

        if r_ap > 0.0:
            self._geo_str += "Cylinder(2) = {{ 0, 0, {}, 0, 0, {}, {}, 2 * Pi }};\n".format(zmin - 0.001,
                                                                                            zlen + 0.002,
                                                                                            r_ap)

            self._geo_str += "BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }\n"

        self._geo_str += """
s() = Surface "*";
Physical Surface({}) = {{ s() }};
""".format(self._domain_idx)

        if reverse_normals:
            self._geo_str += """
ReverseMesh Surface { s() };
            """

        return 0


class PyRFQVane(PyRFQElectrode):
    def __init__(self,
                 parent,
                 vane_type,
                 cells,
                 voltage,
                 occ_tolerance=1e-5,
                 occ_npart=1,
                 debug=False,
                 reverse_normals=False,
                 h=0.05):

        self._cells = cells
        self._length = np.sum([cell.length for cell in self._cells])  # type: float

        super().__init__(name="vane_" + vane_type,
                         parent=parent,
                         zmin=0.0,
                         zlen=self._length,
                         voltage=voltage,
                         reverse_normals=reverse_normals,
                         h=h,
                         debug=debug)

        self._occ_npart = occ_npart
        self._type = vane_type
        self._has_profile = False
        self._fudge = False

        self._mesh_params = {"dx": 0.001,  # step length along z (m)
                             "nz": 100,  # Number of steps along z, consolidate with dx!
                             "h": 0.005,  # gmsh meshing parameter (m)
                             "tip": "semi-circle",
                             "r_tip": 0.005,  # Radius of curvature of vane tip (m)
                             "h_block": 0.01,  # height of block sitting atop the curvature (m)
                             "h_type": 'absolute',  # whether the block height is measured from midplane or modulation
                             "symmetry": False,
                             "mirror": False,
                             "geo_str": None,
                             "msh_fn": None,
                             "refine_steps": 0,  # Number of times gmsh is called to "refine by splitting"
                             "reverse_mesh": False
                             }

        self._occ_params = {"tolerance": occ_tolerance,
                            "solid": None,  # The OCC solid body,
                            "bbox": None,  # The bounding box ofthe OCC solid body
                            }

        self._mesh = None

    @property
    def has_profile(self):
        return self._has_profile

    @property
    def length(self):
        return self.length  # type: float

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

    def set_vane_type(self, vane_type=None):
        if vane_type is not None:
            self._type = vane_type
            self._name = "vane_" + vane_type

    def set_mesh_parameter(self, keyword=None, value=None):

        if keyword is None or value is None:
            print("In 'set_mesh_parameter': Either keyword or value were not specified.")
            return 1

        if keyword not in self._mesh_params.keys():
            print("In 'set_mesh_parameter': Unrecognized keyword '{}'.".format(keyword))
            return 1

        self._mesh_params[keyword] = value

        return 0

    def get_parameter(self, key):

        if key in self._mesh_params.keys():
            return self._mesh_params[key]
        else:
            return None

    def set_voltage(self, voltage):
        self._voltage = voltage

    def set_domain_index(self, idx):
        self._mesh_params["domain_idx"] = idx

    def generate_geo_str(self):

        r_tip = None
        h_block = None
        h_type = None
        symmetry = None
        mirror = None
        reverse_mesh = None

        h = dx = self._h

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

        if h is not None:
            self._mesh_params["h"] = h
        else:
            h = self._mesh_params["h"]

        if r_tip is not None:
            self._mesh_params["r_tip"] = r_tip
        else:
            r_tip = self._mesh_params["r_tip"]

        if h_block is not None:
            self._mesh_params["h_block"] = h_block
        else:
            h_block = self._mesh_params["h_block"]

        if h_type is not None:
            self._mesh_params["h_type"] = h_type
        else:
            h_type = self._mesh_params["h_type"]

        if reverse_mesh is not None:
            self._mesh_params["reverse_mesh"] = reverse_mesh
        else:
            reverse_mesh = self._mesh_params["reverse_mesh"]

        # Calculate z_data and vane profile:
        z, profile = self.get_profile(nz=self._mesh_params["nz"])
        pmax = profile.max()

        # Calculate minimum possible absolute height (1 mm above the maximum vane modulation):
        h_min = 0.0
        has_rms = False
        for _cell in self._cells:
            if _cell.cell_type == "rms":
                has_rms = True
            # Check for maximum modulated vanes plus 1 mm for safety.
            if _cell.cell_type not in ["start", "rms"]:
                _h = _cell.aperture * _cell.modulation + 0.001
                if h_min < _h:
                    h_min = _h

        # Consistency check for absolute h_type
        if h_type == 'absolute':

            if h_block >= pmax:

                ymax = h_block

            elif h_block >= h_min:

                print("*** Warning: h_block < pmax, but larger than maximum vane modulation. "
                      "This will cut into the RMS Section! Continuing.")
                ymax = h_block

            else:

                print("It seems that the 'absolute' h_block (height) value is too small" \
                      " and would leave less than 1 mm material in some places above the modulation. " \
                      "Aborting.")
                return 1

        elif h_type == 'relative':

            ymax = pmax + h_block

            print("h_type 'relative' deactivated for the moment. Aborting. -DW")
            return 1

        else:

            print("Unknown 'h_type'.")
            return 1

        # TODO: Look into what the best meshing parameters are!
        # TODO: Look into number of threads!
        geo_str = """SetFactory("OpenCASCADE");
Geometry.NumSubEdges = 500; // nicer display of curve
//General.NumThreads = 2;
Mesh.CharacteristicLengthMax = {};
h = {};

""".format(h, h)

        if symmetry:
            assert self._type not in ["ym", "xm"], "Sorry, mesh generation with symmetry only works for vanes " \
                                                   "located in positive axis directions (i.e. 'yp', 'xp'). "

            # if "x" in self._type:
            #     sign = -1

        if "y" in self._type:
            self._domain_idx = 2
        else:
            self._domain_idx = 1

        new_pt = 1
        new_ln = 1
        new_loop = 1
        new_surf = 1
        new_vol = 1
        spline1_pts = [new_pt]

        # Center spline
        # TODO: Here we could add an option for the cut-ins -DW
        geo_str += "// Center Spline:\n"
        for _z, _a in zip(z, profile):
            geo_str += "Point({}) = {{ {}, {}, {}, h }};\n".format(spline1_pts[-1], 0.0, _a, _z)
            spline1_pts.append(spline1_pts[-1] + 1)

        new_pt = spline1_pts[-1]
        spline1_pts.pop(-1)

        geo_str += """
Spline({}) = {{ {}:{} }}; 

""".format(new_ln, spline1_pts[0], spline1_pts[-1])

        # Immediately delete the points used up in the spline
        geo_str += "Recursive Delete {{ Point{{ {}:{} }}; }}\n".format(spline1_pts[1], spline1_pts[-2])

        spline_ln = new_ln
        new_ln += 1

        # --- Make a profile to follow the modulation path ('sweep' in Inventor, 'pipe' in OpenCascade) --- #
        profile_start_angle = np.arctan2(profile[1] - profile[0], z[1] - z[0])
        profile_end_angle = np.arctan2(profile[-1] - profile[-2], z[-1] - z[-2])

        print("Profile Start Angle = {} deg".format(-np.rad2deg(profile_start_angle)))
        print("Profile End Angle = {} deg".format(-np.rad2deg(profile_end_angle)))

        adj_psa_deg = -np.rad2deg(profile_start_angle)
        adj_pea_deg = np.rad2deg(profile_end_angle)

        geo_str += "// Points making up the sweep face:\n"
        face_pts = list(range(new_pt, new_pt + 4))

        # Square points:
        geo_str += "Point({}) = {{ {}, {}, {}, h }};\n".format(face_pts[0], -r_tip, profile[0] + r_tip, z[0])
        geo_str += "Point({}) = {{ {}, {}, {}, h }};\n".format(face_pts[1], r_tip, profile[0] + r_tip, z[0])

        # Semi-circle center:
        geo_str += "Point({}) = {{ {}, {}, {}, h }};\n".format(face_pts[2], 0.0, profile[0] + r_tip, z[0])
        geo_str += "\n"

        # Lines for sweep face:
        face_lns = []
        for i in range(1):
            face_lns.append(new_ln)
            geo_str += "Line({}) = {{ {}, {} }};\n".format(new_ln, face_pts[i], face_pts[i + 1])
            new_ln += 1

        # Semi-circle:
        face_lns.append(new_ln)
        geo_str += "Circle({}) = {{ {}, {}, {}}};\n".format(new_ln, face_pts[1], face_pts[2], face_pts[0])
        geo_str += "\n"
        new_ln += 1

        # Sweep Face:
        geo_str += "Curve Loop({}) = {{ {}, {} }};\n".format(new_loop,
                                                             face_lns[0],
                                                             face_lns[1],
                                                             )
        new_loop += 1

        sweep_surf = new_surf
        geo_str += "Plane Surface({}) = {{ {} }};\n".format(new_surf, new_loop - 1)
        geo_str += "Rotate {{{{1, 0, 0}}, {{ {}, {}, {}}}, {}}} {{Surface {{ {} }}; }}\n".format(0.0,
                                                                                                 profile[0],
                                                                                                 z[0],
                                                                                                 -profile_start_angle,
                                                                                                 new_surf)
        geo_str += "\n"
        new_surf += 1

        # Delete now unused center-point of circle (was duplicated)
        geo_str += "Recursive Delete {{ Point{{ {} }}; }}\n".format(face_pts[2])

        # Extrusion:
        geo_str += "Wire({}) = {{ {} }};\n".format(new_loop, spline_ln)
        geo_str += "Extrude {{ Surface{{ {} }}; }} Using Wire {{ {} }}\n".format(sweep_surf, new_loop)
        new_loop += 1
        extrude_vol_1 = new_vol
        new_vol += 1  # Extrude creates a volume

        # Delete initial sweep surface (now redundant)
        geo_str += "Recursive Delete {{ Surface {{ {} }}; }}\n".format(sweep_surf)
        # Delete the spline (now redundant)
        geo_str += "Recursive Delete {{ Curve{{ {} }}; }}\n".format(spline_ln)

        # We now have a volume of the modulated part regardless of h_block and RMS section yes/no.
        # All redundant points, lines and surfaces have been deleted.
        # ------------------------------------------------------------------------------------------------------------ #

        # --- Next step: Fill up the volume above to make height of vane = ymax -------------------------------------- #
        # - Cases:
        # 1. Both start and end angles are tilted inwards /===\ (using minimum tilt of 1 deg for now).
        # 2. Both start and end angles are straight or tilted outwards |===| or \===/
        # 3. Start angle is tilted inwards, end angle is straight or tilted outwards /===| (e.g. ony using start RMS)
        # 4. Start angle is straight or tilted outwards, end angle is tilted inwards |===\ (e.g. only using exit RMS)
        if adj_psa_deg >= 1.0 and adj_pea_deg >= 1.0:
            case = 1
        elif adj_psa_deg < 1.0 and adj_pea_deg < 1.0:
            case = 2
        elif adj_pea_deg < 1.0 <= adj_psa_deg:
            case = 3
        else:
            case = 4

        # In case 1, we can extend the end-caps upwards 1 m (just some large number),
        # then cut off a big block from the top. End caps will be surfaces 2 and 5
        if case == 1:
            geo_str += "Extrude {0, 1, 0} { Surface{ 2 }; }\n"
            geo_str += "Extrude {0, 1, 0} { Surface{ 5 }; }\n\n"

            geo_str += "// Delete redundant volumes, surfaces, lines to form a new volume later\n"
            geo_str += "Delete { Volume{ 1, 2, 3 }; }\n"
            geo_str += "Delete { Surface{ 2, 3, 5, 6, 9 }; }\n"
            geo_str += "Delete { Curve{ 4, 8 }; }\n"

            geo_str += "Line(18) = {{ {}, {} }};\n".format(new_pt + 12, new_pt + 10)
            geo_str += "Line(19) = {{ {}, {} }};\n".format(new_pt + 9, new_pt + 11)
            geo_str += """
Curve Loop(13) = {19, 16, 18, -12};
Plane Surface(12) = {13};
Curve Loop(14) = {18, -11, 7, 15};
Plane Surface(13) = {14};
Curve Loop(15) = {19, -14, -6, 10};
Plane Surface(14) = {15};
Surface Loop(4) = {13, 12, 14, 10, 11, 4, 7, 8};
Volume(1) = {4};
Delete { Surface{ 7, 10}; }
"""
        # In case 2 we create a block above the 4 endpoints of the semi-circles
        elif case == 2:
            geo_str += "Translate {{ 0, 1, 0 }} {{ Duplicata{{ Point{{ {}, {}, {}, {} }}; }} }}\n".format(new_pt + 5,
                                                                                                          new_pt + 6,
                                                                                                          new_pt + 7,
                                                                                                          new_pt + 8)
            geo_str += "Delete { Volume{ 1 }; }\n"
            geo_str += "Delete { Surface{ 3 }; }\n"

            geo_str += "Line(10) = {{ {}, {} }};\n".format(new_pt + 10, new_pt + 9)
            geo_str += "Line(11) = {{ {}, {} }};\n".format(new_pt + 9, new_pt + 11)
            geo_str += "Line(12) = {{ {}, {} }};\n".format(new_pt + 11, new_pt + 12)
            geo_str += "Line(13) = {{ {}, {} }};\n".format(new_pt + 12, new_pt + 10)
            geo_str += "Line(14) = {{ {}, {} }};\n".format(new_pt + 8, new_pt + 12)
            geo_str += "Line(15) = {{ {}, {} }};\n".format(new_pt + 11, new_pt + 7)
            geo_str += "Line(16) = {{ {}, {} }};\n".format(new_pt + 6, new_pt + 10)
            geo_str += "Line(17) = {{ {}, {}}};\n".format(new_pt + 9, new_pt + 5)

            geo_str += """
Curve Loop(7) = {13, 10, 11, 12}; Plane Surface(6) = {7};
Curve Loop(8) = {12, -14, -8, -15}; Plane Surface(7) = {8};
Curve Loop(9) = {16, 10, 17, 4}; Plane Surface(8) = {9};
Curve Loop(10) = {13, -16, 7, 14}; Plane Surface(9) = {10};
Curve Loop(11) = {15, -6, -17, 11}; Plane Surface(10) = {11};
Surface Loop(2) = {6, 9, 8, 10, 7, 5, 4, 2}; Volume(1) = {2};
"""
        elif case == 3:
            geo_str += "Extrude {0, 1, 0} { Surface{ 2 }; }\n"
            geo_str += "Translate {{ 0, 1, 0 }} {{ Duplicata{{ Point{{ {}, {} }}; }} }}\n".format(new_pt + 7,
                                                                                                  new_pt + 8)

            geo_str += "// Delete redundant volumes, surfaces, lines to form a new volume later\n"
            geo_str += "Delete { Volume{ 1, 2 }; }\n"
            geo_str += "Delete { Surface{ 2, 3, 6}; }\n"
            geo_str += "Delete { Curve{ 4 }; }\n"

            geo_str += "Line(14) = {{ {}, {} }};\n".format(new_pt + 10, new_pt + 12)
            geo_str += "Line(15) = {{ {}, {} }};\n".format(new_pt + 9, new_pt + 11)
            geo_str += "Line(16) = {{ {}, {} }};\n".format(new_pt + 11, new_pt + 12)
            geo_str += "Line(17) = {{ {}, {}}};\n".format(new_pt + 12, new_pt + 8)
            geo_str += "Line(18) = {{ {}, {} }};\n".format(new_pt + 11, new_pt + 7)

            geo_str += """
Curve Loop(10) = {16, -14, -12, 15}; Plane Surface(9) = {10};
Curve Loop(11) = {17, -7, 11, 14}; Plane Surface(10) = {11};
Curve Loop(12) = {17, -8, -18, 16}; Plane Surface(11) = {12};
Curve Loop(13) = {18, -6, 10, 15}; Plane Surface(12) = {13}; 
Surface Loop(3) = {10, 11, 5, 4, 12, 7, 8, 9}; Volume(1) = {3};
"""
            geo_str += "Delete { Surface{ 7 }; }\n"

        elif case == 4:
            geo_str += "Extrude {0, 1, 0} { Surface{ 5 }; }\n\n"
            geo_str += "Translate {{ 0, 1, 0 }} {{ Duplicata{{ Point{{ {}, {} }}; }} }}\n".format(new_pt + 5,
                                                                                                  new_pt + 6)

            geo_str += "// Delete redundant volumes, surfaces, lines to form a new volume later\n"
            geo_str += "Delete { Volume{ 1, 2 }; }\n"
            geo_str += "Delete { Surface{3, 5, 6}; }\n"
            geo_str += "Delete { Curve{ 8 }; }\n"

            geo_str += "Line(14) = {{ {}, {} }};\n".format(new_pt + 10, new_pt + 12)
            geo_str += "Line(15) = {{ {}, {} }};\n".format(new_pt + 9, new_pt + 11)
            geo_str += "Line(16) = {{ {}, {} }};\n".format(new_pt + 12, new_pt + 11)
            geo_str += "Line(17) = {{ {}, {}}};\n".format(new_pt + 6, new_pt + 12)
            geo_str += "Line(18) = {{ {}, {} }};\n".format(new_pt + 5, new_pt + 11)

            geo_str += """
Curve Loop(10) = {14, 16, -15, 12}; Plane Surface(9) = {10};
Curve Loop(11) = {14, -17, 7, 11}; Plane Surface(10) = {11};
Curve Loop(12) = {6, 10, 15, -18}; Plane Surface(11) = {12};
Curve Loop(13) = {16, -18, 4, 17}; Plane Surface(12) = {13};
Surface Loop(3) = {10, 9, 12, 11, 4, 7, 8, 2}; Volume(1) = {3};
"""
            geo_str += "Delete { Surface{ 7 }; }\n"
        # ------------------------------------------------ END CASES ------------------------------------------------- #

        geo_str += "Box(2) = {{ -0.5, {}, {}, 1, 2, {} }};\n".format(ymax, z[0] - 0.25, z[-1] - z[0] + 0.5)
        geo_str += """
BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }
"""
        # Add physical surface to identify this vane in gmsh (unmirrored)
        geo_str += """
s() = Surface "*";
Physical Surface({}) = {{ s() }};
""".format(self._domain_idx)

        # Rotate according to vane type
        if self.vane_type == "xp":
            geo_str += "Rotate {{{{0, 0, 1}}, {{0, 0, 0}}, {}}} " \
                       "{{Volume {{ {} }}; }}\n".format(-0.5 * np.pi, extrude_vol_1)
        elif self.vane_type == "xm":
            geo_str += "Rotate {{{{0, 0, 1}}, {{0, 0, 0}}, {}}} " \
                       "{{Volume {{ {} }}; }}\n".format(0.5 * np.pi, extrude_vol_1)
        elif self.vane_type == "ym":
            geo_str += "Rotate {{{{0, 0, 1}}, {{0, 0, 0}}, {}}} " \
                       "{{Volume {{ {} }}; }}\n".format(np.pi, extrude_vol_1)

        if reverse_mesh:
            geo_str += """
ReverseMesh Surface { s() };
"""
        # TODO: Adjust the transfinite surfaces for all the correct ones for the different cases.
        if case == 1:
            geo_str += """
Transfinite Surface { 2, 3 };
"""
        elif case == 2:
            geo_str += """
Transfinite Surface { 3 };
"""
        elif case == 3:
            geo_str += """
Transfinite Surface { 3, 4 };
"""
        elif case == 4:
            geo_str += """
Transfinite Surface { 3 };
"""

        self._geo_str = geo_str

        return geo_str

    def calculate_profile(self, fudge=None):

        if fudge is None:
            fudge = self._fudge

        for cell_no in range(len(self._cells)):
            self._cells[cell_no].calculate_profile(cell_no, self._type, fudge=fudge)

        sys.stdout.flush()

        self._has_profile = True

        return 0

    def get_profile(self, nz=1000):

        assert self._has_profile, "No profile has been generated!"

        # Cutting the RFQ short by 1e-10 to not get out of bound error in interpolation
        z = np.round(np.linspace(0.0, self._length - 1e-10, nz), decimals)
        vane = np.zeros(z.shape)

        cum_len = 0.0
        # count = 0
        for cell in self._cells:

            if cell.cell_type != "start":
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


# noinspection PyUnresolvedReferences
class PyRFQ(object):
    def __init__(self, voltage, occ_tolerance=1e-5, debug=False, fudge_vanes=False):

        self._debug = debug
        self._fudge_vanes = fudge_vanes
        self._voltage = voltage
        self._vanes = []
        self._elec_objects = []
        self._cells = []
        self._cell_nos = []
        self._length = 0.0
        self._full_mesh = None
        self._full_mesh_fn = None
        self._occ_tolerance = occ_tolerance  # Tolerace for bounding box and intersection tests in pythonocc-core
        self._temp_dir = None

        # Flags
        self._have_geo_str = False
        self._have_occ_obj = False
        self._have_bem_obj = False
        self._initialized = False

        self._variables_gmtry = {"vane_type": "hybrid",
                                 "vane_radius": 0.005,  # m
                                 "vane_height": 0.05,  # m
                                 "vane_height_type": 'absolute',
                                 "nz": 500  # number of points to use for modulation spline along z
                                 # TODO: nz is confusing, now we have dx, numz and nz that could all determine
                                 # TODO: the step length along axis for geometry purposes! -DW
                                 }

        self._variables_bempp = {
                                 # "solution": None,
                                 # "f_space": None,
                                 # "operator": None,
                                 # "grid_fun": None,
                                 "grid_res": 0.005,  # grid resolution in (m)
                                 "refine_steps": 0,
                                 "reverse_mesh": False,
                                 "n_fun_coeff": None,  # Coefficients of the Neumann GridFunction
                                 "d_fun_coeff": None,  # Coefficients of the Dirichlet GridFunction
                                 "ef_itp": None,  # type: Field
                                 "ef_phi": None,  # type: np.ndarray
                                 "ef_mask": None,  # A numpy boolean array holding flags for points inside electrodes
                                 # This can help with jitter on z axis where pot ~ 0 otherwise
                                 # TODO: Should put pot in its own class that also holds dx, nx, etc.
                                 "add_cyl": False,  # Do we want to add a grounded cylinder to the BEMPP problem
                                 "add_endplates": False,  # Or just grounded end plates
                                 "cyl_id": 0.2,  # Inner diameter of surrounding cylinder
                                 "ap_id": 0.02,  # Entrance and exit aperture diameter TODO: Make this asymmetric!
                                 "cyl_gap": 0.01,  # gap between vanes and cylinder TODO: Make this asymmetric!
                                 "d": None,
                                 "n": None,
                                 "limits": None
                                 }

        self.create_temp_dir()

    @property
    def temp_dir(self):
        return self._temp_dir

    def create_temp_dir(self):

        if RANK == 0:

            tmp_path = os.path.join(os.getcwd(), "temp")

            if not os.path.exists(tmp_path):
                os.mkdir(tmp_path)
            else:
                shutil.rmtree(tmp_path)
                os.mkdir(tmp_path)

            if os.path.exists(tmp_path):
                global HAVE_TEMP_FOLDER
                HAVE_TEMP_FOLDER = True
            else:
                print("Could not create temp folder. Aborting.")
                exit(1)

            mpi_data = {"tmp_path": tmp_path}

        else:

            mpi_data = None

        if MPI is not None:
            mpi_data = COMM.bcast(mpi_data, root=0)

        self._temp_dir = mpi_data["tmp_path"]

        return self._temp_dir

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

        if keyword is None:
            print("In 'set_bempp_parameter': No keyword specified.")
            return 1

        if keyword not in self._variables_bempp.keys():
            print("In 'set_bempp_parameter': Unrecognized keyword '{}'.".format(keyword))
            return 1

        return self._variables_bempp[keyword]

    def set_geometry_parameter(self, keyword=None, value=None):

        if keyword is None or value is None:
            print("In 'set_geometry_parameter': Either keyword or value were not specified.")
            return 1

        if keyword not in self._variables_gmtry.keys():
            print("In 'set_geometry_parameter': Unrecognized keyword '{}'.".format(keyword))
            return 1

        self._variables_gmtry[keyword] = value

        return 0

    def get_geometry_parameter(self, keyword=None):

        if keyword is None:
            print("In 'set_geometry_parameter': No keyword specified.")
            return 1

        if keyword not in self._variables_gmtry.keys():
            print("In 'set_geometry_parameter': Unrecognized keyword '{}'.".format(keyword))
            return 1

        return self._variables_gmtry[keyword]

    def add_cells_from_file(self, filename=None, ignore_rms=False):
        """
        Reads a file with cell parameters and generates the respective RFQCell objects
        :param filename:
        :param ignore_rms: Bool. If True, any radial matching cells in the file are ignored.
        :return:
        """

        if filename is None:

            if RANK == 0:
                fd = FileDialog()
                mpi_data = {"fn": fd.get_filename('open')}
            else:
                mpi_data = None

            if MPI is not None:
                mpi_data = COMM.bcast(mpi_data, root=0)

            filename = mpi_data["fn"]

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

    def add_cylinder(self):

        for _elec_obj in self._elec_objects:
            if "plate" in _elec_obj.name:
                print("Cannot create cylinder if there are endplates already!")
                return 1

        print("Cylinder not yet implemented :(")

        return 0

    def add_endplates(self,
                      gap_sta,
                      gap_end,
                      thickness,
                      plate_dia,
                      voltage=0.0,
                      aperture_dia=0.0):

        for _elec_obj in self._elec_objects:
            if "cylinder" in _elec_obj.name:
                print("Cannot create endplates if there is an outer cylinder already!")
                return 1

        # Delete all existing plate objects
        self._elec_objects = [_elec_obj for _elec_obj in self._elec_objects if "plate" not in _elec_obj.name]

        # Entrance Plate
        zmin = 0.0 - gap_sta - thickness

        plate_sta = PyRFQAnnulus(name="entrance_plate",
                                 parent=self,
                                 zmin=zmin,
                                 zlen=thickness,
                                 voltage=voltage,
                                 debug=self._debug,
                                 reverse_normals=self._variables_bempp["reverse_mesh"],
                                 h=self._variables_bempp["grid_res"],
                                 plate_dia=plate_dia,
                                 aperture_dia=aperture_dia)

        self._elec_objects.append(plate_sta)

        # Exit Plate
        zmin = self._length + gap_end

        plate_sta = PyRFQAnnulus(name="exit_plate",
                                 parent=self,
                                 zmin=zmin,
                                 zlen=thickness,
                                 voltage=voltage,
                                 debug=self._debug,
                                 reverse_normals=self._variables_bempp["reverse_mesh"],
                                 h=self._variables_bempp["grid_res"],
                                 plate_dia=plate_dia,
                                 aperture_dia=aperture_dia)

        self._elec_objects.append(plate_sta)

        return 0

    def append_cell(self,
                    cell_type,
                    **kwargs):

        assert cell_type in ["start", "rms", "regular",
                             "transition", "transition_auto", "drift", "trapezoidal"], \
            "cell_type not recognized!"

        if len(self._cells) > 0:
            pc = self._cells[-1]
        else:
            pc = None

        self.reset()

        self._cells.append(PyRFQCell(cell_type=cell_type,
                                     prev_cell=pc,
                                     next_cell=None,
                                     debug=self._debug,
                                     **kwargs))

        if len(self._cells) > 1:
            self._cells[-2].set_next_cell(self._cells[-1])

        self._cell_nos = range(len(self._cells))
        self._length = np.sum([cell.length for cell in self._cells])

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
                        self._cells.append(PyRFQCell(cell_type="start",
                                                     V=params[0] * 1000.0,
                                                     Wsyn=params[1],
                                                     Sig0T=params[2],
                                                     Sig0L=params[3],
                                                     A10=params[4],
                                                     Phi=params[5],
                                                     a=params[6] * 0.01,
                                                     B=params[8],
                                                     debug=self._debug))

                    continue

                # For now we ignore "special" cells and add them manually
                if "T" in cell_no or "M" in cell_no or "F" in cell_no:
                    print("Ignored cell {}".format(cell_no))
                    continue

                if "R" in cell_no:
                    cell_type = "rms"
                    if ignore_rms:
                        print("Ignored cell {}".format(cell_no))
                        continue
                else:
                    cell_type = "regular"

                if len(self._cells) > 0:
                    pc = self._cells[-1]
                else:
                    pc = None

                if cell_type == "rms":

                    self._cells.append(PyRFQCell(cell_type=cell_type,
                                                 V=params[0] * 1000.0,
                                                 Wsyn=params[1],
                                                 Sig0T=params[2],
                                                 Sig0L=params[3],
                                                 A10=params[4],
                                                 Phi=params[5],
                                                 a=params[6] * 0.01,
                                                 m=params[7],
                                                 B=params[8],
                                                 L=params[9] * 0.01,
                                                 prev_cell=pc,
                                                 debug=self._debug))
                else:

                    self._cells.append(PyRFQCell(cell_type=cell_type,
                                                 V=params[0] * 1000.0,
                                                 Wsyn=params[1],
                                                 Sig0T=params[2],
                                                 Sig0L=params[3],
                                                 A10=params[4],
                                                 Phi=params[5],
                                                 a=params[6] * 0.01,
                                                 m=params[7],
                                                 B=params[8],
                                                 L=params[9] * 0.01,
                                                 A0=params[11],
                                                 RFdef=params[12],
                                                 Oct=params[13],
                                                 A1=params[14],
                                                 prev_cell=pc,
                                                 debug=self._debug))

                if len(self._cells) > 1:
                    self._cells[-2].set_next_cell(self._cells[-1])

        self._cell_nos = range(len(self._cells))
        self._length = np.sum([cell.length for cell in self._cells])

        return 0

    def read_input_vecc(self, filename, ignore_rms):
        print("Loading from VECC files is currently not supported (function needs to be mofernized)!")
        exit(1)

        with open(filename, "r") as infile:

            for line in infile:

                params = [float(item) for item in line.strip().split()]

                if params[4] == 1.0:
                    cell_type = "rms"
                    if ignore_rms:
                        continue
                else:
                    cell_type = "regular"

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

        # TODO: Replace gradient with something that accepts mask
        _d = self._variables_bempp["d"]
        phi_masked = np.ma.masked_array(self._variables_bempp["ef_phi"],
                                        mask=self._variables_bempp["ef_mask"])
        ex, ey, ez = np.gradient(phi_masked,
                                 _d[X], _d[Y], _d[Z])

        if RANK == 0:

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

        mpi_data = COMM.bcast(mpi_data, root=0)

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

        _mesh_data = self._full_mesh
        _n_data = self._variables_bempp["n_fun_coeff"]
        # _d_data = self._variables_bempp["d_fun_coeff"]

        assert _mesh_data is not None and _n_data is not None, \
            "One of grid, dirichlet_function, neumann_function is None!"

        _ts = time.time()

        self.message("Re-Generating Grid, GridFunctions, and FunctionSpace")

        _mesh = bempp.api.grid.grid_from_element_data(_mesh_data["verts"], _mesh_data["elems"], _mesh_data["domns"])
        dp0_space = bempp.api.function_space(_mesh, "DP", 0)
        n_fun = bempp.api.GridFunction(dp0_space, coefficients=_n_data)
        # d_fun = bempp.api.GridFunction(dp0_space, coefficients=_d_data)
        # dp0_space = self._variables_bempp["dp0_space"]
        # p1_space = self._variables_bempp["p1_space"]

        self.message("Re-Generating took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - _ts)))))

        # noinspection PyUnresolvedReferences
        all_vert = self._full_mesh["verts"]

        # get limits from electrodes
        limits_elec = np.array([[np.min(all_vert[i, :]), np.max(all_vert[i, :])] for i in XYZ])

        # replace None limits with electrode limits
        limits[np.where(limits is None)] = limits_elec[np.where(limits is None)]

        res = np.array([res]).ravel()
        _n = np.array(np.round((limits[:, 1] - limits[:, 0]) / res, 10), int) + 1

        # Recalculate resolution to match integer n's
        _d = (limits[:, 1] - limits[:, 0]) / (_n - 1)

        # Generate a full mesh to be indexed later
        _r = np.array([np.linspace(limits[i, 0], limits[i, 1], _n[i]) for i in XYZ])
        mesh = np.meshgrid(_r[X], _r[Y], _r[Z], indexing='ij')  # type: np.ndarray

        # Initialize potential array
        pot = np.zeros(mesh[0].shape)

        # Index borders (can be float)
        borders = np.array([np.linspace(0, _n[i], domain_decomp[i] + 1) for i in XYZ])

        # Indices (must be int)
        # note: rounding will likely lead to domains that are off in size by one index, but that's fine
        start_idxs = np.array([np.array(borders[i][:-1], int) - overlap for i in XYZ])
        end_idxs = np.array([np.array(borders[i][1:], int) + overlap for i in XYZ])

        for i in XYZ:
            start_idxs[i][0] = 0
            end_idxs[i][-1] = int(borders[i][-1])

        # Print out domain information
        if RANK == 0 and self._debug:
            print("Potential Calculation. "
                  "Grid spacings: ({:.4f}, {:.4f}, {:.4f}), number of meshes: {}".format(_d[0], _d[1], _d[2], _n))
            print("Number of Subdomains: {}, "
                  "Domain decomposition {}:".format(np.product(domain_decomp), domain_decomp))

            for i, dirs in enumerate(["x", "y", "z"]):
                print("{}: Indices {} to {}".format(dirs, start_idxs[i], end_idxs[i] - 1))

        # Calculate mask (True if inside/on surface of an electrode)
        all_grid_pts = np.vstack([_mesh.ravel() for _mesh in mesh]).T
        mymask = np.zeros(all_grid_pts.shape[0], dtype=bool)

        _ts = time.time()
        if RANK == 0:
            print("\n*** Calculating mask for {} points ***".format(all_grid_pts.shape[0]))

        for _elec_object in self._elec_objects:

            self.message("[{}] Working on electrode object {}".format(
                time.strftime('%H:%M:%S', time.gmtime(int(time.time() - _ts))), _elec_object.name))

            mymask = mymask | _elec_object.points_inside(all_grid_pts)

        # Number of masked points
        n_masked = np.where(mymask is True)[0].shape[0]

        # Reshape mask to match original mesh
        mymask = mymask.T.reshape(mesh[0].shape)

        self.message("Generating mask took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - _ts)))))
        self.message("\n*** Calculating potential for {} points ***".format(all_grid_pts.shape[0] - n_masked))

        _ts = time.time()

        # Iterate over all the dimensions, calculate the subset of potential
        if RANK == 0:
            domain_idx = 1
            for x1, x2 in zip(start_idxs[X], end_idxs[X]):
                for y1, y2 in zip(start_idxs[Y], end_idxs[Y]):
                    for z1, z2 in zip(start_idxs[Z], end_idxs[Z]):

                        # Create mask subset for this set of points and only calculate those
                        local_mask = mymask[x1:x2, y1:y2, z1:z2].ravel()
                        grid_pts = np.vstack([_mesh[x1:x2, y1:y2, z1:z2].ravel() for _mesh in mesh])
                        grid_pts_len = grid_pts.shape[1]  # save shape for later
                        grid_pts = grid_pts[:, ~local_mask]  # reduce for faster calculation

                        self.message(
                            "[{}] Domain {}/{}, Index Limits: x = ({}, {}), y = ({}, {}), z = ({}, {})".format(
                                time.strftime('%H:%M:%S', time.gmtime(int(time.time() - _ts))), domain_idx,
                                np.product(domain_decomp), x1, x2 - 1, y1, y2 - 1, z1, z2 - 1))
                        self.message("Removed {} points due to mask".format(grid_pts_len - grid_pts.shape[1]))

                        temp_pot = bempp.api.operators.potential.laplace.single_layer(dp0_space, grid_pts) * n_fun

                        # Create array of original shape and fill with result at right place,
                        # then move into master array
                        _pot = np.zeros(grid_pts_len)

                        _pot[~local_mask] = temp_pot[0]
                        pot[x1:x2, y1:y2, z1:z2] = _pot.reshape([x2 - x1, y2 - y1, z2 - z1])

                        domain_idx += 1

            try:

                del grid_pts
                del _pot
                del temp_pot

            except Exception as _e:

                print("Exception {} happened, but trying to carry on...".format(_e))

        # TODO: Distribute results to other nodes -DW

        self._variables_bempp["ef_phi"] = pot
        self._variables_bempp["ef_mask"] = mymask
        self._variables_bempp["d"] = _d
        self._variables_bempp["n"] = _n
        self._variables_bempp["limits"] = limits

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

        return {"phi": self._variables_bempp["ef_phi"],
                "mask": self._variables_bempp["ef_mask"],
                "limits": self._variables_bempp["limits"],
                "d": self._variables_bempp["d"],
                "n": self._variables_bempp["n"]}

    def solve_bempp(self):

        if self._full_mesh is None:
            print("No mesh generated, trying now...")
            mesh = self.generate_full_mesh()
        else:
            mesh = bempp.api.grid.grid_from_element_data(self._full_mesh["verts"],
                                                         self._full_mesh["elems"],
                                                         self._full_mesh["domns"])

        dp0_space = bempp.api.function_space(mesh, "DP", 0)

        domain_mapping = {}
        for _elec_obj in self._elec_objects:
            domain_mapping[_elec_obj.domain_idx] = _elec_obj.voltage

        def f(*args):
            domain_index = args[2]
            result = args[3]
            result[0] = domain_mapping[domain_index]

        dirichlet_fun = bempp.api.GridFunction(dp0_space, fun=f)
        self._variables_bempp["d_fun_coeff"] = dirichlet_fun.coefficients

        if self._debug and RANK == 0:
            dirichlet_fun.plot()

        # Solve BEMPP problem only on 1 cpu (has internal multiprocessing)
        if RANK == 0:
            slp = bempp.api.operators.boundary.laplace.single_layer(dp0_space, dp0_space, dp0_space)
            neumann_fun, info = bempp.api.linalg.gmres(slp, dirichlet_fun, tol=1e-6, use_strong_form=True)
            mpi_data = {"n_coeff": neumann_fun.coefficients,
                        "info": info}
        else:
            mpi_data = None

        mpi_data = COMM.bcast(mpi_data, root=0)

        COMM.barrier()

        self._variables_bempp["n_fun_coeff"] = mpi_data["n_coeff"]

        return 0

    def initialize(self):

        assert len(self._cells) > 0, "No cells have been added, no vanes can be generated."

        # Delete all existing vane objects
        self._elec_objects = [_elec_obj for _elec_obj in self._elec_objects if "vane" not in _elec_obj.name]

        # There are four vanes (rods) in the RFQ
        # x = horizontal, y = vertical, with p, m denoting positive and negative axis directions
        # but they are symmetric, so we createonly two and copy them later
        local_vanes = [PyRFQVane(parent=self,
                                 vane_type="yp",
                                 cells=self._cells,
                                 voltage=self._voltage * 0.5,  # Given voltage is 'inter-vane'
                                 occ_tolerance=self._occ_tolerance,
                                 occ_npart=15,
                                 h=self._variables_bempp["grid_res"],
                                 debug=self._debug),

                       PyRFQVane(parent=self,
                                 vane_type="xp",
                                 cells=self._cells,
                                 voltage=-self._voltage * 0.5,  # Given voltage is 'inter-vane'
                                 occ_tolerance=self._occ_tolerance,
                                 occ_npart=15,
                                 h=self._variables_bempp["grid_res"],
                                 debug=self._debug)]

        for _vane in local_vanes:
            _vane.set_mesh_parameter("r_tip", self.get_geometry_parameter("vane_radius"))
            _vane.set_mesh_parameter("h_type", self.get_geometry_parameter("vane_height_type"))
            _vane.set_mesh_parameter("h_block", self.get_geometry_parameter("vane_height"))
            _vane.set_mesh_parameter("refine_steps", self.get_bempp_parameter("refine_steps"))
            _vane.set_mesh_parameter("reverse_mesh", self.get_bempp_parameter("reverse_mesh"))
            _vane.set_mesh_parameter("nz", self.get_geometry_parameter("nz"))

        # Generate the two vanes in parallel:
        if MPI is None or SIZE == 1:
            if USE_MULTIPROC:
                p = Pool(2)
                local_vanes = p.map(self._worker_generate_vane_profile, local_vanes)
            else:
                for i, _vane in enumerate(local_vanes):
                    local_vanes[i] = self._worker_generate_vane_profile(_vane)

        else:
            if RANK == 0:
                self.message("Proc {} working on vane {}".format(RANK, local_vanes[0].vane_type), rank=RANK)
                _vane = self._worker_generate_vane_profile(local_vanes[0])
                mpi_data = {"vanes": [_vane, COMM.recv(source=1)]}
            elif RANK == 1:
                self.message("Proc {} working on vane {}".format(RANK, local_vanes[1].vane_type), rank=RANK)
                _vane = self._worker_generate_vane_profile(local_vanes[1])
                COMM.send(_vane, dest=0)
                mpi_data = None
            else:
                if self._debug:
                    self.message("Proc {} idle.".format(RANK), rank=RANK)
                mpi_data = None

            mpi_data = COMM.bcast(mpi_data, root=0)
            local_vanes = mpi_data["vanes"]

        # --- Now make copies, set vane_type
        self.message("Copying vanes...")

        for i, vane_type in enumerate(["ym", "xm"]):
            new_vane = copy.deepcopy(local_vanes[i])  # First one is y direction
            new_vane.set_vane_type(vane_type)

            local_vanes.append(new_vane)

        for _vane in local_vanes:
            self._elec_objects.append(_vane)

        self._initialized = True

        COMM.barrier()  # TODO: Necessary?

        return 0

    def generate_full_mesh(self):

        assert HAVE_BEMPP, "Can only create the full mesh with BEMPP at the moment!"

        # TODO: Better assertion/init/generate
        # assert self._vanes is not None, "No vanes generated yet, cannot mesh..."
        assert len(self._elec_objects) != 0, "Need at least one electrode object to generate full mesh!"

        # Initialize empty arrays of the correct shape (3 x n)
        vertices = np.zeros([3, 0])
        elements = np.zeros([3, 0])
        vertex_counter = 0
        domains = np.zeros([0], int)

        # For now, do this only on the first node
        if RANK == 0:

            # Trying to do this with gmsh, but running into trouble with identical surface numbers
            # self._full_mesh_fn = os.path.join(self._temp_dir, "full_rfq.msh")
            # command = [GMSH_EXE, '-merge']
            #
            # for _elec in self._elec_objects:
            #     command.append(_elec.mesh_fn)
            #
            # command.append('-o')
            # command.append(self._full_mesh_fn)
            # command.append('-format')
            # command.append('msh2')
            # command.append('-save')
            #
            # output = subprocess.run(command)
            #
            # if self._debug:
            #     print(output)

            for _elec in self._elec_objects:
                mesh = bempp.api.import_grid(_elec.mesh_fn)

                _vertices = mesh.leaf_view.vertices
                _elements = mesh.leaf_view.elements
                _domain_ids = mesh.leaf_view.domain_indices

                vertices = np.concatenate((vertices, _vertices), axis=1)
                elements = np.concatenate((elements, _elements + vertex_counter), axis=1)
                domains = np.concatenate((domains, _domain_ids), axis=0)

                # Increase the running counters
                vertex_counter += _vertices.shape[1]

            self._full_mesh = {"verts": vertices,
                               "elems": elements,
                               "domns": domains}

            if self._debug:
                bempp.api.grid.grid_from_element_data(vertices,
                                                      elements,
                                                      domains).plot()

        # Broadcast results to all nodes
        self._full_mesh = COMM.bcast(self._full_mesh, root=0)

        COMM.barrier()

        return self._full_mesh

    def generate_geo_str(self):

        # Check if RFQ has been initialized
        if not self._initialized:
            print("RFQ needs to be initialized, attempting now...")

        # if MPI is None or SIZE == 1:
        if RANK == 0:
            # no mpi or single core: use python multiprocessing and at least have threads for speedup
            if USE_MULTIPROC:
                p = Pool()
                _elec_objects = p.map(self._worker_generate_geo_str, self._elec_objects)
            else:
                _elec_objects = []
                for i, _elec_obj in enumerate(self._elec_objects):
                    _elec_objects.append(self._worker_generate_geo_str(_elec_obj))

            mpi_data = {"elecs": _elec_objects}

        else:
            mpi_data = None

        self._elec_objects = COMM.bcast(mpi_data, root=0)["elecs"]
        COMM.barrier()

        # TODO: MPI-ify this?
        # elif SIZE >= 4:
        #     # We have 4 or more procs and can use a single processor per vane
        #
        #     if RANK <= 3:
        #         # Generate on proc 0-3
        #         print("Proc {} working on electrode {}.".format(RANK + 1, self._elec_objects[RANK].name))
        #         sys.stdout.flush()
        #         _vane = self._worker_generate_geo_str(self._elec_objects[RANK])
        #
        #         if RANK == 0:
        #             # Assemble on proc 0
        #             mpi_data = {"vanes": [_vane,
        #                                   COMM.recv(source=1),
        #                                   COMM.recv(source=2),
        #                                   COMM.recv(source=3)]}
        #         else:
        #             COMM.send(_vane, dest=0)
        #             mpi_data = None
        #
        #     else:
        #         print("Proc {} idle.".format(RANK + 1))
        #         sys.stdout.flush()
        #         mpi_data = None
        #
        #     # Distribute
        #     self._elec_objects = COMM.bcast(mpi_data, root=0)["vanes"]
        #     COMM.barrier()

        # else:
        #     # We have 2 or 3 procs, so do two vanes each on proc 0 and proc 1
        #     if RANK <= 1:
        #         # Generate on proc 0, 1
        #         print("Proc {} working on vanes {} and {}.".format(RANK + 1,
        #                                                            self._elec_objects[RANK].vane_type,
        #                                                            self._elec_objects[RANK + 2].vane_type))
        #         sys.stdout.flush()
        #         local_vanes = [self._worker_generate_geo_str(self._elec_objects[RANK]),
        #                        self._worker_generate_geo_str(self._elec_objects[RANK + 2])]
        #
        #         if RANK == 0:
        #             # Assemble on proc 0
        #             other_vanes = COMM.recv(source=1)
        #             mpi_data = {"vanes": [local_vanes[0],
        #                                   other_vanes[0],
        #                                   local_vanes[1],
        #                                   other_vanes[1]]}
        #         else:
        #             COMM.send(local_vanes, dest=0)
        #             mpi_data = None
        #
        #     else:
        #         print("Proc {} idle.".format(RANK + 1))
        #         sys.stdout.flush()
        #         mpi_data = None
        #
        #     # Distribute
        #     self._elec_objects = COMM.bcast(mpi_data, root=0)["vanes"]
        #     COMM.barrier()

        self._have_geo_str = True

        return 0

    def generate_gmsh_files(self):

        if not HAVE_GMSH:
            print("Gmsh could not be found (check path?) not creating msh files and brep.")
            return 1

        # Check for existing geo string
        if not self._have_geo_str:
            print("No geo string has been generated, attemting to do so now...")
            self.generate_geo_str()

        if RANK == 0:
            print("Generating gmsh files of the electrodes.")
            sys.stdout.flush()

            # no mpi or single core: use python multiprocessing and at least have threads for speedup
            if USE_MULTIPROC:
                p = Pool()
                _elec_objects = p.map(self._worker_generate_gmsh_files, self._elec_objects)
            else:
                _elec_objects = []
                for i, _elec_obj in enumerate(self._elec_objects):
                    _elec_objects.append(self._worker_generate_gmsh_files(_elec_obj))

            mpi_data = {"elecs": _elec_objects}

        else:
            mpi_data = None

        self._elec_objects = COMM.bcast(mpi_data, root=0)["elecs"]
        COMM.barrier()

        # if MPI is None or SIZE == 1:
        #     # no mpi or single core: use python multiprocessing and at least have threads for speedup
        #     if USE_MULTIPROC:
        #         p = Pool()
        #         self._elec_objects = p.map(self._worker_generate_gmsh_files, self._elec_objects)
        #     else:
        #         for i, _vane in enumerate(self._elec_objects):
        #             self._elec_objects[i] = self._worker_generate_gmsh_files(_vane)
        #
        # elif SIZE >= 4:
        #     # We have 4 or more procs and can use a single processor per vane
        #
        #     if RANK <= 3:
        #         # Generate on proc 0-3
        #         print("Proc {} working on vane {}.".format(RANK + 1, self._elec_objects[RANK].vane_type))
        #         sys.stdout.flush()
        #         _vane = self._worker_generate_gmsh_files(self._elec_objects[RANK])
        #
        #         if RANK == 0:
        #             # Assemble on proc 0
        #             mpi_data = {"vanes": [_vane,
        #                                   COMM.recv(source=1),
        #                                   COMM.recv(source=2),
        #                                   COMM.recv(source=3)]}
        #         else:
        #             COMM.send(_vane, dest=0)
        #             mpi_data = None
        #
        #     else:
        #         print("Proc {} idle.".format(RANK + 1))
        #         sys.stdout.flush()
        #         mpi_data = None
        #
        #     # Distribute
        #     self._elec_objects = COMM.bcast(mpi_data, root=0)["vanes"]
        #     COMM.barrier()
        #
        # else:
        #     # We have 2 or 3 procs, so do two vanes each on proc 0 and proc 1
        #     if RANK <= 1:
        #         # Generate on proc 0, 1
        #         print("Proc {} working on vanes {} and {}.".format(RANK + 1,
        #                                                            self._elec_objects[RANK].vane_type,
        #                                                            self._elec_objects[RANK + 2].vane_type))
        #         sys.stdout.flush()
        #         local_vanes = [self._worker_generate_gmsh_files(self._elec_objects[RANK]),
        #                        self._worker_generate_gmsh_files(self._elec_objects[RANK + 2])]
        #
        #         if RANK == 0:
        #             # Assemble on proc 0
        #             other_vanes = COMM.recv(source=1)
        #             mpi_data = {"vanes": [local_vanes[0],
        #                                   other_vanes[0],
        #                                   local_vanes[1],
        #                                   other_vanes[1]]}
        #         else:
        #             COMM.send(local_vanes, dest=0)
        #             mpi_data = None
        #
        #     else:
        #         print("Proc {} idle.".format(RANK + 1))
        #         sys.stdout.flush()
        #         mpi_data = None
        #
        #     # Distribute
        #     self._elec_objects = COMM.bcast(mpi_data, root=0)["vanes"]
        #     COMM.barrier()

        return 0

    def generate_occ(self):

        # Unfortunately, multiprocessing/MPI can't handle SwigPyObject objects
        for _elec_object in self._elec_objects:

            _elec_object.generate_occ()

        return 0

    def plot_vane_profile(self):

        _vanes = [_elec_obj for _elec_obj in self._elec_objects if "vane" in _elec_obj.name]

        assert len(_vanes) != 0, "No vanes calculated yet!"

        _fig, _ax = plt.subplots()

        for vane in _vanes:
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

    @staticmethod
    def message(*args, rank=0):
        if RANK == rank:
            print(*args)
            sys.stdout.flush()
        return 0

    def reset(self):

        self._vanes = []
        self._elec_objects = []
        self._full_mesh = None
        self._have_geo_str = False
        self._have_occ_obj = False
        self._have_bem_obj = False

        self._initialized = False

        return 0

    def save_to_file(self, filename=None):

        # if RANK == 0:
        #     if filename is None:
        #         filename = FileDialog().get_filename(action="save")

        for key, item in self.__dict__.items():
            print(key, ":", item)

        # with open(filename, "wb") as of:
        #     pass

        return 0

    @staticmethod
    def _worker_generate_gmsh_files(electrode):

        electrode.generate_gmsh_files()

        return electrode

    @staticmethod
    def _worker_generate_geo_str(electrode):

        electrode.generate_geo_str()

        return electrode

    def _worker_generate_vane_profile(self, vane):

        vane.calculate_profile(fudge=self._fudge_vanes)

        return vane

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
            assert key in self._variables_gmtry.keys(), "write_inventor_macro: Unrecognized kwarg '{}'".format(key)
            self._variables_gmtry[key] = value

        assert self._variables_gmtry["vane_type"] != "rod", "vane_type == 'rod' not implemented yet. Aborting"

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
""".format(self._variables_gmtry["vane_radius"] * 100.0,
           self._variables_gmtry["vane_height"] * 100.0)

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
    ' Note: I am not sure if keeping the profile perpendicular to the path is more accurate, 
    ' but unfortunately for trapezoidal cells (small fillets) it doesn't work
    ' so it has to be a 'parallel to original profile' kinda sweep -- or not? , kParallelToOriginalProfile
    Set oSweep = oCompDef.Features.SweepFeatures.AddUsingPath(oProfile, oPath, kJoinOperation)
"""

            # Small modification depending on absolute or relative vane height:
            if self._variables_gmtry["vane_height_type"] == 'relative':
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
                            z, x = vane.get_profile(nz=self._variables_gmtry["nz"])
                            min_x = np.min(x)
                            max_x = np.max(x)
                            z_start = np.min(z)
                            z_end = np.max(z)

                    for _x, _z in zip(x, z):
                        outfile.write("{:.6f}, {:.6f}, {:.6f}\r\n".format(
                            _x * 100.0,  # For some weird reason Inventor uses cm as default...
                            0.0,
                            _z * 100.0))

                else:
                    for vane in self._vanes:
                        if vane.vane_type == "yp":
                            z, y = vane.get_profile(nz=self._variables_gmtry["nz"])
                            min_y = np.min(y)
                            max_y = np.max(y)

                    for _y, _z in zip(y, z):
                        outfile.write("{:.6f}, {:.6f}, {:.6f}\r\n".format(
                            0.0,
                            _y * 100.0,  # For some weird reason Inventor uses cm as default...
                            _z * 100.0))

        # Write an info file with some useful information:
        with open(os.path.join(save_folder, "Info.txt"), "w") as outfile:

            datestr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            outfile.write("Inventor Macros and Profile generated on {}\n\n".format(datestr))

            outfile.write("Parameters:\n")
            for key, value in self._variables_gmtry.items():
                outfile.write("{}: {}\n".format(key, value))

            if self._variables_gmtry["vane_height_type"] == 'absolute':
                max_extent_x = max_extent_y = self._variables_gmtry["vane_height"]
            else:
                max_extent_x = self._variables_gmtry["vane_height"] + min_x
                max_extent_y = self._variables_gmtry["vane_height"] + min_y

            outfile.write("\nOther useful values:\n")
            outfile.write("Maximum Extent in X: {} m\n".format(max_extent_x))
            outfile.write("Maximum Extent in Y: {} m\n".format(max_extent_y))
            outfile.write("Z Start: {} m\n".format(z_start))
            outfile.write("Z End: {} m\n".format(z_end))

        return 0


if __name__ == "__main__":

    mydebug = False
    myfn = "PARMTEQOUT_mod.TXT"

    r_vane = 0.0093
    h_vane = 0.05
    nz = 750
    grid_res = 0.005
    pot_res = 0.005

    # --- Jungbae's RFQ Design with RMS section
    myrfq = PyRFQ(voltage=22000.0, fudge_vanes=True, debug=mydebug)

    # myrfq.append_cell(cell_type="start",
    #                   aperture=0.009709,
    #                   modulation=1.0,
    #                   length=0.0)

    # Load the base RFQ design from the parmteq file
    if myrfq.add_cells_from_file(filename=myfn,
                                 ignore_rms=False) != 0:
        exit()

    # Transition cell
    myrfq.append_cell(cell_type="transition_auto",
                      a='auto',
                      m='auto',
                      L='auto')

    myrfq.append_cell(cell_type="drift",
                      a='auto',
                      L=0.02)

    # Trapezoidal Rebunching Cell
    # TODO: Maybe frame TRC in TCS's?
    myrfq.append_cell(cell_type="trapezoidal",
                      a='auto',
                      m=1.5,
                      L=0.075,
                      fillet_radius=2 * r_vane)  # Needs to be larger than r_vane for sweep

    myrfq.append_cell(cell_type="drift",
                      a='auto',
                      L=0.02)

    myrfq.append_cell(cell_type="rms",
                      a=0.009718,
                      L=0.018339)

    myrfq.append_cell(cell_type="rms",
                      a=0.010944,
                      L=0.018339)

    myrfq.append_cell(cell_type="rms",
                      a=0.016344,
                      L=0.018339)

    myrfq.append_cell(cell_type="rms",
                      a=0.041051,
                      L=0.018339)

    myrfq.append_cell(cell_type="rms",
                      a=0.15,
                      L=0.018339)

    myrfq.message(myrfq)  # = print(myrfq), message without kwargs prints on RANK 0 and flushes output immediately.

    # TODO: Idea: Make ElectrodeObject class from which other electrodes inherit?
    # TODO: Idea: Make ElectrostaticSolver class that can be reused (e.g. for Spiral Inflector)?
    # myrfq.set_bempp_parameter("add_endplates", True)  # TODO: Correct handling of OCC objects for endplates
    # myrfq.set_bempp_parameter("cyl_id", 0.12)
    myrfq.set_bempp_parameter("reverse_mesh", True)
    myrfq.set_bempp_parameter("grid_res", grid_res)  # characteristic mesh size during initial meshing
    myrfq.set_bempp_parameter("refine_steps", 0)  # number of times gmsh is called to "refine by splitting"

    myrfq.set_geometry_parameter("vane_radius", r_vane)
    myrfq.set_geometry_parameter("vane_height", h_vane)
    myrfq.set_geometry_parameter("vane_height_type", 'absolute')

    # Steps along z for spline interpolation of vane profile
    # Cave: Needs to be fairly high-res to resolve trapezoidal cells
    myrfq.set_geometry_parameter("nz", nz)

    myrfq.add_endplates(gap_sta=0.025,
                        gap_end=0.01,
                        thickness=0.02,
                        plate_dia=0.2,
                        voltage=0.0,
                        aperture_dia=0.02)

    myrfq.message("Initializing vanes...")
    ts = time.time()
    myrfq.initialize()
    myrfq.message("Initializing vanes took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))

    # if RANK == 0:
    #     myrfq.plot_vane_profile()
    #     myrfq.write_inventor_macro(vane_type='vane',
    #                                vane_radius=r_vane,
    #                                vane_height=h_vane,
    #                                vane_height_type='absolute',
    #                                nz=nz)

    myrfq.generate_geo_str()
    myrfq.generate_gmsh_files()

    # if RANK == 0:
    #     import pickle
    #     with open("full_rfq_save.pickle", "wb") as of:
    #         pickle.dump(myrfq, of)
    #
    # import pickle
    # with open("full_rfq_save.pickle", "rb") as inf:
    #     myrfq = pickle.load(inf)

    myrfq.message("Loading and assembling full mesh for BEMPP")
    ts = time.time()
    myrfq.generate_full_mesh()
    myrfq.message("Assembling mesh took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))

    # Increase eps
    bempp.api.global_parameters.hmat.eps = 1e-5

    # Increase quadrature orders
    # bempp.api.global_parameters.quadrature.near.single_order = 5
    # bempp.api.global_parameters.quadrature.near.double_order = 5
    # bempp.api.global_parameters.quadrature.medium.single_order = 4
    # bempp.api.global_parameters.quadrature.medium.double_order = 4
    # bempp.api.global_parameters.quadrature.far.single_order = 3
    # bempp.api.global_parameters.quadrature.far.double_order = 3

    myrfq.message("Solving BEMPP problem...")
    ts = time.time()
    myrfq.solve_bempp()
    myrfq.message("Solving BEMPP took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))

    if RANK == 0:
        import pickle
        with open("full_rfq_save.pickle", "wb") as of:
            pickle.dump(myrfq, of)

    # Increase hmat block size
    bempp.api.global_parameters.hmat.min_block_size = 1000

    myrfq.message("Generating OCC objects...")
    ts = time.time()
    myrfq.generate_occ()
    myrfq.message("Generating OCC objects took {}".format(time.strftime('%H:%M:%S',
                                                                        time.gmtime(int(time.time() - ts)))))

    myrfq.message("Calculating Mask & Potential...")
    ts = time.time()
    myres = [pot_res, pot_res, pot_res]
    rlim = 0.02
    xlims = (-rlim, rlim)
    ylims = (-rlim, rlim)
    zlims = (-0.05, 1.45)
    myrfq.calculate_potential(limits=(xlims, ylims, zlims),
                              res=myres,
                              domain_decomp=(1, 1, 10),
                              overlap=0)
    myrfq.message("Mask & Potential took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))

    if RANK == 0:
        import pickle

        with open("ef_phi.field", "wb") as outfile:
            pickle.dump(myrfq.get_phi(), outfile)

    # ts = time.time()
    # myrfq.calculate_efield()
    # if RANK == 0:
    #     print("Field took {}".format(time.strftime('%H:%M:%S', time.gmtime(int(time.time() - ts)))))
    #
    # if RANK == 0:
    #     myrfq.plot_combo(xypos=0.005, xyscale=1.0, zlim=(-0.1, 1.35))
    #
    # import pickle
    #
    # with open("efield_out.field", "wb") as outfile:
    #     pickle.dump(myrfq._variables_bempp["ef_itp"], outfile)

    # myrfq.plot_vane_profile()
