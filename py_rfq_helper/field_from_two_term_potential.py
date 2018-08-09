from dans_pymodules import *
from scipy.interpolate import interp1d
from scipy import meshgrid
from scipy.special import iv as bessel1
from scipy.optimize import root
import platform
import gc
import time

np.set_printoptions(threshold=10000)

__author__ = "Siddhartha Dechoudhury, Daniel Winklehner"
__doc__ = """Calculate RFQ fields from loaded cell parameters using the two-term potential"""

# Initialize some global constants
amu = const.value("atomic mass constant energy equivalent in MeV")
echarge = const.value("elementary charge")
clight = const.value("speed of light in vacuum")

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


class Vane(object):
    def __init__(self,
                 vane_type=None,
                 mesh_size=0.001,
                 curvature=None,
                 height=0.001,
                 vane_z_data=None,
                 vane_y_data=None,
                 rotation=0.0,
                 debug=False
                 ):

        self._debug = debug

        self._vane_type = vane_type
        self._curvature = curvature
        self._rotation = rotation

        # Additional straight extension beyond maximum transversal extent
        # Always falls back to at least 1 mm to avoid infinitely thin materials
        if height < 0.001:
            self._height = 0.001
        else:
            self._height = height

        self._vane_z_data = vane_z_data
        self._vane_y_data = vane_y_data

        self._mesh_size = mesh_size
        self._grid = None

    def __add__(self, other):

        assert self.has_grid() and other.has_grid(), \
            "Can only add two vanes with computed grids. Use generate_grid() first!"

        # TODO: Collision detection???

        vertices1, elements1 = self._grid.leaf_view.vertices, self._grid.leaf_view.elements
        no_vert_grid1 = vertices1.shape[1]

        vertices2, elements2 = other.get_grid().leaf_view.vertices, other.get_grid().leaf_view.elements

        vertices = np.append(vertices1, vertices2, axis=1)
        elements = np.append(elements1, elements2 + no_vert_grid1, axis=1)

        # Adding together two Vane objects makes vane_z_data and vane_y_data obsolete
        new_vane = Vane(vane_type=self._vane_type,
                        mesh_size=self._mesh_size,
                        curvature=self._curvature,
                        height=self._height,
                        vane_z_data=None,
                        vane_y_data=None,
                        debug=self._debug)

        import bempp.api

        new_vane.set_grid(bempp.api.grid.grid_from_element_data(vertices, elements))

        return new_vane

    def generate_grid(self, mesh_size=None):

        if mesh_size is not None:
            self._mesh_size = mesh_size

        zmin = np.min(self._vane_z_data)
        zmax = np.max(self._vane_z_data)
        numz = np.round((zmax - zmin) / self._mesh_size, 0) + 1

        if numz == len(self._vane_z_data):

            vane_z_data = self._vane_z_data
            vane_y_data = self._vane_y_data

        else:
            vane_interp = interp1d(self._vane_z_data, self._vane_y_data)
            vane_z_data = np.linspace(zmin, zmax, numz)
            vane_y_data = vane_interp(vane_z_data)

        if self._vane_type == "semi-circle":
            # Calculate approximate angular resolution corresponding to desired mesh size
            num_phi = np.round(self._curvature * np.pi / self._mesh_size, 0)
            phi = np.pi / num_phi

            print("With mesh_size {} m, we have {} points per semi-circle".format(self._mesh_size, num_phi))

            # We need two sets of phi values so that subsequent z positions form triangles rather than squares
            phi_set = [np.linspace(np.pi, 2.0 * np.pi, num_phi),
                       np.linspace(np.pi + 0.5 * phi, 2.0 * np.pi - 0.5 * phi, num_phi - 1)]

            # maximum vertical extent:
            ymax = self._curvature + np.max(vane_y_data) + self._height
            print("Maximum vertical extent: {} m".format(ymax))
            # TODO: Think about whether it's a good idea to use only half the mesh size on block part
            num_vert_pts = int(np.round((ymax - self._curvature - np.min(vane_y_data)) / 2.0 / self._mesh_size, 0))
            print("Number of points in vertical direction: {}".format(num_vert_pts))
            num_horz_pts = int(np.round(2.0 * self._curvature / 2.0 / self._mesh_size, 0)) - 1  # Not counting corners
            horz_step_size = (2.0 * self._curvature) / (num_horz_pts + 1)

            # Create point cloud for testing
            points = []
            for i, _z in enumerate(vane_z_data):

                # Create curved surface points:
                for _phi in phi_set[int(i % 2.0)]:
                    points.append((self._curvature * np.cos(_phi),
                                   self._curvature * np.sin(_phi) + self._curvature + vane_y_data[i],
                                   _z))

                # Create straight lines points:
                # Looking with z, we start at the right end of the curvature and go up-left-down
                vert_step = (ymax - self._curvature - vane_y_data[i]) / num_vert_pts

                # If we are in phi_set 0, we start one step up, in phi set 1 a half step up
                for j in range(num_vert_pts):
                    points.append((self._curvature, points[-1][1] + vert_step, _z))

                if i % 2.0 == 0.0:
                    points.pop(-1)

                # Append the corner point
                points.append((self._curvature, ymax, _z))

                if i % 2.0 == 0.0:
                    for j in range(num_horz_pts + 1):
                        points.append((self._curvature + 0.5 * horz_step_size - (j + 1) * horz_step_size, ymax, _z))
                else:
                    for j in range(num_horz_pts):
                        points.append((self._curvature - (j + 1) * horz_step_size, ymax, _z))

                # Append the corner point
                points.append((-self._curvature, ymax, _z))

                # Finish up by going down on the left side
                for j in range(num_vert_pts):
                    points.append((-self._curvature, ymax - (j + 1) * vert_step + 0.5 * (i % 2.0) * vert_step, _z))

                if i % 2.0 == 0.0:
                    points.pop(-1)

            points = np.array(points, dtype=[("x", float), ("y", float), ("z", float)])

            # Apply rotation
            if self._rotation != 0.0:

                angle = self._rotation * np.pi / 180.0

                x_old = points["x"].copy()
                y_old = points["y"].copy()

                points["x"] = x_old * np.cos(angle) - y_old * np.sin(angle)
                points["y"] = x_old * np.sin(angle) + y_old * np.cos(angle)

                del x_old
                del y_old

            points_per_slice = len(np.where(points["z"] == vane_z_data[0])[0])
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

            # get first slice indices
            # idx = np.where(points[2] == vane_z_data[0])
            # plt.scatter(points[0][idx], points[1][idx], color="blue")

            # get second slice indices
            # idx = np.where(points[2] == vane_z_data[1])
            # plt.scatter(points[0][idx], points[1][idx], color="red")
            #
            # plt.gca().set_aspect('equal')

            # from mpl_toolkits.mplot3d import Axes3D
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # for element in elements:
            #     triangle = np.append(points[element], points[element[0]])
            #     ax.plot(triangle["x"], triangle["y"], triangle["z"])
            # # for i, point in enumerate(points):
            # #     ax.scatter(point["x"], point["y"], point["z"], marker=r"${}$".format(i), s=49)
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_aspect("equal")
            # plt.show()
            #
            # exit()

            vertices = np.array([points["x"], points["y"], points["z"]])
            elements = np.array(elements).T

            # Test bem++ calculation for single vane
            try:

                import bempp.api

            except ImportError as e:

                print("Couldn't find module bempp. This only works in Linux environments!")
                print("Error was: {}".format(e))
                exit(1)

            self._grid = bempp.api.grid.grid_from_element_data(vertices, elements)

            if self._debug:

                self._grid.plot()

        else:
            print("The vane type '{}' is not (yet) implemented. Aborting.".format(self._vane_type))
            return 1

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


class FieldGenerator(object):
    def __init__(self, resolution=0.001, xy_limits=None):
        self._colors = MyColors()

        self._filename = None
        self._parameters = []
        self._nrms = None
        self._max_cells = 0
        self._plot_tcs_vanes = False

        self._voltage = 50.0e3  # (V)
        self._frequency = 32.8e6  # (Hz)
        self._a_init = 0.038802  # (m)
        self._resolution = resolution
        self._bempp_mesh_size = resolution
        self._xy_limits = xy_limits
        self._total_length = 0.0  # (m)

        self._nx = int((self._xy_limits[1] - self._xy_limits[0]) / self._resolution) + 1
        self._ny = int((self._xy_limits[3] - self._xy_limits[2]) / self._resolution) + 1

        self._mesh_x = None
        self._mesh_y = None
        self._mesh_z = None

        self._z_linear = None

        self._pot = None

        self._ex = None
        self._ey = None
        self._ez = None

        self._ex2 = None
        self._ey2 = None
        self._ez2 = None

        self._calculate_vane_profile = False
        self._vane_profile_x = None
        self._vane_profile_y = None


        self._vanes = []

        self._cell_dtype = [("cell type", '|U16'),
                            ("flip_z", bool),
                            ("shift_cell_no", bool),
                            ("cell no", int),
                            ("energy", float),
                            ("phase", float),
                            ("aperture", float),
                            ("modulation", float),
                            ("focusing factor", float),
                            ("cell length", float),
                            ("cumulative length", float)]

    def add_cell(self,
                 cell_type,
                 aperture,
                 modulation,
                 length,
                 flip_z=False,
                 shift_cell_no=False):

        assert cell_type in ["STA", "RMS", "NCS", "TCS", "DCS"], "cell_type must be one of RMS, NCS, TCS, DCS!"

        if (len(self._parameters) == 0):
            cell_no = 1
            cumulative_length = length
        else:
            cell_no = self._parameters[-1]["cell no"] + 1
            cumulative_length = self._parameters[-1]["cumulative length"] + length

        data = tuple([cell_type, flip_z, shift_cell_no, cell_no, 0.0, 0.0,
                      aperture, modulation, 0.0, length, cumulative_length])

        print(data)
        self._parameters = np.append(self._parameters, [np.array(data, dtype=self._cell_dtype)], axis=0)

        return 0

    def load_parameters_from_file(self, filename=None):
        """
        Load the cell parameters.
        :param filename:
        :return:
        """

        if filename is not None:
            self._filename = filename

        if self._filename is None:
            # TODO: File Dialog!
            print("No filename specified for parameter file. Closing.")
            exit(1)

        with open(filename, "r") as infile:
            if "Parmteqm" in infile.readline():
                self.load_from_parmteq(filename)
            else:
                self.load_from_vecc(filename)

    def load_from_vecc(self, filename=None):  
        data = []

        # noinspection PyTypeChecker
        with open(self._filename, "r") as infile:

            for line in infile:
                # noinspection PyTypeChecker
                data.append(tuple(["NCS", False, False]) + tuple([float(item) for item in line.strip().split()]))

        self._parameters = np.array(data, dtype=self._cell_dtype)


        # test = [x * y for x, y in zip(self._parameters["aperture"], self._parameters['modulation'])]
        # test = [x + y for x, y in zip(test, self._parameters["aperture"])]
        # test = [x / 2 for x in test]
        # print(np.mean(test))

        if self._nrms is None:
            self._nrms = len(np.where(self._parameters["modulation"] == 1.0)[0])
            print("I found {} cells with modulation 1 in the file..."
                  "assuming this is the entrance Radial Matching Section (RMS)."
                  " If this is incorrect, plase specify manually.".format(self._nrms))

    def load_from_parmteq(self, filename=None):
            
        with open(filename, "r") as infile:
            data = []
            version = infile.readline().strip().split()[1].split(",")[0]

            for line in infile:
                if "Cell" in line and "V" in line:
                    break

            for line in infile:
                if "Cell" in line and "V" in line:
                    break

                items = line.strip().split()
                cell_no = items[0]
                params = [float(item) for item in items[1:]]

                if len(items) == 10 and cell_no == "0":
                    if len(self._parameters) == 0:
                        self._parameters.append(np.array(("STA", False, False, 1, 0.0, 0.0, 
                                                     params[6] * 0.01,
                                                     params[7], 0.0, 0.0, 0.0), dtype=self._cell_dtype))

                    
                    continue

                if "T" in cell_no or "M" in cell_no or "F" in cell_no:
                    print("Ignored cell {}".format(cell_no))
                    continue


                if params[7] == 1.0:
                    cell_type = "RMS"

                else:
                    cell_type = "NCS"

                print("Cell no:  {}   Cell type:   {}".format(cell_no, cell_type))

                self.add_cell(cell_type=cell_type,
                              aperture=params[6] * 0.01,
                              modulation=params[7],
                              length=params[9] * 0.01,
                              flip_z=False,
                              shift_cell_no=False)

        if self._nrms is None:
            self._nrms = len(np.where(self._parameters["modulation"] == 1.0)[0])
            

        return 0


    def save_field_to_file(self, filename):

        with open(filename, 'w') as outfile:
            outfile.write("mesh_x, mesh_y, mesh_z, ex, ey, ez\n")
            for x, y, z, ex, ey, ez in zip(self._mesh_x.flatten(), self._mesh_y.flatten(), self._mesh_z.flatten(),
                                           self._ex.flatten(), self._ey.flatten(), self._ez.flatten()):
                outfile.write("{:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e}\n".format(x, y, z, ex, ey, ez))

    def calculate_pot_rms(self, idx, rms_a):

        x = self._mesh_x[idx]
        y = self._mesh_y[idx]
        z = self._mesh_z[idx]

        self._pot[idx] = 0.5 * self._voltage * (x ** 2.0 - y ** 2.0) / rms_a(z) ** 2.0

    def calculate_pot_drift(self, idx, aperture):
        x = self._mesh_x[idx]
        y = self._mesh_y[idx]
        z = self._mesh_z[idx]

        self._pot[idx] = 0.5 * self._voltage * (x ** 2.0 - y ** 2.0) / aperture ** 2.0

        if self._calculate_vane_profile:
            idx = np.where((np.min(z) <= self._z_linear) & (self._z_linear <= np.max(z)))

            self._vane_profile_x[idx] = aperture
            self._vane_profile_y[idx] = aperture

        return 0

    def calculate_profiles_2term(self, idx, cell_number):

        cell_parameters = self._parameters[cell_number]
        cell_start = cell_parameters["cumulative length"] - cell_parameters["cell length"]
        z = self._mesh_z[idx] - cell_start  # z has to be adjusted such that the cell starts at 0.0

        m = cell_parameters["modulation"]
        a = cell_parameters["aperture"]

        print("parameters len: {}".format(len(self._parameters)))

        if 0 < cell_number:
            ma_fudge_begin = 0.5 * (1.0 + self._parameters["aperture"][cell_number - 1] *
                                    self._parameters["modulation"][cell_number - 1] / m / a)
            a_fudge_begin = 0.5 * (1.0 + self._parameters["aperture"][cell_number - 1] / a)
        else:
            a_fudge_begin = ma_fudge_begin = 1.0

        if (cell_number + 1) == len(self._parameters):
            a_fudge_end = ma_fudge_end = 1.0
        elif cell_number < len(self._parameters):
            ma_fudge_end = 0.5 * (
                1.0 + self._parameters["aperture"][cell_number + 1] * self._parameters["modulation"][
                    cell_number + 1] / m / a)
            a_fudge_end = 0.5 * (1.0 + self._parameters["aperture"][cell_number + 1] / a)
        else:
            a_fudge_end = ma_fudge_end = 1.0

        a_fudge = interp1d([0.0, cell_parameters["cell length"]], [a_fudge_begin, a_fudge_end])
        ma_fudge = interp1d([0.0, cell_parameters["cell length"]], [ma_fudge_begin, ma_fudge_end])

        kp = np.pi / cell_parameters["cell length"]
        # mp = cell_parameters["modulation"]
        # ap = cell_parameters["aperture"]

        # denom = mp ** 2.0 * bessel1(0, kp * ap) + bessel1(0, mp * kp * ap)
        # a10 = (mp ** 2.0 - 1.0) / denom
        # r0 = ap / np.sqrt(1.0 - (mp ** 2.0 * bessel1(0, kp * ap) - bessel1(0, kp * ap)) / denom)

        sign = (-1.0) ** (cell_parameters["cell no"] + 1)

        idx2 = np.where((np.min(z) + cell_start <= self._z_linear) & (self._z_linear <= np.max(z) + cell_start))

        def ap(zz):
            return a * a_fudge(zz)

        def mp(zz):
            return m * ma_fudge(zz) / a_fudge(zz)

        def a10(zz):
            return (mp(zz) ** 2.0 - 1.0) / (mp(zz) ** 2.0 * bessel1(0, kp * ap(zz)) + bessel1(0, mp(zz) * kp * ap(zz)))

        def r0(zz):
            return ap(zz) / np.sqrt(1.0 - (mp(zz) ** 2.0 * bessel1(0, kp * ap(zz)) - bessel1(0, kp * ap(zz))) /
                                    (mp(zz) ** 2.0 * bessel1(0, kp * ap(zz)) + bessel1(0, mp(zz) * kp * ap(zz))))

        def vane_x(xx):
            return + sign * (xx / r0(_z)) ** 2.0 + a10(_z) * bessel1(0.0, kp * xx) * np.cos(kp * _z) - sign

        def vane_y(yy):
            return - sign * (yy / r0(_z)) ** 2.0 + a10(_z) * bessel1(0.0, kp * yy) * np.cos(kp * _z) + sign

        _vane_x = []
        _vane_y = []

        for _z in self._z_linear[idx2]:
            _z -= cell_start
            # noinspection PyTypeChecker
            _vane_x.append(root(vane_x, ap(_z)).x[0])
            # noinspection PyTypeChecker
            _vane_y.append(root(vane_y, ap(_z)).x[0])

        self._vane_profile_x[idx2] = _vane_x
        self._vane_profile_y[idx2] = _vane_y

        x = self._mesh_x[idx]
        y = self._mesh_y[idx]
        r = np.sqrt(x ** 2.0 + y ** 2.0)

        self._pot[idx] = 0.5 * self._voltage * ((x ** 2.0 - y ** 2.0) / r0(z) ** 2.0
                                                + sign * a10(z) * bessel1(0, kp * r) * np.cos(kp * z))

        return 0

    def calculate_pot_2term(self, idx, cell_parameters):

        x = self._mesh_x[idx]
        y = self._mesh_y[idx]
        cell_start = np.min(self._mesh_z[idx])
        z = self._mesh_z[idx] - cell_start  # z has to go from 0 to cell_length

        kp = np.pi / cell_parameters["cell length"]
        mp = cell_parameters["modulation"]
        ap = cell_parameters["aperture"]

        denom = mp ** 2.0 * bessel1(0, kp * ap) + bessel1(0, mp * kp * ap)
        a10 = (mp ** 2.0 - 1.0) / denom
        r0 = ap / np.sqrt(1.0 - (mp ** 2.0 * bessel1(0, kp * ap) - bessel1(0, kp * ap)) / denom)
        r = np.sqrt(x ** 2.0 + y ** 2.0)

        sign = (-1.0) ** (cell_parameters["cell no"] + 1)

        self._pot[idx] = 0.5 * self._voltage * ((x ** 2.0 - y ** 2.0) / r0 ** 2.0
                                                + sign * a10 * bessel1(0, kp * r) * np.cos(kp * z))

        return 0

    def calculate_e_2term(self, idx, cell_parameters):
        x = self._mesh_x[idx]
        y = self._mesh_y[idx]
        z = self._mesh_z[idx] - np.min(self._mesh_z[idx])  # z has to go from 0 to cell_length

        kp = np.pi / cell_parameters["cell length"]
        mp = cell_parameters["modulation"]
        ap = cell_parameters["aperture"]

        denom = mp ** 2.0 * bessel1(0, kp * ap) + bessel1(0, mp * kp * ap)
        a10 = (mp ** 2.0 - 1.0) / denom
        xim = 1.0 - a10 * bessel1(0, kp * ap)
        sign = (-1.0) ** (cell_parameters["cell no"])

        r = np.sqrt(x ** 2.0 + y ** 2.0)

        self._ex2[idx] = x * 0.5 * self._voltage * (+ 2.0 * xim / ap ** 2.0
                                                    - sign * kp / r * a10 * bessel1(1.0, kp * r) * np.cos(kp * z))

        self._ey2[idx] = y * 0.5 * self._voltage * (- 2.0 * xim / ap ** 2.0
                                                    - sign * kp / r * a10 * bessel1(1.0, kp * r) * np.cos(kp * z))

        self._ez2[idx] = 0.5 * self._voltage * sign * kp * a10 * bessel1(0.0, kp * r) * np.sin(kp * z)

    @staticmethod
    def calculate_transition_cell_length(cell_parameters):

        k = np.pi / np.sqrt(3.0) / cell_parameters["cell length"]
        m = cell_parameters["modulation"]
        a = cell_parameters["aperture"]
        r0 = 0.5 * (a + m * a)

        def eta(kk):
            return bessel1(0.0, kk * r0) / (3.0 * bessel1(0.0, 3.0 * kk * r0))

        def func(kk):
            return (bessel1(0.0, kk * m * a) + eta(kk) * bessel1(0.0, 3.0 * kk * m * a)) / \
                   (bessel1(0.0, kk * a) + eta(kk) * bessel1(0.0, 3.0 * kk * a)) \
                   + ((m * a / r0) ** 2.0 - 1.0) / ((a / r0) ** 2.0 - 1.0)

        k = root(func, k).x[0]
        tcs_length = np.pi / 2.0 / k
        print("Transition cell has length {} which is {} * cell length, ".format(tcs_length,
                                                                                 tcs_length / cell_parameters[
                                                                                     "cell length"]), end="")
        print("the remainder will be filled with a drift.")

        assert tcs_length <= cell_parameters["cell length"], "Numerical determination of transition cell length " \
                                                             "yielded value larger than cell length parameter!"

        return np.pi / 2.0 / k

    def calculate_pot_3term(self, idx, cell_parameters):

        x = self._mesh_x[idx]
        y = self._mesh_y[idx]
        cell_start = np.min(self._mesh_z[idx])
        z = self._mesh_z[idx] - cell_start  # z has to go from 0 to cell_length

        k = np.pi / np.sqrt(3.0) / cell_parameters["cell length"]
        m = cell_parameters["modulation"]
        a = cell_parameters["aperture"]
        r0 = 0.5 * (a + m * a)

        print("Average radius of transition cell (a + ma) / 2 = {}".format(r0))

        def eta(kk):
            return bessel1(0.0, kk * r0) / (3.0 * bessel1(0.0, 3.0 * kk * r0))

        def a10(kk):
            return ((m * a / r0) ** 2.0 - 1.0) / (bessel1(0.0, kk * m * a) + eta(kk) * bessel1(0.0, 3.0 * kk * m * a))

        def a30(kk):
            return eta(kk) * a10(kk)

        def func(kk):
            return (bessel1(0.0, kk * m * a) + eta(kk) * bessel1(0.0, 3.0 * kk * m * a)) / \
                   (bessel1(0.0, kk * a) + eta(kk) * bessel1(0.0, 3.0 * kk * a)) \
                   + ((m * a / r0) ** 2.0 - 1.0) / ((a / r0) ** 2.0 - 1.0)

        k = root(func, k).x[0]

        if cell_parameters["shift_cell_no"]:
            sign = (-1.0) ** (cell_parameters["cell no"] + 1)
        else:
            sign = (-1.0) ** (cell_parameters["cell no"])

        r = np.sqrt(x ** 2.0 + y ** 2.0)

        # print(np.cos(k * np.max(z)))
        # print(np.cos(3.0 * k * np.max(z)))

        self._pot[idx] = np.round(0.5 * self._voltage * (
            + (x ** 2.0 - y ** 2.0) / r0 ** 2.0
            - sign * a10(k) * bessel1(0.0, k * r) * np.cos(k * z)
            - sign * a30(k) * bessel1(0.0, 3.0 * k * r) * np.cos(3.0 * k * z)
        ), 5)

        if self._plot_tcs_vanes or self._calculate_vane_profile:

            idx = np.where((np.min(z) + cell_start <= self._z_linear) & (self._z_linear <= np.max(z) + cell_start))

            def vane_x(xx):
                return - (xx / r0) ** 2.0 \
                       + sign * a10(k) * bessel1(0.0, k * xx) * np.cos(k * _z) \
                       + sign * a30(k) * bessel1(0.0, 3.0 * k * xx) * np.cos(3.0 * k * _z) + 1.0

            def vane_y(yy):
                return + (yy / r0) ** 2.0 \
                       + sign * a10(k) * bessel1(0.0, k * yy) * np.cos(k * _z) \
                       + sign * a30(k) * bessel1(0.0, 3.0 * k * yy) * np.cos(3.0 * k * _z) - 1.0

            def pot_on_axis():
                return 0.5 * self._voltage * (
                    + a10(k) * (np.cos(k * _z))
                    + a30(k) * (np.cos(3.0 * k * _z)))

            _vane_x = []
            _vane_y = []
            _z_plot = []
            _pot_plot = []

            for _z in self._z_linear[idx]:
                _z -= cell_start
                # noinspection PyTypeChecker
                _vane_x.append(root(vane_x, r0).x[0])
                # noinspection PyTypeChecker
                _vane_y.append(root(vane_y, r0).x[0])
                _z_plot.append(_z)
                _pot_plot.append(pot_on_axis())

            if self._plot_tcs_vanes:
                plt.plot(_z_plot, _vane_x, color=self._colors[0])
                plt.plot(_z_plot, _vane_y, color=self._colors[2])
                plt.scatter([0.0, 0.0, np.max(np.unique(z))], [a, m * a, r0], s=16, color=self._colors[1])

                plt.twinx()

                plt.plot(_z_plot, _pot_plot, color=self._colors[3])

                plt.show()

            if self._calculate_vane_profile:
                if cell_parameters["flip_z"]:
                    self._vane_profile_x[idx] = _vane_x[::-1]
                    self._vane_profile_y[idx] = _vane_y[::-1]
                else:
                    self._vane_profile_x[idx] = _vane_x
                    self._vane_profile_y[idx] = _vane_y

        return 0

    @staticmethod
    def pot_8term(coeff, _r, _t, _z, k):

        a01, a03, a10, a12, a21, a23, a30, a32 = coeff

        return + a01 * _r ** 2.0 * np.cos(2.0 * _t) \
               + a03 * _r ** 6.0 * np.cos(6.0 * _t) \
               + a10 * bessel1(0, k * _r) * np.cos(k * _z) \
               + a12 * bessel1(4, k * _r) * np.cos(4.0 * _t) * np.cos(k * _z) \
               + a21 * bessel1(2, 2.0 * k * _r) * np.cos(2.0 * _t) * np.cos(2.0 * k * _z) \
               + a23 * bessel1(6, 2.0 * k * _r) * np.cos(6.0 * _t) * np.cos(2.0 * k * _z) \
               + a30 * bessel1(0, 3.0 * k * _r) * np.cos(3.0 * k * _z) \
               + a32 * bessel1(4, 3.0 * k * _r) * np.cos(4.0 * _t) * np.cos(3.0 * k * _z)

    def calculate_pot_8term(self, idx, cell_parameters):

        # --- Calculate the eight coefficients using a least squares fit: --- #
        # First, generate a matrix of surface points
        # Assume that cell starts with x = a and y = ma at z = 0
        # lets start with 5 mm radius half-circle at z=0
        samples = 1000

        vane_radius = 0.005  # (m)

        vane_opening_angle = 45.0  # (deg)
        vane_opening_angle *= np.pi / 180.0

        xvane_angles = np.linspace(np.pi - vane_opening_angle,
                                   np.pi + vane_opening_angle,
                                   samples)
        yvane_angles = np.linspace(1.5 * np.pi - vane_opening_angle,
                                   1.5 * np.pi + vane_opening_angle,
                                   samples)

        xvane_x = vane_radius * np.cos(xvane_angles)
        xvane_y = vane_radius * np.sin(xvane_angles)
        yvane_x = vane_radius * np.cos(yvane_angles)
        yvane_y = vane_radius * np.sin(yvane_angles)

        if cell_parameters["cell no"] % 2.0 == 0.0:

            ap1 = cell_parameters["aperture"] * cell_parameters["modulation"]
            ap2 = 0.5 * (ap1 + cell_parameters["aperture"])
            ap3 = cell_parameters["aperture"]

        else:

            ap1 = cell_parameters["aperture"]
            ap2 = 0.5 * (cell_parameters["modulation"] * ap1 + ap1)
            ap3 = cell_parameters["modulation"] * ap1

        r1 = np.sqrt((xvane_x + ap1 + vane_radius) ** 2.0 + xvane_y ** 2.0)
        r2 = np.sqrt((xvane_x + ap2 + vane_radius) ** 2.0 + xvane_y ** 2.0)
        r3 = np.sqrt((xvane_x + ap3 + vane_radius) ** 2.0 + xvane_y ** 2.0)

        r4 = np.sqrt(yvane_x ** 2.0 + (yvane_y + ap3 + vane_radius) ** 2.0)
        r5 = np.sqrt(yvane_x ** 2.0 + (yvane_y + ap2 + vane_radius) ** 2.0)
        r6 = np.sqrt(yvane_x ** 2.0 + (yvane_y + ap1 + vane_radius) ** 2.0)

        z1 = np.zeros(samples)
        z2 = np.ones(samples) * 0.5 * cell_parameters["cell length"]
        z3 = np.ones(samples) * cell_parameters["cell length"]

        t1 = np.arctan2(xvane_y, xvane_x + ap1 + vane_radius)
        t2 = np.arctan2(xvane_y, xvane_x + ap2 + vane_radius)
        t3 = np.arctan2(xvane_y, xvane_x + ap3 + vane_radius)

        t4 = np.arctan2(yvane_y + ap3 + vane_radius, yvane_x)
        t5 = np.arctan2(yvane_y + ap2 + vane_radius, yvane_x)
        t6 = np.arctan2(yvane_y + ap1 + vane_radius, yvane_x)

        # plt.plot(r1 * np.cos(t1), r1 * np.sin(t1), linestyle='dashed', color='red', label="x1")
        # plt.plot(r2 * np.cos(t2), r2 * np.sin(t2), linestyle='dotted', color='red', label="x2")
        # plt.plot(r3 * np.cos(t3), r3 * np.sin(t3), color='red', label="x3")
        # plt.plot(r4 * np.cos(t4), r4 * np.sin(t4), linestyle='dashed', color='blue', label="y1")
        # plt.plot(r5 * np.cos(t5), r5 * np.sin(t5), linestyle='dotted', color='blue', label="y2")
        # plt.plot(r6 * np.cos(t6), r6 * np.sin(t6), color='blue', label="y3")
        # plt.legend()
        # plt.show()
        # exit()

        # Finalize arrays containing the surface points in cylinder coordinates
        r = np.array([r1, r2, r3, r4, r5, r6]).flatten()
        t = np.array([t1, t2, t3, t4, t5, t6]).flatten()
        z = np.array([z1, z2, z3, z1, z2, z3]).flatten()

        # Generate an array of "results" (along the surface, all potentials are V/2)
        y1 = np.ones(3 * samples) * +0.5 * self._voltage
        y2 = np.ones(3 * samples) * -0.5 * self._voltage

        y = np.array([y1, y2]).flatten()

        k = np.pi / cell_parameters["cell length"]

        matrix = []
        for _r, _t, _z in zip(r, t, z):
            matrix.append([_r ** 2.0 * np.cos(2.0 * _t),
                           _r ** 6.0 * np.cos(6.0 * _t),
                           bessel1(0, k * _r) * np.cos(k * _z),
                           bessel1(4, k * _r) * np.cos(4.0 * _t) * np.cos(k * _z),
                           bessel1(2, 2.0 * k * _r) * np.cos(2.0 * _t) * np.cos(2.0 * k * _z),
                           bessel1(6, 2.0 * k * _r) * np.cos(6.0 * _t) * np.cos(2.0 * k * _z),
                           bessel1(0, 3.0 * k * _r) * np.cos(3.0 * k * _z),
                           bessel1(4, 3.0 * k * _r) * np.cos(4.0 * _t) * np.cos(3.0 * k * _z),
                           ])

        matrix = np.array(matrix)

        # print(matrix)

        # Call the least squares function and get the 8 coefficients
        coeff = np.linalg.lstsq(matrix, y)[0]

        x = self._mesh_x[idx]
        y = self._mesh_y[idx]
        z = self._mesh_z[idx]

        z_min = np.min(z)
        # z_half = 0.5 * (z_min + np.max(z))

        z -= z_min

        self._pot[idx] = self.pot_8term(coeff, np.sqrt(x ** 2.0 + y ** 2.0), np.arctan2(y, x), z, k)

        return 0

    def calculate_e_from_pot(self, idx, flip_z, ttf=0.0, phase=0.0):

        if flip_z:
            pot = self._pot[:, :, idx[2][-1]:idx[2][0] - 1:-1]
        else:
            pot = self._pot[:, :, idx[2][0]:idx[2][-1] + 1]

        tt_factor = -np.cos((2.0 * np.pi * self._frequency * ttf) + (phase * np.pi / 180.0))

        a, b, c = np.gradient(pot, self._resolution, edge_order=2)

        self._ex[idx] = tt_factor * a.flatten()
        self._ey[idx] = tt_factor * b.flatten()
        self._ez[idx] = tt_factor * c.flatten()

        return 0

    def calculate_ex_rms(self, idx, rms_a, transit_time=0.0, phase=0.0):

        x = self._mesh_x[idx]
        z = self._mesh_z[idx]

        self._ex[idx] = -(self._voltage * x / rms_a(z) ** 2.0) * (
            np.cos((2.0 * np.pi * self._frequency * transit_time) + (phase * np.pi / 180.0)))

        return 0

    def calculate_ey_rms(self, idx, rms_a, transit_time=0.0, phase=0.0):

        y = self._mesh_y[idx]
        z = self._mesh_z[idx]

        self._ey[idx] = (self._voltage * y / rms_a(z) ** 2.0) * (
            np.cos((2.0 * np.pi * self._frequency * transit_time) + (phase * np.pi / 180.0)))

        return 0

    def calculate_dcs(self, cell_parameters):

        # Find all z values that are within the cell
        cell_idx = np.where((self._mesh_z <= cell_parameters["cumulative length"]) &
                            (self._mesh_z >= (cell_parameters["cumulative length"] -
                                              cell_parameters["cell length"])))

        print("Calculating Drift Cell # {}, ".format(cell_parameters["cell no"]), end="")
        print("from z = {} m to {} m".format(cell_parameters["cumulative length"] -
                                             cell_parameters["cell length"],
                                             cell_parameters["cumulative length"]))

        self.calculate_pot_drift(cell_idx, cell_parameters["aperture"])
        self.calculate_e_from_pot(cell_idx, cell_parameters["flip_z"])

        return 0

    def calculate_ncs(self, cell_parameters, cell_number=None):
        print("Calcncs cell num: {}   passed:  {}".format(cell_parameters["cell no"], cell_number))
        # Find all z values that are within the cell
        cell_idx = np.where((self._mesh_z <= cell_parameters["cumulative length"]) &
                            (self._mesh_z >= (cell_parameters["cumulative length"] -
                                              cell_parameters["cell length"])))

        print("Calculating Normal Cell # {}, ".format(cell_parameters["cell no"]), end="")
        print("from z = {} m to {} m".format(cell_parameters["cumulative length"] -
                                             cell_parameters["cell length"],
                                             cell_parameters["cumulative length"]))

        self.calculate_profiles_2term(cell_idx, cell_number)
        # self.calculate_pot_2term(cell_idx, cell_parameters)
        # self.calculate_pot_8term(cell_idx, cell_parameters)
        self.calculate_e_from_pot(cell_idx, cell_parameters["flip_z"])
        # self.calculate_e_2term(cell_idx, cell_parameters)

    def calculate_rms_in(self):

        # b_match = self._parameters[self._nrms + 1]["focusing factor"]
        rms_length = self._parameters[self._nrms - 1]["cumulative length"]

        print(rms_length)

        rms_idx = np.where(self._mesh_z <= rms_length)

        print("Calculating Radial Matching Section from z = {} m to {} m".format(0.0, rms_length))

        # The RMS has a smooth decrease in aperture size from a_init to the size of the first RFQ cell
        # Also, the first datapoint (at z = 0.0) is not included in the file, so we add it here.
        rms_a = interp1d(np.append(np.array([0.0]), self._parameters["cumulative length"][:self._nrms]),
                         np.append(np.array([self._a_init]), self._parameters["aperture"][:self._nrms]),
                         kind="cubic")


        self.calculate_pot_rms(rms_idx, rms_a)
        # self.calculate_e_from_pot(rms_idx)
        self.calculate_ex_rms(rms_idx, rms_a, 0.0, 0.0)
        self.calculate_ey_rms(rms_idx, rms_a, 0.0, 0.0)

        if self._calculate_vane_profile:
            idx = np.where(self._z_linear <= rms_length)

            self._vane_profile_x[idx] = rms_a(self._z_linear[idx])
            self._vane_profile_y[idx] = rms_a(self._z_linear[idx])

    def calculate_tcs(self, cell_parameters):
        print("Calculating Transition Cell # {}, ".format(cell_parameters["cell no"]), end="")
        print("from z = {} m to {} m".format(cell_parameters["cumulative length"] -
                                             cell_parameters["cell length"],
                                             cell_parameters["cumulative length"]))

        tcs_length = self.calculate_transition_cell_length(cell_parameters)
        tcs_begin = cell_parameters["cumulative length"] - cell_parameters["cell length"]

        if cell_parameters["flip_z"]:

            cell_idx = np.where((self._mesh_z >= tcs_begin) &
                                (self._mesh_z <= cell_parameters["cumulative length"] - tcs_length))

            r0 = 0.5 * (cell_parameters["aperture"] + cell_parameters["modulation"] * cell_parameters["aperture"])

            self.calculate_pot_drift(cell_idx, r0)

            cell_idx = np.where((self._mesh_z >= cell_parameters["cumulative length"] - tcs_length) &
                                (self._mesh_z <= cell_parameters["cumulative length"]))

            self.calculate_pot_3term(cell_idx, cell_parameters)

        else:

            cell_idx = np.where((self._mesh_z >= tcs_begin) &
                                (self._mesh_z <= (tcs_begin + tcs_length)))

            self.calculate_pot_3term(cell_idx, cell_parameters)

            cell_idx = np.where((self._mesh_z >= (tcs_begin + tcs_length)) &
                                (self._mesh_z <= cell_parameters["cumulative length"]))

            r0 = 0.5 * (cell_parameters["aperture"] + cell_parameters["modulation"] * cell_parameters["aperture"])

            self.calculate_pot_drift(cell_idx, r0)

        cell_idx = np.where((self._mesh_z >= (cell_parameters["cumulative length"] -
                                              cell_parameters["cell length"])) &
                            (self._mesh_z <= cell_parameters["cumulative length"]))

        self.calculate_e_from_pot(cell_idx, cell_parameters["flip_z"])

    def generate(self):

        # Find total number of cells
        self._max_cells = self._parameters[-1]["cell no"]

        # Generating the mesh points for the full length RFQ
        self.generate_mesh()

        # the RFQ has several sections that are calculated differently:
        # *) Radial matching sections (RMS)
        # *) Normal cell section for gentle buncher + accelerating section (NCS)
        # *) Transition cell sections (TCS)
        # *) Drift cell sections (DCS)

        # Omit RMS if there are no cells with modulation 1
        if self._nrms > 0:
            print("within the self nrms!")
            #self.calculate_rms_in()
        # Loop over all remaining cells starting after RMS
        for cn, cell_parameters in enumerate(self._parameters[self._nrms:]):

            cn += self._nrms
            # Calculate the potential and field according to cell type
            if cell_parameters["cell type"] == "NCS":
                self.calculate_ncs(cell_parameters, cell_number=cn)

            elif cell_parameters["cell type"] == "TCS":
                self.calculate_tcs(cell_parameters)

            elif cell_parameters["cell type"] == "DCS":
                self.calculate_dcs(cell_parameters)

        # self.calculate_rms_out()

        # Do the whole thing above and now genrate a mesh for bempp
        # TODO: use multiprocessing for the above stuff
        # TODO: Clean up unnecessary calculations
        # TODO: use c-wrappers for some of the above to speed up?
        # TODO: BEM++ calculation using MPI? Only on Linux systems...
        # TODO: Calculate only a quarter of the RFQ. Make use of SYMMETRY!
        # self.mesh_vanes()

        return 0

    def generate_mesh(self):

        x_values = np.linspace(self._xy_limits[0], self._xy_limits[1], self._nx)
        y_values = np.linspace(self._xy_limits[2], self._xy_limits[3], self._ny)

        total_z_length = self._parameters["cumulative length"][-1] + self._resolution

        z_values = np.arange(0.0, total_z_length, self._resolution)

        if z_values[-1] > self._parameters["cumulative length"][-1]:
            z_values = z_values[:-1]

        self._total_length = np.max(z_values)

        self._mesh_x, self._mesh_y, self._mesh_z = meshgrid(x_values, y_values, z_values, indexing='ij')

        self._z_linear = np.sort(np.unique(self._mesh_z))

        self._pot = np.zeros(self._mesh_x.shape)

        self._ex = np.zeros(self._mesh_x.shape)
        self._ey = np.zeros(self._mesh_x.shape)
        self._ez = np.zeros(self._mesh_x.shape)

        if self._calculate_vane_profile:
            self._vane_profile_x = np.zeros(self._z_linear.shape)
            self._vane_profile_y = np.zeros(self._z_linear.shape)

        # self._ex2 = np.zeros(self._mesh_x.shape)
        # self._ey2 = np.zeros(self._mesh_x.shape)
        # self._ez2 = np.zeros(self._mesh_x.shape)

        return 0

    def plot_e_xy(self, z=0.0):

        plot_idx = np.where(self._mesh_z == z)

        plt.quiver(self._mesh_x[plot_idx].flatten(),
                   self._mesh_y[plot_idx].flatten(),
                   self._ex[plot_idx].flatten(),
                   self._ey[plot_idx].flatten())

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        plt.show()

    def plot_pot_xy(self, z=0.0):

        # Find nearest mesh value to desired z
        zvals = self._mesh_z.flatten()
        z_close = zvals[np.abs(zvals - z).argmin()]

        z_slice_idx = np.where(self._mesh_z[0, 0, :] == z_close)[0][0]

        plt.contour(self._mesh_x[:, :, z_slice_idx],
                    self._mesh_y[:, :, z_slice_idx],
                    self._pot[:, :, z_slice_idx],
                    40,
                    cmap=plt.get_cmap("jet"))

        plt.colorbar()

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.gca().set_aspect('equal')

        plt.show()

    def plot_ex_of_z(self, x=0.0):

        # Find nearest mesh value to desired x
        xvals = self._mesh_x.flatten()
        x_close = xvals[np.abs(xvals - x).argmin()]

        print(x_close)

        plot_idx = np.where((self._mesh_x == x_close) & (self._mesh_y == 0.0))

        plt.plot(self._mesh_z[plot_idx], self._ex[plot_idx])

        plt.xlabel("z (m)")
        plt.ylabel("Ex (V/m)")
        plt.title("Ex(z) at x = {} m, y = 0 m".format(x_close))

        plt.show()

    def plot_ez_of_z(self):

        plot_idx = np.where((self._mesh_x == 0.0) & (self._mesh_y == 0.0))

        plt.plot(self._mesh_z[plot_idx], self._ez[plot_idx] / 100.0)

        plt.xlim(0.0, 0.12)

        plt.xlabel("z (m)")
        plt.ylabel("Ez (V/m)")

        plt.show()

    def plot_pot_of_z(self, opera_data_fn=None):

        plot_idx = np.where((self._mesh_x == 0.0) & (self._mesh_y == 0.0))

        plt.plot(self._mesh_z[plot_idx], self._pot[plot_idx])

        # plt.xlim(0.0, 0.12)

        plt.xlabel("z (m)")
        plt.ylabel("Potential (V)")
        plt.title("U(z) at x = 0 m, y = 0 m")

        if opera_data_fn is not None:
            opera_data = []

            # noinspection PyTypeChecker
            with open(opera_data_fn, "r") as infile:

                for k in range(5):
                    infile.readline()

                for line in infile:
                    # noinspection PyTypeChecker
                    opera_data.append(tuple([float(item) for item in line.strip().split()]))

            mydtype = [("z", float),
                       ("v", float),
                       ("ez", float)]

            opera_data = np.array(opera_data, dtype=mydtype)

            plt.plot(0.01 * opera_data["z"] + 1.2070849776907893, -opera_data["v"])

        plt.show()

    def set_bempp_mesh_size(self, mesh_size):
        print("voltage {}".format(self._voltage))
        self._bempp_mesh_size = mesh_size

    def set_calculate_vane_profile(self, calculate_vane_profile=True):
        self._calculate_vane_profile = calculate_vane_profile

    def set_plot_tcs_vanes(self, plot_tcs_vanes=True):
        self._plot_tcs_vanes = plot_tcs_vanes

    def plot_combo(self, opera_data_fn=None):

        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        # plt.rc('font', size=18)

        # Find nearest mesh value to desired x
        x_plot = 0.05
        xvals = self._mesh_x.flatten()
        x_close = xvals[np.abs(xvals - x_plot).argmin()]

        plot_idx = np.where((self._mesh_x == x_close) & (self._mesh_y == 0.0))
        plt.plot(self._mesh_z[plot_idx], self._ex[plot_idx] / 10, color=self._colors[2], label="$\mathrm{E}_\mathrm{x}$")
        # plt.plot(self._mesh_z[plot_idx], self._ex2[plot_idx], linestyle='dashed', color='red')

        # Find nearest mesh value to desired y
        y_plot = 0.05
        yvals = self._mesh_y.flatten()
        y_close = yvals[np.abs(yvals - y_plot).argmin()]

        plot_idx = np.where((self._mesh_y == y_close) & (self._mesh_x == 0.0))
        plt.plot(self._mesh_z[plot_idx], self._ey[plot_idx] / 10, color=self._colors[1], label="$\mathrm{E}_\mathrm{y}$")
        # plt.plot(self._mesh_z[plot_idx], self._ey2[plot_idx], linestyle='dashed', color='green')

        plot_idx = np.where((self._mesh_x == 0.0) & (self._mesh_y == 0.0))
        plt.plot(self._mesh_z[plot_idx], self._ez[plot_idx], color=self._colors[0], label="$\mathrm{E}_\mathrm{z}$")
        # plt.plot(self._mesh_z[plot_idx], self._ez2[plot_idx], linestyle='dashed', color='blue')

        plt.xlabel("Z (m)")
        plt.ylabel("Electric Field (V/m)")
        plt.xlim(0.0, None)
        plt.ylim(-800000.0, 800000.0)

        if opera_data_fn is not None:
            opera_data = []

            # noinspection PyTypeChecker
            with open(opera_data_fn, "r") as infile:

                for k in range(5):
                    infile.readline()

                for line in infile:
                    # noinspection PyTypeChecker
                    opera_data.append(tuple([float(item) for item in line.strip().split()]))

            mydtype = [("z", float),
                       ("v", float),
                       ("ez", float)]

            opera_data = np.array(opera_data, dtype=mydtype)

            plt.plot(0.01 * opera_data["z"] + 1.2070849776907893, -opera_data["ez"] * 100.0, color=self._colors[3])

        plt.legend(loc=2)

        plt.show()

    def plot_vane_profile(self):
        print("==========\nPLOTTING VANE PROFILE\n==========")
        if not self._calculate_vane_profile:
            print("Vane profile was not calculated, use set_calculate_vane_profile() before generating.")

            return 1

        plt.plot(self._z_linear[:-1], self._vane_profile_x[:-1], color=self._colors[0])
        plt.plot(self._z_linear[:-1], -self._vane_profile_y[:-1], color=self._colors[1])

        interp_profile = interp1d(self._z_linear[:-1], self._vane_profile_x[:-1], kind='cubic')

        z_plot = np.linspace(self._z_linear[1], self._z_linear[-2], 5000)

        plt.plot(z_plot,
                 interp_profile(z_plot),
                 color=self._colors[2])

        plt.xlim(min(self._z_linear), max(self._z_linear))
        # plt.xlim(0.0, 0.12)
        plt.ylim(-1.1 * max(self._vane_profile_x), 1.1 * max(self._vane_profile_x))

        plt.show()

        return 0

    def mesh_vanes(self):

        # We have already generated the vane profile, now we have to generate a mesh
        # In BEMPP the mesh is generated from a set of vertices (given by arrays of z, y and z coordinates)
        # and a set of elements that define how the vertices are connected.

        # self._mesh_x = None
        # self._mesh_y = None
        # self._mesh_z = None
        self._pot = None
        self._ex = None
        self._ey = None
        self._ez = None

        gc.collect()

        stopwatch("start")

        # If the user has not defined a separate bempp mesh size, the resolution will be used.
        x_vane1 = Vane(vane_type="semi-circle",
                       mesh_size=self._bempp_mesh_size,
                       curvature=0.005,
                       height=0.001,
                       vane_z_data=self._z_linear,
                       vane_y_data=self._vane_profile_x,
                       rotation=90.0)

        x_vane2 = Vane(vane_type="semi-circle",
                       mesh_size=self._bempp_mesh_size,
                       curvature=0.005,
                       height=0.001,
                       vane_z_data=self._z_linear,
                       vane_y_data=self._vane_profile_x,
                       rotation=-90.0)

        y_vane1 = Vane(vane_type="semi-circle",
                       mesh_size=self._bempp_mesh_size,
                       curvature=0.005,
                       height=0.001,
                       vane_z_data=self._z_linear,
                       vane_y_data=self._vane_profile_y,
                       rotation=0.0)

        y_vane2 = Vane(vane_type="semi-circle",
                       mesh_size=self._bempp_mesh_size,
                       curvature=0.005,
                       height=0.001,
                       vane_z_data=self._z_linear,
                       vane_y_data=self._vane_profile_y,
                       rotation=180.0)

        x_vane1.generate_grid()
        x_vane2.generate_grid()
        y_vane1.generate_grid()
        y_vane2.generate_grid()

        complete_system = x_vane1 + x_vane2 + y_vane1 + y_vane2
        # complete_system.plot_grid()

        del x_vane1
        del x_vane2
        del y_vane1
        del y_vane2

        stopwatch('Vanes created')

        # import bempp.api
        #
        # space = bempp.api.function_space(complete_system.get_grid(), "DP", 0)
        # slp = bempp.api.operators.boundary.laplace.single_layer(space, space, space)
        #
        # def f(r, n, domain_index, result):
        #     if abs(r[0]) > 0.011:
        #         result[0] = 25000.0
        #     else:
        #         result[0] = -25000.0
        #
        # rhs = bempp.api.GridFunction(space, fun=f)
        # sol, _ = bempp.api.linalg.gmres(slp, rhs)
        #
        # stopwatch('BEM problem solved')
        #
        # # sol.plot()
        #
        # # nvals = 251
        # # z_vals = np.linspace(0.0, 0.250, nvals)
        # # nvals = len(self._z_linear)
        # # points = np.vstack([np.zeros(nvals), np.zeros(nvals), self._z_linear])
        # points = np.vstack([self._mesh_x.flatten(),
        #                     self._mesh_y.flatten(),
        #                     self._mesh_z.flatten()])
        #
        # nearfield = bempp.api.operators.potential.laplace.single_layer(space, points)
        # pot_discrete = nearfield * sol
        #
        # self._pot = pot_discrete[0].reshape(self._mesh_x.shape)
        #
        # stopwatch('potential calculated')
        #
        # self._ex, self._ey, self._ez = -np.gradient(self._pot, self._resolution, edge_order=2)
        #
        # stopwatch('field calculated')

        return 0



    def write_inventor_macro(self, save_folder=None):

        if save_folder is None:
            fd = FileDialog()
            save_folder, _ = fd.get_filename('folder')

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
                            Dim oSketch As Sketch3D
                            Dim oSpline As SketchSplines3D
                            Dim vertexCollection1 As ObjectCollection
                            Dim oLine As SketchLines3D
                            Dim number_of_points As Long
                            Dim loft_section_index As Long
                            Dim frequency As Integer: frequency = 10
                            Dim oLoftDef As LoftDefinition
                            Dim oLoftSections As ObjectCollection
                            Dim spiral_electrode As LoftFeature
                        """.format(direction)

            electrode_text = """
                                Set oPart = oApp.Documents.Add(kPartDocumentObject, , True)

                                Set oCompDef = oPart.ComponentDefinition

                            """
            electrode_text += """
                                Set oSketch = oCompDef.Sketches3D.Add
                                Set oSpline = oSketch.SketchSplines3D
                                Set vertexCollection1 = oApp.TransientObjects.CreateObjectCollection(Null)

                                FileName = "{}"
                                fileNo = FreeFile 'Get first free file number
                                
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
                                    
                                Loop
                                
                                Close #fileNo

                                Call oSpline.Add(vertexCollection1)

                                """.format(os.path.join(save_folder, "Transition_{}.txt".format(direction)))

            footer_text = """
                            oPart.UnitsOfMeasure.LengthUnits = kMillimeterLengthUnits

                            ThisApplication.ActiveView.Fit

                        End Sub
                        """

            with open(os.path.join(save_folder, "Transition_{}.ivb".format(direction)), "w") as outfile:

                outfile.write(header_text + electrode_text + footer_text)

            z_start = 0.10972618296477678

            with open(os.path.join(save_folder, "Transition_{}.txt".format(direction)), "w") as outfile:

                if direction == "X":

                    # for i in range(len(self._z_linear)):
                    for i in np.where(self._z_linear >= z_start)[0]:
                        outfile.write("{:.6f}, {:.6f}, {:.6f}\n".format(
                            self._vane_profile_x[i] * 100.0,  # For some weird reason Inventor uses cm as default...
                            0.0,
                            self._z_linear[i] * 100.0))

                else:

                    # for i in range(len(self._z_linear)):
                    for i in np.where(self._z_linear >= z_start)[0]:
                        outfile.write("{:.6f}, {:.6f}, {:.6f}\n".format(
                            0.0,
                            self._vane_profile_y[i] * 100.0,  # For some weird reason Inventor uses cm as default...
                            self._z_linear[i] * 100.0))

        return 0


if __name__ == "__main__":

    if platform.node() == "Mailuefterl":
        folder = r"D:\Daniel\Dropbox (MIT)\Projects" \
                 r"\RFQ Direct Injection\RFQ_Tests\VECC-Design\rfq_vecc_004"
    elif platform.node() == "TARDIS":
        folder = r"D:\Dropbox (MIT)\Projects" \
                 r"\RFQ Direct Injection\RFQ_Tests\VECC-Design\rfq_vecc_004"
    else:
        folder = r"C:\Users\Daniel Winklehner\Dropbox (MIT)\Projects" \
                 r"\RFQ Direct Injection\RFQ_Tests\VECC-Design\rfq_vecc_004"

    # loadpath = os.path.join(folder, r"Parm_50_63cells.dat")
    # loadpath = os.path.join(folder, r"Parm_50_last3.dat")
    #savepath = os.path.join(".", r"vecc_rfq_003_py.dat")
    #savepath = os.path.join(".", r"vecc_rfq_004_py.dat")
    #savepath = os.path.join('.', r"fieldw015width.dat")
    savepath = os.path.join('.', r"parmteqresults.dat")


    operapath = os.path.join(folder, "Transition_cell_pot_field.table")
    #loadpath = r"Parm_50_last3.dat"
    loadpath = r"input/PARMTEQOUT.TXT"
    #loadpath = r"input/Parm_50_63cells.dat"
    fg = FieldGenerator(resolution=0.002, xy_limits=[-0.015, 0.015, -0.015, 0.015])
    fg.set_bempp_mesh_size(0.002)

    fg.load_parameters_from_file(filename=loadpath)

    # fg.add_cell(cell_type="TCS",

    #             aperture=0.011255045027294745,
    #             modulation=1.6686390559337798,
    #             length=0.0427)
    # # 0.10972618296477678

    # fg.add_cell(cell_type="DCS",
    #             aperture=0.015017826368066015,
    #             modulation=1.0,
    #             length=0.13)

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
    # fg.plot_vane_profile()
    # fg.write_inventor_macro()

    # fg.plot_pot_of_z()
    fg.plot_combo()
    # fg.plot_e_xy(z=1.33)
    fg.plot_ex_of_z(x=0.05)
    # fg.plot_pot_xy(z=1.33)
    # fg.plot_ez_of_z()

    # fg.write_inventor_macro(save_folder=folder)


    fg.save_field_to_file(filename=savepath)
