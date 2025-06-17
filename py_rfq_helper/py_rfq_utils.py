# py_rfq_utils.py
# Written by Jared Hwang in August 2018
#
# Contains the PyRfqUtils class designed to work in tandem with the RFQ object from
# the py_rfq_module (py_rfq_designer), and a corresponding WARP simulation.
#

from warp import *
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import bisect
from dans_pymodules import MyColors
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtCore import QThread
import h5py
from random import sample
import datetime
from mpi4py import MPI
import itertools

__author__ = "Jared Hwang"
__doc__ = """Utilities for the PyRFQ Module"""

colors = MyColors()

class PyRfqUtils(object):

    def __init__(self, rfq, beam=[]):

        self._velocity_calculated = False
        self._zclose = rfq._field._zmax
        self._zfar = self._zclose + 0.01
        self._velocityarray = []
        self._velocityarray = np.array(self._velocityarray)
        self._average_velocity = 0.0
        self._wavelength = 0.0
        self._bunch_particles = {}
        self._bunchfound = False
        self._beam = []
        self._beam += beam
        self._rfq = rfq
        self._wavelengthbound = None

        self._max_steps_find_bunch = None

        self._velocity_count = top.npinject * 150

        self._app = pg.mkQApp()

        self._view = None
        self._x_top_rms = None
        self._x_bottom_rms = None
        self._y_top_rms = None
        self._y_bottom_rms = None

        self._view_scatter = None
        self._scatter_x = None
        self._scatter_y = None

        self._particle_outfile = None
        self._particle_data_group = None

        self._data_out_called = False

        # winon()

    # Helper functions to get all particles from all beams
    def _get_all_z_part(self, beamlist):
        return np.ravel([elem.getz() for elem in beamlist])

    def _get_all_x_part(self, beamlist):
        return np.ravel([elem.getx() for elem in beamlist])

    def _get_all_y_part(self, beamlist):
        return np.ravel([elem.gety() for elem in beamlist])

    def find_bunch_p(self, bunch_beam, max_steps):
        self._max_steps_find_bunch = top.it + max_steps
        if (np.max(bunch_beam.getz()) < self._rfq._field._zmax):
            print("Particles have not yet reached the end of the RFQ. Abandoning bunch finding.")
            return None

        starttime = time.time()

        for i in range(0, max_steps):
            step(1)
            self.measure_bunch_p(bunch_beam=bunch_beam)
            if (self._bunchfound):
                break

        if (not self._bunchfound):
            self._bunch_particles = None

        endtime = time.time()
        print("It took {} seconds to find a bunch.".format(endtime - starttime))

        return self._bunch_particles

    def measure_bunch_p(self, bunch_beam=None, beamlist=None):
        if bunch_beam==None:
            bunch_beam = self._beam[-1]

        if beamlist==None:
            beamlist = self._beam

        if self._bunchfound:
            return

        if not self._velocity_calculated:
            step_zdata = bunch_beam.getz()
            crossedZ = np.where(np.logical_and(step_zdata>(self._zclose-0.005), step_zdata<(self._zclose + 0.005)))
            velocities = bunch_beam.getvz()
            particle_velocities = velocities[crossedZ]
            self._velocityarray = np.concatenate((self._velocityarray, particle_velocities))
            if (len(self._velocityarray) > self._velocity_count):
                self._average_velocity = np.mean(self._velocityarray)
                self._wavelength = self._average_velocity / self._rfq.rf_freq
                self._velocity_calculated = True
                self._zfar = self._zclose + self._wavelength
                self._wavelengthbound = self._zfar
                print('_wavelengthbound: {}'.format(self._wavelengthbound))
                return

        elif self._velocity_calculated:

            print("self._zclose: {}  self._zfar: {}".format(self._zclose, self._zfar))
            z_positions = [elem for elem in bunch_beam.getz() if (self._zclose < elem < self._zfar)]
            print("Restul: {},  Desired:  {}".format(np.around(np.mean(z_positions), decimals=3), np.around((self._zfar + self._zclose) / 2, decimals=3)))

            if (np.around(np.mean(z_positions), decimals=3) == (np.around(((self._zfar - self._zclose) / 2) + self._zclose, decimals=3))):
                self._bunchfound = True

                for beam in self._beam:

                    step_zdata = beam.getz()
                    bunchparticles_indices = np.where(np.logical_and(step_zdata>(self._zclose), step_zdata<(self._zfar)))
                    self._bunch_particles[beam.name] = {}

                    self._bunch_particles[beam.name]["x"] = beam.getx()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["y"] = beam.gety()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["z"] = beam.getz()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["r"] = beam.getr()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["theta"] = beam.gettheta()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["vx"] = beam.getvx()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["vy"] = beam.getvy()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["vz"] = beam.getvz()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["ux"] = beam.getux()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["uy"] = beam.getuy()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["uz"] = beam.getuz()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["xp"] = beam.getxp()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["yp"] = beam.getyp()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["rp"] = beam.getrp()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["gaminv"] = beam.getgaminv()[bunchparticles_indices]


                bunch_particles = self._bunch_particles


                i = 0
                while os.path.exists("bunch_particles.%s.dump" % i):
                    i += 1


                comm = MPI.COMM_WORLD

                comm.Barrier()
                if (comm.Get_rank() == 0):
                    pickle.dump(bunch_particles, open("bunch_particles.%s.dump" % i, "wb"))

                print("Bunch found.")

    def find_bunch(self, bunch_beam, max_steps):

        self._max_steps_find_bunch = top.it + max_steps
        if (np.max(bunch_beam.getz()) < self._rfq._field._zmax):
            print("Particles have not yet reached the end of the RFQ. Abandoning bunch finding.")
            return None

        # starttime = time.time()

        for i in range(0, max_steps):
            step(1)
            self.measure_bunch(bunch_beam=bunch_beam)
            if (self._bunchfound):
                break

        if (not self._bunchfound):
            self._bunch_particles = None

        # endtime = time.time()
        print("It took {} seconds to find a bunch.".format(endtime - starttime))

        return self._bunch_particles


    def measure_bunch(self, bunch_beam=None, beamlist=None):

        if bunch_beam==None:
            bunch_beam = self._beam[-1]

        if beamlist==None:
            beamlist = self._beam

        if self._bunchfound:
            return

        if not self._velocity_calculated:
            crossedZ = bunch_beam.selectparticles(zc=self._zclose)
            velocities = bunch_beam.getvz()
            particle_velocities = velocities[crossedZ]
            self._velocityarray = np.concatenate((self._velocityarray, particle_velocities))
            if (len(self._velocityarray) > self._velocity_count):
                self._average_velocity = np.mean(self._velocityarray)
                self._wavelength = self._average_velocity / self._rfq.rf_freq
                self._velocity_calculated = True
                self._zfar = self._zclose + self._wavelength
                self._wavelengthbound = self._zfar
                print('_wavelengthbound: {}'.format(self._wavelengthbound))
                return

        elif self._velocity_calculated:

            tot_particles = list(zip(bunch_beam.getx(), bunch_beam.gety(), bunch_beam.getz()))
            #tot_particles = np.array(tot_particles)

            print("self._zclose: {}  self._zfar: {}".format(self._zclose, self._zfar))
            particles = [item for item in tot_particles if (self._zclose < item[2] < self._zfar)]
            z_positions = [item[2] for item in particles]
            print("Result: {},  Desired: {}".format(np.mean(z_positions), (self._zfar + self._zclose) / 2))
            print("RestulR: {},  Desired:  {}".format(np.around(np.mean(z_positions), decimals=2), np.around((self._zfar + self._zclose) / 2, decimals=2)))

            if (np.around(np.mean(z_positions), decimals=3) == (np.around(((self._zfar - self._zclose) / 2) + self._zclose, decimals=3))):
                self._bunchfound = True

                for beam in self._beam:
                    bunchparticles_indices = beam.selectparticles(zl=self._zclose, zu=self._zfar)
                    self._bunch_particles[beam.name] = {}

                    self._bunch_particles[beam.name]["x"] = beam.getx()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["y"] = beam.gety()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["z"] = beam.getz()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["r"] = beam.getr()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["theta"] = beam.gettheta()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["vx"] = beam.getvx()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["vy"] = beam.getvy()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["vz"] = beam.getvz()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["ux"] = beam.getux()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["uy"] = beam.getuy()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["uz"] = beam.getuz()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["xp"] = beam.getxp()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["yp"] = beam.getyp()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["rp"] = beam.getrp()[bunchparticles_indices]
                    self._bunch_particles[beam.name]["gaminv"] = beam.getgaminv()[bunchparticles_indices]


                bunch_particles = self._bunch_particles


                i = 0
                while os.path.exists("bunch_particles.%s.dump" % i):
                    i += 1


                comm = MPI.COMM_WORLD

                comm.Barrier()
                pickle.dump(bunch_particles, open("bunch_particles.%s.dump" % i, "wb"))

                print("Bunch found.")




    def plotXZparticles(self, beamlist=None, view=1):

        if beamlist==None:
            beamlist = self._beam

        plsys(view)

        plg([w3d.xmmin,w3d.xmmax],[self._rfq._field._zmin, self._rfq._field._zmin], color=red)
        plg([w3d.xmmin,w3d.xmmax],[self._rfq._field._zmax, self._rfq._field._zmax], color=red)

        if (self._wavelengthbound):
            plg([w3d.xmmin,w3d.xmmax],[self._wavelengthbound, self._wavelengthbound], color=red)


        self._rfq._conductors.draw()
        # pfzx(plotsg=0, cond=0, titles=False, view=view)
        for beam in beamlist:
            beam.ppzx(titles=False, view=view)

        limits(w3d.zmminglobal, w3d.zmmaxglobal)
        ptitles("", "Z (m)", "X (m)")

    def plotYZparticles(self, beamlist=None,  view=1):

        if beamlist==None:
            beamlist = self._beam

        plsys(view)

        plg([w3d.ymmin,w3d.ymmax],[self._rfq._field._zmin, self._rfq._field._zmin], color=red)
        plg([w3d.ymmin,w3d.ymmax],[self._rfq._field._zmax, self._rfq._field._zmax], color=red)


        if (self._wavelengthbound):
            plg([w3d.ymmin,w3d.ymmax],[self._wavelengthbound, self._wavelengthbound], color=red)


        self._rfq._conductors.draw()
        # pfzy(plotsg=0, cond=0, titles=False, view=view)

        for beam in beamlist:
            beam.ppzy(titles=False, view=view)

        limits(w3d.zmminglobal, w3d.zmmaxglobal)
        ptitles("", "Z (m)", "Y (m)")

    def plotXphase(self, beamlist=None, view=1):

        if beamlist==None:
            beamlist=self._beam

        plsys(view)
        for beam in beamlist:
            beam.ppxp()

    def plotYphase(self, beamlist=None, view=1):

        if beamlist==None:
            beamlist=self._beam

        plsys(view)
        for beam in beamlist:
            beam.ppyp()

    def beamplots(self, beamlist=None):

        if beamlist==None:
            beamlist=self._beam

        window()
        # fma()
        self.plotXZparticles(beamlist=beamlist, view=9)
        # refresh()

        # window(winnum=2)
        # fma()
        self.plotYZparticles(beamlist=beamlist, view=10)
        fma()
        refresh()

        # window(2)
        # fma()
        # self.plotXphase()
        # refresh()

        # window(3)
        # fma()
        # self.plotYphase()
        # refresh()

    def make_plots(self, beamlist=None, rate=10):

        if beamlist==None:
            beamlist=self._beam

        if top.it%rate == 0:
            self.beamplots(beamlist=beamlist)

    # def plot_rms_graph(self, start, end, bucketsize=0.001):

    #     beam = self._beam

    #     x = beam.getx()
    #     y = beam.gety()
    #     z = beam.getz()

    #     data = np.array(list(zip(x, y, z)))

    #     def rms(ray):
    #         temp = np.array(ray)
    #         temp = temp ** 2
    #         avg = temp.mean()
    #         avg = np.sqrt(avg)
    #         return avg

    #     bins = np.arange(start, end, bucketsize)
    #     zdigitized = np.digitize(z,bins)

    #     xrms_ray = []
    #     yrms_ray = []

    #     for i in range(1, len(bins) + 1):
    #         to_rms = data[zdigitized == i]
    #         if (len(to_rms) == 0):
    #             xrms_ray.append(0)
    #             yrms_ray.append(0)
    #             continue
    #         unzipped = list(zip(*to_rms))
    #         # if (rms(unzipped[0]) > 0.02):
    #         #     xrms_ray.append(0.02)
    #         # else:
    #         #     xrms_ray.append(rms(unzipped[0]))
    #         # if (rms(unzipped[1]) > 0.02):
    #         #     yrms_ray.append(0.02)
    #         # else:
    #         #     yrms_ray.append(rms(unzipped[1]))
    #         xrms_ray.append(rms(unzipped[0]))
    #         yrms_ray.append(rms(unzipped[1]))
    #         # xrms_ray.append(np.mean(unzipped[0]))
    #         # yrms_ray.append(np.mean(unzipped[1]))

    #     plt.plot(bins, xrms_ray)
    #     plt.plot(bins, yrms_ray)
    #     plt.show()

    def find_nearest(self, array,value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        else:
            return idx

    # Assumes symmetry about the z-axis
    def find_vane_mesh_boundaries(self, NX, sim_start, sim_end, sim_xmin, sim_xmax, vane_dist, vane_rad):

        refinement_array = np.linspace(0, 2*sim_xmax, NX)
        vane_top_edge = sim_xmax - vane_dist - (vane_rad)
        tvane_bottom_edge = vane_top_edge + (2*vane_rad)

        vane_top2_edge = sim_xmax + vane_dist - vane_rad
        bvane_bottom_edge = sim_xmax + vane_dist + vane_rad

        # Finding the place where the mesh around the north vane ends. Finds which refinement box it
        # lands between, then increases its size by one course box worth
        ymax_north_idx = bisect.bisect_left(refinement_array, tvane_bottom_edge)
        ymax_north = refinement_array[ymax_north_idx]

        if (ymax_north < 2*sim_xmax):
            ymax_north += (2*sim_xmax)/(NX-1)

        # Similarly for the mesh around the southern vane
        ymax_south = refinement_array[bisect.bisect_left(refinement_array, bvane_bottom_edge)]
        if (ymax_south < 2*sim_xmax):
            ymax_south += (2*sim_xmax)/(NX-1)

        ymin_north_idx = bisect.bisect_left(refinement_array, vane_top_edge)
        if (ymin_north_idx > 0):
            ymin_north_idx -= 1
        if (ymin_north_idx > 0):
            ymin_north_idx -= 1
        ymin_north = refinement_array[ymin_north_idx]

        ymin_south_idx = bisect.bisect_left(refinement_array, vane_top2_edge)
        if (ymin_south_idx > 0):
            ymin_south_idx -= 1
        if (ymin_south_idx > 0):
            ymin_south_idx -= 1
        ymin_south = refinement_array[ymin_south_idx]
        xmax_east = ymax_south
        xmin_east = ymin_south
        xmax_west = ymax_north
        xmin_west = ymin_north

        # Similar process for mesh boundaries across central axis
        vertical_xmax = refinement_array[bisect.bisect_left(refinement_array, sim_xmax + vane_rad)]
        if (vertical_xmax < 2*sim_xmax):
            vertical_xmax += (2*sim_xmax)/(NX-1)
        vertical_xmin_idx = bisect.bisect_left(refinement_array, sim_xmax - vane_rad)
        if (vertical_xmin_idx > 0):
            vertical_xmin_idx -= 1
        if (vertical_xmin_idx > 0):
            vertical_xmin_idx -= 1
        vertical_xmin = refinement_array[vertical_xmin_idx]

        lateral_ymin = vertical_xmin
        lateral_ymax = vertical_xmax

        xmax_east -= sim_xmax
        ymax_south -= sim_xmax
        xmin_east -= sim_xmax
        ymin_south -= sim_xmax
        xmax_west -= sim_xmax
        ymax_north -= sim_xmax
        xmin_west -= sim_xmax
        ymin_north -= sim_xmax

        lateral_ymin -= sim_xmax
        lateral_ymax -= sim_xmax
        vertical_xmin -= sim_xmax
        vertical_xmax -= sim_xmax

        boundaries = {
            "northmins": [vertical_xmin, ymin_north, sim_start],
            "northmaxs": [vertical_xmax, ymax_north, sim_end],
            "southmins": [vertical_xmin, ymin_south, sim_start],
            "southmaxs": [vertical_xmax, ymax_south, sim_end],
            "westmins":  [xmin_west, lateral_ymin, sim_start],
            "westmaxs":  [xmax_west, lateral_ymax, sim_end],
            "eastmins":  [xmin_east, lateral_ymin, sim_start],
            "eastmaxs":  [xmax_east, lateral_ymax, sim_end]
        }

        return boundaries


    def my_extractvar(self, name, varsuffix=None, pkg='top', ff=None):
        """
        Helper function which, given a name, returns the appropriate data. Note that
        name could actually be the variable itself, in which case, it is just
        returned.
        """
        if isinstance(name,str):
            # --- if varsuffix is specified, try to evaluate the name with the
            # --- suffix. If ok, return the result, otherwise, default to the
            # --- fortran variable in the specified package.
            if varsuffix is not None:
                vname = name + str(varsuffix)
                try:    result = ff.read(vname)
                except: result = None
                if result is not None: return result
                try:    result = __main__.__dict__[vname]
                except: result = None
                if result is not None: return result
            try:    result = ff.read(name+'@'+pkg)
            except: result = None
            if result is not None: return result
            return getattr(packageobject(pkg),name)
        else:
            return name

    def plot_xedges(self, plot_item_top, plot_item_bottom, pen=(200,200,200), symbol=None, symbolPen=(200,200,200), symbolBrush=(50,50,150), fillLevel=None,
                    brush=None, js=-1, zoffset=None,zscale=1.,scale=1., titleb=None,titles=1):
        """Plots beam X edges (centroid +- twice X rms) versus Z
        - symbol: A string describing the shape of symbols to use for each point. Optionally, this may also be a sequence of strings with a different symbol for each point.
        - symbolPen: The pen (or sequence of pens) to use when drawing the symbol outline.
        - symbolBrush: The brush (or sequence of brushes) to use when filling the symbol.
        - fillLevel: Fills the area under the plot curve to this Y-value.
        - brush: The brush to use when filling under the curve.
        - js=-1: species number, zero based. When -1, plots data combined from all
                 species
        - zoffset=zbeam: offset added to axis
        - zscale=1: scale of axis
          plots versus (zoffset + zmntmesh)/zscale
        - scale=1.: factor to scale data by
        - titleb="Z": bottom title
        - titles=1: specifies whether or not to plot titles"""


        varsuffix = None
        ff = None

        if zscale == 0.:
            raise Exception("zscale must be nonzero")

        if titleb is None:
            if zscale == 1.: titleb = "Z (m)"
            else: titleb = "Z"
        xbarz = self.my_extractvar('xbarz',varsuffix,'top',ff)[...,js]*scale
        xrmsz = self.my_extractvar('xrmsz',varsuffix,'top',ff)[...,js]*scale
        zmntmesh = self.my_extractvar('zmntmesh',varsuffix,'top',ff)
        if zoffset is None: zoffset = self.my_extractvar('zbeam',varsuffix,'top',ff)

        # plot_item.plot((zoffset+zmntmesh)/zscale, xbarz+2.*xrmsz, pen=pen, symbol=symbol, symbolPen=symbolPen, symbolBrush=symbolBrush, fillLevel=fillLevel, brush=brush)
        # plot_item.plot((zoffset+zmntmesh)/zscale, xbarz-2.*xrmsz, pen=pen, symbol=symbol, symbolPen=symbolPen, symbolBrush=symbolBrush, fillLevel=fillLevel, brush=brush)

        plot_item_top.setData((zoffset+zmntmesh)/zscale, xbarz+2.*xrmsz)
        plot_item_bottom.setData((zoffset+zmntmesh)/zscale, xbarz-2.*xrmsz)

        def gettitler(js):
            if js == -1: return "All species"
            else:        return "Species %d"%js

        # if titles:
        #     # ptitles("Beam X edges (xbar+-2*rms)",titleb,"(m)",
        #             # gettitler(js))

    #   pzxedges: Plots beam X edges (centroid +- twice Xrms) versus Z
    def plot_yedges(self, plot_item_top, plot_item_bottom, pen=(200,200,200), symbol=None, symbolPen=(200,200,200), symbolBrush=(50,50,150), fillLevel=None,
                    brush=None, js=-1, zoffset=None,zscale=1.,scale=1., titleb=None,titles=1):
        """Plots beam X edges (centroid +- twice X rms) versus Z
        - symbol: A string describing the shape of symbols to use for each point. Optionally, this may also be a sequence of strings with a different symbol for each point.
        - symbolPen: The pen (or sequence of pens) to use when drawing the symbol outline.
        - symbolBrush: The brush (or sequence of brushes) to use when filling the symbol.
        - fillLevel: Fills the area under the plot curve to this Y-value.
        - brush: The brush to use when filling under the curve.
        - js=-1: species number, zero based. When -1, plots data combined from all
                 species
        - zoffset=zbeam: offset added to axis
        - zscale=1: scale of axis
          plots versus (zoffset + zmntmesh)/zscale
        - scale=1.: factor to scale data by
        - titleb="Z": bottom title
        - titles=1: specifies whether or not to plot titles"""


        varsuffix = None
        ff = None

        if zscale == 0.:
            raise Exception("zscale must be nonzero")
        if titleb is None:
            if zscale == 1.: titleb = "Z (m)"
            else: titleb = "Z"
        ybarz = self.my_extractvar('ybarz',varsuffix,'top',ff)[...,js]*scale
        yrmsz = self.my_extractvar('yrmsz',varsuffix,'top',ff)[...,js]*scale
        zmntmesh = self.my_extractvar('zmntmesh',varsuffix,'top',ff)
        if zoffset is None: zoffset = self.my_extractvar('zbeam',varsuffix,'top',ff)

        plot_item_top.setData((zoffset+zmntmesh)/zscale, ybarz+2.*yrmsz)
        plot_item_bottom.setData((zoffset+zmntmesh)/zscale, ybarz-2.*yrmsz)

        def gettitler(js):
            if js == -1: return "All species"
            else:        return "Species %d"%js

        # if titles:
        #     ptitles("Beam Y edges (ybar+-2*rms)",titleb,"(m)",
        #             gettitler(js))


    # Setup the PyQTGraph realtime RMS Plot
    def rms_plot_setup(self, xpen=pg.mkPen(width=1.5, color=colors[6]), ypen=pg.mkPen(width=1.5,color=colors[5]),
                        xrange=[-0.1,1], yrange=[-0.01,0.01], title=None, labels=None):
        self._view = pg.PlotWidget(title=title, labels=labels)
        self._x_top_rms = pg.PlotDataItem(pen=xpen)
        self._x_bottom_rms = pg.PlotDataItem(pen=xpen)
        self._y_top_rms = pg.PlotDataItem(pen=ypen)
        self._y_bottom_rms = pg.PlotDataItem(pen=ypen)
        self._view.setRange(xRange=xrange, yRange=yrange)
        self._view.addItem(self._x_top_rms)
        self._view.addItem(self._x_bottom_rms)
        self._view.addItem(self._y_top_rms)
        self._view.addItem(self._y_bottom_rms)


    def plot_rms(self):
        #   pzxedges: Plots beam X and Y edges (centroid +- twice Xrms) versus Z
        #   Call me every time step
        self._view.show()
        self.plot_xedges(self._x_top_rms, self._x_bottom_rms)
        self.plot_yedges(self._y_top_rms, self._y_bottom_rms)

        QtGui.QGuiApplication.processEvents()


    def particle_plot_setup(self, xpen=pg.mkPen(width=1, color=colors[6]), ypen=pg.mkPen(width=1, color=colors[5]),
                            symbol='s', size=0.25, xrange=[-0.1,1], yrange=[-0.01,0.01], title=None, labels=None):
        # Setup the PyQTGraph realtime particle plot
        self._view_scatter = pg.PlotWidget(title=title, labels=labels)
        self._view_scatter.show()
        self._scatter_x = pg.ScatterPlotItem(pen=xpen, symbol=symbol, size=size)
        self._scatter_y = pg.ScatterPlotItem(pen=ypen, symbol=symbol, size=size)
        self._view_scatter.setRange(xRange=xrange, yRange=yrange)
        self._view_scatter.addItem(self._scatter_x)
        self._view_scatter.addItem(self._scatter_y)

    def plot_particles(self, factor=1, beamlist=None):
        # Plot the particles X and Y positions vs Z
        # Call me every time step
        if beamlist == None:
            beamlist = self._beam

        x_by_z_particles = list(zip(self._get_all_z_part(beamlist), self._get_all_x_part(beamlist)))
        factored_x_by_z = sample(x_by_z_particles, int(len(x_by_z_particles)*factor))
        self._scatter_x.setData(pos=factored_x_by_z)

        y_by_z_particles = list(zip(self._get_all_z_part(beamlist), self._get_all_y_part(beamlist)))
        factored_y_by_z = sample(y_by_z_particles, int(len(y_by_z_particles)*factor))
        self._scatter_y.setData(pos=factored_y_by_z)

        QtGui.QGuiApplication.processEvents()

    def get_rms_widget(self):
        return self._view

    def get_particle_widget(self):
        return self._view_scatter

    def write_hdf5_data(self, step_num, beamlist=None):
        # Write out the particle data to hdf5 file
        # step_num refers to top.it
        # Beamlist is a list of the WARP beam objects that the user wants data outputted for
        # Used in SERIAL
        if beamlist == None:
            beamlist = self._beam

        if not self._data_out_called:
            date = datetime.datetime.today()
            filename = date.strftime('%Y-%m-%dT%H:%M') + "_particle_data.hdf5"
            self._particle_outfile = h5py.File(filename, 'w')
            self._particle_outfile.attrs.__setitem__('PY_RFQ_HELPER', b'0.0.1')
            self._data_out_called = True

            # Store data to identify species later
            beam_identifier_list = self._particle_outfile.create_group('SpeciesList')
            for beam in beamlist:
                beam_identifier_list.create_dataset(beam.name, data=[beam.sm, beam.charge, beam.charge_state, beam.type.A, beam.type.Z])

        step_str = "Step#{}".format(step_num)

        _part_data = {'x': [], 'y': [], 'z': [],
                      'px': [], 'py': [], 'pz': [],
                      'm': [], 'q': [], "ENERGY": [],
                      'vx': [], 'vy': [], 'vz': [],
                      'ux': [], 'uy': [], 'uz': [],
                      'xp': [], 'yp': [], 'id': []}

        step_grp = self._particle_outfile.create_group(step_str)

        for beam in beamlist:
            _npart = beam.getn()
            _mass = beam.sm

            _part_data['x'] = np.concatenate((_part_data['x'], beam.getx()))
            _part_data['y'] = np.concatenate((_part_data['y'], beam.gety()))
            _part_data['z'] = np.concatenate((_part_data['z'], beam.getz()))
            _part_data['px'] = np.concatenate((_part_data['px'], beam.getux() * _mass))
            _part_data['py'] = np.concatenate((_part_data['py'], beam.getuy() * _mass))
            _part_data['pz'] = np.concatenate((_part_data['pz'], beam.getuz() * _mass))
            _part_data['m'] = np.concatenate((_part_data['m'], np.full(_npart, _mass)))
            _part_data['q'] = np.concatenate((_part_data['q'], np.full(_npart, beam.charge)))
            _part_data['ENERGY'] = np.concatenate((_part_data['ENERGY'], np.full(_npart, beam.ekin)))
            _part_data['vx'] = np.concatenate((_part_data['vx'], beam.getvx()))
            _part_data['vy'] = np.concatenate((_part_data['vy'], beam.getvy()))
            _part_data['vz'] = np.concatenate((_part_data['vz'], beam.getvz()))
            _part_data['ux'] = np.concatenate((_part_data['ux'], beam.getux()))
            _part_data['uy'] = np.concatenate((_part_data['uy'], beam.getuy()))
            _part_data['uz'] = np.concatenate((_part_data['uz'], beam.getuz()))
            _part_data['xp'] = np.concatenate((_part_data['xp'], beam.getxp()))
            _part_data['yp'] = np.concatenate((_part_data['yp'], beam.getyp()))
            _part_data['id'] = np.concatenate((_part_data['id'], beam.getssn()))

        for key in _part_data:
            step_grp.create_dataset(key, data=_part_data[key])


    def write_hdf5_data_p(self, step_num, beamlist=None):
        # Write out the particle data to hdf5 file
        # step_num refers to top.it
        # Beamlist is a list of the WARP beam objects that the user wants data outputted for
        # Used in PARALLEL
        if beamlist == None:
            beamlist = self._beam

        comm = MPI.COMM_WORLD

        if (comm.Get_rank() == 0):
            if not self._data_out_called:
                date = datetime.datetime.today()
                filename = date.strftime('%Y-%m-%dT%H:%M') + "_particle_data.hdf5"
                self._particle_outfile = h5py.File(filename, 'w')
                self._particle_outfile.attrs.__setitem__('PY_RFQ_HELPER', b'0.0.1')
                self._data_out_called = True

                # Store data to identify species later
                beam_identifier_list = self._particle_outfile.create_group('SpeciesList')
                for beam in beamlist:
                    # MASS, CHARGE
                    beam_identifier_list.create_dataset(beam.name, data=[beam.mass, beam.charge, beam.charge_state, beam.type.A, beam.type.Z])


        step_str = "Step#{}".format(step_num)

        _part_data = {'x': [], 'y': [], 'z': [], 'px': [], 'py': [], 'pz': [], 'm': [], 'q': [], "ENERGY": [],
                      'vx': [], 'vy': [], 'vz': [], 'id': []}

        for beam in beamlist:

            x_gathered = comm.gather(beam.xp, root=0)
            y_gathered = comm.gather(beam.yp, root=0)
            z_gathered = comm.gather(beam.zp, root=0)
            vx_gathered = comm.gather(beam.uxp, root=0)
            vy_gathered = comm.gather(beam.uyp, root=0)
            vz_gathered = comm.gather(beam.uzp, root=0)
            id_gathered = comm.gather(beam.ssn, root=0)

            if (comm.Get_rank() == 0):
                x_gathered = np.array(list(itertools.chain.from_iterable(x_gathered)))
                y_gathered = np.array(list(itertools.chain.from_iterable(y_gathered)))
                z_gathered = np.array(list(itertools.chain.from_iterable(z_gathered)))
                vx_gathered = np.array(list(itertools.chain.from_iterable(vx_gathered)))
                vy_gathered = np.array(list(itertools.chain.from_iterable(vy_gathered)))
                vz_gathered = np.array(list(itertools.chain.from_iterable(vz_gathered)))
                id_gathered = np.array(list(itertools.chain.from_iterable(id_gathered)))
                _npart = len(x_gathered)
                _mass = beam.mass
                px_gathered = vx_gathered * _mass
                py_gathered = vy_gathered * _mass
                pz_gathered = vz_gathered * _mass

                _part_data['x'] = np.concatenate((_part_data['x'], x_gathered))
                _part_data['y'] = np.concatenate((_part_data['y'], y_gathered))
                _part_data['z'] = np.concatenate((_part_data['z'], z_gathered))
                _part_data['px'] = np.concatenate((_part_data['px'], px_gathered)) # momenta
                _part_data['py'] = np.concatenate((_part_data['py'], py_gathered))
                _part_data['pz'] = np.concatenate((_part_data['pz'], pz_gathered))
                _part_data['m'] = np.concatenate((_part_data['m'], np.full(_npart, _mass)))
                _part_data['q'] = np.concatenate((_part_data['q'], np.full(_npart, beam.charge)))
                _part_data['ENERGY'] = np.concatenate((_part_data['ENERGY'], np.full(_npart, beam.ekin)))
                _part_data['vx'] = np.concatenate((_part_data['vx'], vx_gathered))
                _part_data['vy'] = np.concatenate((_part_data['vy'], vy_gathered))
                _part_data['vz'] = np.concatenate((_part_data['vz'], vz_gathered))
                _part_data['id'] = np.concatenate((_part_data['id'], id_gathered))

        if (comm.Get_rank() == 0):
            step_grp = self._particle_outfile.create_group(step_str)
            for key in _part_data:
                step_grp.create_dataset(key, data=_part_data[key])
