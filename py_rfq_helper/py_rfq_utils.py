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

colors = MyColors()

class PyRfqUtils(object):

    def __init__(self, rfq, beam): 

        self._velocity_calculated = False
        self._zclose = rfq._field._zmax
        self._zfar = self._zclose + 0.01
        self._velocityarray = []
        self._velocityarray = np.array(self._velocityarray)
        self._average_velocity = 0.0
        self._wavelength = 0.0
        self._bunch_particles = {}
        self._bunchfound = False
        self._beam = beam
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

        self._particle_data_file = None
        self._particle_data_group = None

        # winon(1, suffix='YZ')
        # winon(2, suffix="X'X")
        # winon(3, suffix="Y'Y")
        winon()

    def find_bunch(self, max_steps):
        self._max_steps_find_bunch = top.it + max_steps
        if (np.max(self._beam.getz()) < self._rfq._field._zmax):
            print("Particles have not yet reached the end of the RFQ. Abandoning bunch finding.")
            return None

        starttime = time.time()

        for i in range(0, max_steps):
            step(1)
            self.measure_bunch()
            if (self._bunchfound):
                break

        if (not self._bunchfound):
            self._bunch_particles = None

        endtime = time.time()
        print("It took {} seconds to find a bunch.".format(endtime - starttime))

        return self._bunch_particles


    def measure_bunch(self):

        if self._bunchfound:
            return

        if not self._velocity_calculated:
            crossedZ = self._beam.selectparticles(zc=self._zclose)
            velocities = self._beam.getvz()
            particle_velocities = velocities[crossedZ]
            self._velocityarray = np.concatenate((self._velocityarray, particle_velocities))

            if (len(self._velocityarray) > self._velocity_count):
                self._average_velocity = np.mean(self._velocityarray)
                self._velocity_calculated = True
                self._wavelength = self._average_velocity / self._rfq.rf_freq
                self._velocity_calculated = True
                self._zfar = self._zclose + self._wavelength

                self._wavelengthbound = self._zfar
                return
        
        if self._velocity_calculated:

            tot_particles = list(zip(self._beam.getx(), self._beam.gety(), self._beam.getz()))
            #tot_particles = np.array(tot_particles)
            
            print("self._zclose: {}  self._zfar: {}".format(self._zclose, self._zfar))
            particles = [item for item in tot_particles if (self._zclose < item[2] < self._zfar)]
            z_positions = [item[2] for item in particles]
            print("Result: {},  Desired: {}".format(np.mean(z_positions), (self._zfar + self._zclose) / 2))
            print("RestulR: {},  Desired:  {}".format(np.around(np.mean(z_positions), decimals=2), np.around((self._zfar + self._zclose) / 2, decimals=2)))

            if (np.around(np.mean(z_positions), decimals=3) == (np.around(((self._zfar - self._zclose) / 2) + self._zclose, decimals=3))):
                self._bunchfound = True
                
                bunchparticles_indices = self._beam.selectparticles(zl=self._zclose, zu=self._zfar)

                self._bunch_particles["x"] = self._beam.getx()[bunchparticles_indices]
                self._bunch_particles["y"] = self._beam.gety()[bunchparticles_indices]
                self._bunch_particles["z"] = self._beam.getz()[bunchparticles_indices]
                self._bunch_particles["r"] = self._beam.getr()[bunchparticles_indices]
                self._bunch_particles["theta"] = self._beam.gettheta()[bunchparticles_indices]
                self._bunch_particles["vx"] = self._beam.getvx()[bunchparticles_indices]
                self._bunch_particles["vy"] = self._beam.getvy()[bunchparticles_indices]
                self._bunch_particles["vz"] = self._beam.getvz()[bunchparticles_indices]
                self._bunch_particles["ux"] = self._beam.getux()[bunchparticles_indices]
                self._bunch_particles["uy"] = self._beam.getuy()[bunchparticles_indices]
                self._bunch_particles["uz"] = self._beam.getuz()[bunchparticles_indices]
                self._bunch_particles["xp"] = self._beam.getxp()[bunchparticles_indices]
                self._bunch_particles["yp"] = self._beam.getyp()[bunchparticles_indices]
                self._bunch_particles["rp"] = self._beam.getrp()[bunchparticles_indices]
                self._bunch_particles["gaminv"] = self._beam.getgaminv()[bunchparticles_indices]

                bunch_particles = self._bunch_particles


                i = 0
                while os.path.exists("bunch_particles.%s.dump" % i):
                    i += 1

                pickle.dump(bunch_particles, open("bunch_particles.%s.dump" % i, "wb"))

                print("Bunch found.")




    def plotXZparticles(self, view=1):

        plsys(view)

        plg([w3d.xmmin,w3d.xmmax],[self._rfq._field._zmin, self._rfq._field._zmin], color=red)
        plg([w3d.xmmin,w3d.xmmax],[self._rfq._field._zmax, self._rfq._field._zmax], color=red)

        if (self._wavelengthbound):
            plg([w3d.xmmin,w3d.xmmax],[self._wavelengthbound, self._wavelengthbound], color=red)   


        self._rfq._conductors.draw()
        # pfzx(plotsg=0, cond=0, titles=False, view=view)
        ppzx(titles=False, view=view)
        limits(w3d.zmminglobal, w3d.zmmaxglobal)
        ptitles("", "Z (m)", "X (m)")

    def plotYZparticles(self, view=1):
        plsys(view)

        plg([w3d.ymmin,w3d.ymmax],[self._rfq._field._zmin, self._rfq._field._zmin], color=red)
        plg([w3d.ymmin,w3d.ymmax],[self._rfq._field._zmax, self._rfq._field._zmax], color=red)


        if (self._wavelengthbound):
            plg([w3d.ymmin,w3d.ymmax],[self._wavelengthbound, self._wavelengthbound], color=red)   

        
        self._rfq._conductors.draw()
        # pfzy(plotsg=0, cond=0, titles=False, view=view)
        ppzy(titles=False, view=view)
        limits(w3d.zmminglobal, w3d.zmmaxglobal)
        ptitles("", "Z (m)", "Y (m)")

    def plotXphase(self, view=1):
        plsys(view)
        self._beam.ppxxp()

    def plotYphase(self, view=1):
        plsys(view)
        self._beam.ppyyp()
    
    def beamplots(self):
        window()
        # fma()
        self.plotXZparticles(view=9)
        # refresh()

        # window(winnum=2)
        # fma()
        self.plotYZparticles(view=10)
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

    def make_plots(self, rate=10):
        if top.it%rate == 0:
            self.beamplots()

    def plot_rms_graph(self, start, end, bucketsize=0.001):
    
        beam = self._beam

        x = beam.getx()
        y = beam.gety()
        z = beam.getz()

        data = np.array(list(zip(x, y, z)))

        def rms(ray):
            temp = np.array(ray)
            temp = temp ** 2
            avg = temp.mean()
            avg = np.sqrt(avg)
            return avg

        bins = np.arange(start, end, bucketsize)
        zdigitized = np.digitize(z,bins)

        xrms_ray = []
        yrms_ray = []

        for i in range(1, len(bins) + 1):
            to_rms = data[zdigitized == i]
            if (len(to_rms) == 0):
                xrms_ray.append(0)
                yrms_ray.append(0)
                continue
            unzipped = list(zip(*to_rms))
            # if (rms(unzipped[0]) > 0.02):
            #     xrms_ray.append(0.02)
            # else:
            #     xrms_ray.append(rms(unzipped[0]))
            # if (rms(unzipped[1]) > 0.02):
            #     yrms_ray.append(0.02)
            # else:
            #     yrms_ray.append(rms(unzipped[1]))
            xrms_ray.append(rms(unzipped[0]))
            yrms_ray.append(rms(unzipped[1]))
            # xrms_ray.append(np.mean(unzipped[0]))
            # yrms_ray.append(np.mean(unzipped[1]))

        plt.plot(bins, xrms_ray)
        plt.plot(bins, yrms_ray)
        plt.show()

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
        self._view.show()
        self.plot_xedges(self._x_top_rms, self._x_bottom_rms)
        self.plot_yedges(self._y_top_rms, self._y_bottom_rms)

        QtGui.QApplication.processEvents()


    def particle_plot_setup(self, xpen=pg.mkPen(width=1, color=colors[6]), ypen=pg.mkPen(width=1, color=colors[5]),
                            symbol='s', size=0.25, xrange=[-0.1,1], yrange=[-0.01,0.01], title=None, labels=None):
        self._view_scatter = pg.PlotWidget(title=title, labels=labels)
        self._view_scatter.show()
        self._scatter_x = pg.ScatterPlotItem(pen=xpen, symbol=symbol, size=size)
        self._scatter_y = pg.ScatterPlotItem(pen=ypen, symbol=symbol, size=size)
        self._view_scatter.setRange(xRange=xrange, yRange=yrange)
        self._view_scatter.addItem(self._scatter_x)
        self._view_scatter.addItem(self._scatter_y)

    def plot_particles(self, factor=1):
        x_by_z_particles = list(zip(self._beam.getz(),self._beam.getx()))
        factored_x_by_z = sample(x_by_z_particles, int(len(x_by_z_particles)*factor))
        self._scatter_x.setData(pos=factored_x_by_z)
        y_by_z_particles = list(zip(self._beam.getz(),self._beam.gety()))
        factored_y_by_z = sample(y_by_z_particles, int(len(y_by_z_particles)*factor))
        self._scatter_y.setData(pos=factored_y_by_z)
        QtGui.QApplication.processEvents()

    def get_rms_widget(self):
        return self._view

    def get_particle_widget(self):
        return self._view_scatter

    def write_particle_data(self, step_num, rate=5):
      
        if top.it == 1:
            date = datetime.datetime.today()
            filename = date.strftime('%Y-%m-%dT%H:%M') + "_particle_data.hdf5"
            self._particle_data_file = h5py.File(filename, 'w')
            self._particle_data_group = self._particle_data_file.create_group("particle_data")
        
        elif top.it%rate != 0:
            return

        particles = list(zip(self._beam.getx(),self._beam.gety(), self._beam.getz()))
        grp = self._particle_data_group.create_dataset(str(step_num), np.shape(particles), data=particles, dtype=np.double)


