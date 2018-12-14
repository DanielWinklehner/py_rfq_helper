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
