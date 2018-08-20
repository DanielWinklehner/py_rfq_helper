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

        winon(1, suffix='YZ')
        winon(2, suffix="X'X")
        winon(3, suffix="Y'Y")
        winon()

    def find_bunch(self, max_steps):
        self._max_steps_find_bunch = top.it + max_steps
        print("Find bunch!")
        print(np.max(self._beam.getz()))
        print(self._rfq._field._zmax)
        if (np.max(self._beam.getz()) < self._rfq._field._zmax):
            print("Particles have not yet reached the end of the RFQ. Abandoning bunch finding.")
            return None

        starttime = time.time()

        for i in range(0, max_steps):
            step(1)
            self.measure_bunch()
            if (self._bunchfound):
                break

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
            print("length: {}".format(len(self._velocityarray)))

            if (len(self._velocityarray) > self._velocity_count):
                print("found a velocity!!!!!!!!!")
                self._average_velocity = np.mean(self._velocityarray)
                self._velocity_calculated = True
                self._wavelength = self._average_velocity / self._rfq.rf_freq
                print("self._wavelength:  {}".format(self._wavelength))
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

            if (np.around(np.mean(z_positions), decimals=2) == (np.around(((self._zfar - self._zclose) / 2) + self._zclose, decimals=2))):
                print("==========================\nFound a bunch!\n=================================")
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
        window(0)
        fma()
        self.plotXZparticles()
        refresh()

        window(1)
        fma()
        self.plotYZparticles()
        refresh()

        window(2)
        fma()
        self.plotXphase()
        refresh()

        window(3)
        fma()
        self.plotYphase()
        refresh()

    def make_plots(self):
        if top.it%1 == 0:
            self.beamplots()

    def plot_rms_graph(self, start, end, bucketsize=0.002):
        
        beam = self._beam

        x = beam.getx()
        y = beam.gety()
        z = beam.getz()

        def rms(ray):
            temp = np.array(ray)
            temp = temp ** 2
            avg = temp.mean()
            avg = np.sqrt(avg)
            return avg

        bins = np.linspace(start, end, bucketsize)
        xdigitized = np.digitize(x, bins)
        ydigitized = np.digitize(y, bins)

