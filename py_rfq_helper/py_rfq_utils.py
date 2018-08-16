from warp import *
import numpy as np

class PyRfqUtils(object):

    def __init__(self, rfq, beam):

        self._velocity_calculated = False
        self._zclose = rfq._field._zmax
        self._zfar = self._zclose + 0.01
        self._velocityarray = []
        self._average_velocity = 0.0
        self._wavelength = 0.0
        self._bunch_particles = []
        self._bunchfound = False
        self._beam = beam
        self._rfq = rfq
        self._wavelengthbound = None
        
    def find_bunch(self):

        if self._bunchfound:
            return

        if not self._velocity_calculated:
            crossedZ = self._beam.selectparticles(zc=self._zclose)
            velocities = self._beam.getvz()
            particle_velocities = [velocities[i] for i in crossedZ]
            self._velocityarray = self._velocityarray + particle_velocities
            print("length: {}".format(len(self._velocityarray)))

            if (len(self._velocityarray) > 10000):
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

                bx = [] # x y z positions
                by = []
                bz = []
                br = [] # r and theta
                btheta = []
                bvx = [] # velocities
                bvy = []
                bvz = []
                bux = [] # momenta
                buy = []
                buz = []
                bxp = [] # tranverse normalized velocities
                byp = []
                brp = []
                bgaminv = [] # gamma inverse 

                tbx = self._beam.getx()
                tby = self._beam.gety()
                tbz = self._beam.getz()
                tbr = self._beam.getr()
                tbtheta = self._beam.gettheta()
                tbvx = self._beam.getvx()
                tbvy = self._beam.getvy()
                tbvz = self._beam.getvz()
                tbux = self._beam.getux()
                tbuy = self._beam.getuy()
                tbuz = self._beam.getuz()
                tbxp = self._beam.getxp()
                tbyp = self._beam.getyp()
                tbrp = self._beam.getrp()
                tbgaminv = self._beam.getgaminv()

                for i in bunchparticles_indices:
                    bx.append(tbx[i])
                    by.append(tby[i])
                    bz.append(tbz[i])
                    br.append(tbr[i])
                    btheta.append(tbtheta[i])
                    bvx.append(tbvx[i])
                    bvy.append(tbvy[i])
                    bvz.append(tbz[i])
                    bux.append(tbux[i]) 
                    buy.append(tbuy[i])
                    buz.append(tbuz[i])
                    bxp.append(tbxp[i])
                    byp.append(tbyp[i])
                    brp.append(tbrp[i])
                    bgaminv.append(tbgaminv[i])


                self._bunch_particles = zip(bx, by, bz, br, btheta, bvx, bvy, bvz, bux, buy, buz, bxp, byp, brp, bgaminv)
                
                with open("bunchparticles.dump", 'w') as outfile:
                    outfile.write("x, y, z, r, theta, vx, vy, vz, ux, uy, uz, xp, yp, rp, gaminv\n")
                    for bx, by, bz, br, btheta, bvx, bvy, bvz, bux, buy, buz, bxp, byp, brp, bgaminv in self._bunch_particles:
                        outfile.write("{:.4e}   {:.4e}   {:.4e}   {:.4e}   {:.4e}   {:.4e}   {:.4e}   {:.4e}   {:.4e}   {:.4e}   {:.4e}   {:.4e}   {:.4e}   {:.4e}   {:.4e}\n".format(bx, by, bz, br, btheta, bvx, bvy, bvz, bux, buy, buz, bxp, byp, brp, bgaminv))

                exit(1)

    def make_plots(self):
