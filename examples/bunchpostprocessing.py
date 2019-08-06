# Post processing tools for PyRFQ bunch data
# Jared Hwang July 2019

import numpy as np
import h5py
import matplotlib.pyplot as plt
from dans_pymodules import FileDialog
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import pickle
import os

DIHYDROGENMASS = 3.34615e-27

class BunchPostProcessor(object):
    def __init__(self):
        self._field_zmax = 1.4
        self._bunch_found = False
        self._velocity_calculated = False
        self._velocity_array = []
        self._velocity_count = 7500
        self._rfq_freq = 32.8e6
        self._zclose = self._field_zmax     
        self._zfar  = None
        self._plot_bunch = True
        DIHYDROGENMASS = 3.34615e-27

    def find_bunch(self, max_steps, start_step, step_interval=1):

        fd = FileDialog()
        filename = fd.get_filename()

        f = h5py.File(filename, 'r')

        max_steps_find_bunch = start_step + max_steps
        if (np.max(f["Step#"+str(start_step*step_interval)]['z']) < self._field_zmax):
            print("Particles have not yet reached the end of the RFQ. Abandoning bunch finding.")
            return None

        for i in range(0, max_steps):
            # step(1)
            print("Currently at: {}".format((start_step+i)*step_interval))
            self.measure_bunch(f["Step#"+str((start_step+i)*step_interval)])
            if (self._bunch_found):
                break

    def measure_bunch(self, step_data):
        global DIHYDROGENMASS

        zvelocities = np.array(step_data['pz']) / (DIHYDROGENMASS)
        if self._bunch_found:
            return

        if not self._velocity_calculated:
            step_zdata = np.array(step_data['z'])

            idx = np.where(np.logical_and(step_zdata>(self._field_zmax-0.01), step_zdata>(self._field_zmax+0.01)))
            self._velocity_array = np.concatenate((self._velocity_array, zvelocities[idx]))

            if (len(self._velocity_array) > self._velocity_count):
                average_velocity = np.mean(self._velocity_array)
                wavelength = average_velocity / self._rfq_freq
                self._velocity_calculated = True
                self._zfar = self._zclose + wavelength
                return

        if self._velocity_calculated:
            zpart = np.array(step_data['z'])
            # particles = [item for item in tot_particles if (self._zclose < item[2] < self._zfar)]
            z_positions = [item for item in zpart if (self._zclose < item < self._zfar)]
            print("Result: {},  Desired: {}".format(np.mean(z_positions), (self._zfar + self._zclose) / 2))
            print("RestulR: {},  Desired:  {}".format(np.around(np.mean(z_positions), decimals=3), np.around((self._zfar + self._zclose) / 2, decimals=3)))

            if (np.around(np.mean(z_positions), decimals=3) == (np.around(((self._zfar - self._zclose) / 2) + self._zclose, decimals=3))):
                self._bunch_found = True
                
                step_zdata = np.array(step_data['z'])

                bunchparticles_indices = np.where(np.logical_and(step_zdata>(self._zclose), step_zdata<(self._zfar)))
                bunch_particles = {}
                bunch_particles['H2+'] = {}
                
                bunch_particles['H2+']["x"] = np.array(np.array(step_data['x'])[bunchparticles_indices])
                bunch_particles['H2+']["y"] = np.array(np.array(step_data['y'])[bunchparticles_indices])
                bunch_particles['H2+']["z"] = np.array(np.array(step_data['z'])[bunchparticles_indices])
                bunch_particles['H2+']["px"] = np.array(np.array(step_data['px'])[bunchparticles_indices])
                bunch_particles['H2+']["py"] = np.array(np.array(step_data['py'])[bunchparticles_indices])
                bunch_particles['H2+']["pz"] = np.array(np.array(step_data['pz'])[bunchparticles_indices])

                vx = np.array(step_data['px'])[bunchparticles_indices] / DIHYDROGENMASS
                vy = np.array(step_data['py'])[bunchparticles_indices] / DIHYDROGENMASS
                vz = np.array(step_data['pz'])[bunchparticles_indices] / DIHYDROGENMASS
                xp = vx / vz
                yp = vy / vz

                bunch_particles['H2+']["vx"] = vx
                bunch_particles['H2+']["vy"] = vy
                bunch_particles['H2+']["vz"] = vz
                bunch_particles['H2+']["xp"] = xp
                bunch_particles['H2+']["yp"] = yp

                i = 0
                while os.path.exists("bunch_particles.%s.dump" % i):
                    i += 1

                pickle.dump(bunch_particles, open("bunch_particles.%s.dump" % i, "wb"))

                print("Bunch found.")

                if self._plot_bunch:
                    fig, ax = plt.subplots(figsize = (20, 8))
                    ax.axvline(x=self._zclose, color='green')
                    ax.axvline(x=self._zfar, color='green')

                    c = 0.5 * DIHYDROGENMASS * (np.square(np.array(step_data['pz']) / DIHYDROGENMASS)) * 6.242e15

                    # particlesall, = plt.plot([], [], 'bo', ms=0.5, color='blue')
                    # particlesbunch, = plt.plot([], [], 'bo', ms=0.5, color='red')
                    ax.set_xlim((-0.1, 1.8))
                    ax.set_ylim((-0.025, 0.025))
                    ax.set_xlabel('Z (m)')
                    ax.set_ylabel('X (m)')
                    ax.set_title('X by Z position down beam')

                    # particlesall.set_data(step_data['z'], step_data['x'])
                    # particlesbunch.set_data(bunch_particles['H2+']["z"], bunch_particles['H2+']["x"])
                    # print(np.max(bunch_particles['H2+']['z']))
                    fig.tight_layout(pad=5)
                    particlesall = ax.scatter(step_data['z'], step_data['x'], s=0.5, c=c, cmap='plasma')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='1%', pad=0.05)
                    clb = fig.colorbar(particlesall, cax=cax)
                    clb.set_label('Energy (keV)', rotation=-90)
                    # particlesbunch = ax.scatter(bunch_particles['H2+']["z"], bunch_particles['H2+']["x"], s=0.5, c='red')

                    plt.show()


    def emittancePlots(self):

        fd = FileDialog()
        filename = fd.get_filename()

        pickle_in = open(filename, "rb")
        bunch_data = pickle.load(pickle_in)    

        fig = plt.figure(figsize=(10, 10)) 
        plt.subplots_adjust(left=0.1, right=0.95, hspace=0.3)  
        ax1 = fig.add_subplot(221, xlim=(-0.02, 0.02), ylim=(-0.4, 0.4))
        ax2 = fig.add_subplot(222, xlim=(-0.02, 0.02), ylim=(-0.4, 0.4))
        ax3 = fig.add_subplot(223)#, ylim=(-1e-16, 6e-15))
        ax4 = fig.add_subplot(224, xlim=(0, 5), ylim=(0, 5))
        ax4.axis('off')

        ax1.set_title("X' by X")
        ax2.set_title("Y' by Y")
        ax3.set_title("E by Z")

        yscale, yunits = 1e-3, 'mrad'
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/yscale))
        ax1.yaxis.set_major_formatter(ticks_y)
        ax2.yaxis.set_major_formatter(ticks_y)
        ax1.set_ylabel("X' " + '(' + yunits + ')')
        ax2.set_ylabel("Y' " + '(' + yunits + ')')

        xscale, xunits = 1e-2, 'cm'
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/xscale))
        ax1.xaxis.set_major_formatter(ticks_x)
        ax2.xaxis.set_major_formatter(ticks_x)
        ax1.set_xlabel("X " + '(' + xunits + ')')
        ax2.set_xlabel("Y " + '(' + xunits + ')')

        escale, eunits = 6.242e18 / 1e3, 'KeV'
        ticks_e = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*escale))
        ax3.yaxis.set_major_formatter(ticks_e)
        ax3.set_ylabel("E " + '(' + eunits + ')')

        zscale, zunits = 1e-2, 'cm'
        ticks_z = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/zscale))
        ax3.xaxis.set_major_formatter(ticks_z)
        ax3.set_xlabel("Z " + '(' + zunits + ')')


        # energy_array = 0.5 * 3.34615e-27 * (np.square(bunch_data['H2+']['vx']) + np.square(bunch_data['H2+']['vy']) + np.square(bunch_data['H2+']['vx']))
        # print(energy_array)
        # energy_mean = np.mean(energy_array)
        # energy_array = energy_array - energy_mean
        energy_array = 0.5 * 3.34615e-27 * (np.square(bunch_data['H2+']['vz']))
        mean_energy = np.mean(energy_array)
        z_array = bunch_data['H2+']['z'] - np.mean(bunch_data['H2+']['z'])

        print(bunch_data['H2+']['vx'][0], bunch_data['H2+']['vz'][0], bunch_data['H2+']['xp'][0])

        ax1.plot(bunch_data['H2+']['x'], bunch_data['H2+']['xp'], 'bo', ms=0.3)
        ax2.plot(bunch_data['H2+']['y'], bunch_data['H2+']['yp'], 'bo', ms=0.3)
        ax3.plot(z_array, energy_array, 'bo', ms=0.3)
        ax4.text(1, 4, 'H2+')
        ax4.text(1, 3, 'Number of particles: {}'.format(len(bunch_data['H2+']['x'])))
        ax4.text(1, 2, "Mean energy: {}".format(mean_energy*6.242e18 / 1e3 ))
        plt.show() 

def main():
    # emittancePlots()
    # find_bunch(2000, 1700)

    postprocessor = BunchPostProcessor()
    postprocessor.find_bunch(2000,900, step_interval=10)
    # postprocessor.emittancePlots()

if __name__ == '__main__':
    main()
