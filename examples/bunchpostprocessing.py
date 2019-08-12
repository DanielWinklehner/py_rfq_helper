# Post processing tools for PyRFQ bunch data
# Jared Hwang July 2019

import numpy as np
from scipy import constants as const
import h5py
import matplotlib.pyplot as plt
from temp_particles import IonSpecies
from dans_pymodules import FileDialog, ParticleDistribution
from bunch_particle_distribution import BunchParticleDistribution
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import pickle
import os

__author__ = "Jared Hwang"
__doc__ = """Post processing utilities for bunches made using PyRFQ module"""


DIHYDROGENMASS = 3.34615e-27

class BunchPostProcessor(object):
    def __init__(self, species_name, rfq_freq, bunch_filename=None, sim_data_filename=None):

        self._species_name = species_name #
        self._rfq_end = None              # end of the rfq
        self._rfq_freq = rfq_freq             # rfq frequency
        self._velocity_min = 7500

        self._bunch_filename = bunch_filename       # Pickle file with bunch data
        self._sim_data_filename = sim_data_filename # hdf5 file with simulation data


        # Internal variables 
        self._bunch_found = False   
        self._velocity_calculated = False
        self._velocity_array = []
        self._zclose = self._rfq_end
        self._zfar  = None
        self._particledistribution = None
        self._distribution_data = None

    def find_bunch(self, max_steps, start_step, rfq_end, filename=None, step_interval=1, plot_bunch=True):

        self._rfq_end = rfq_end
        self._zclose = self._rfq_end

        if (self._sim_data_filename == None):
            if(filename == None):
                fd = FileDialog()
                filename = fd.get_filename()
        else:
            filename = self._sim_data_filename

        f = h5py.File(filename, 'r')

        max_steps_find_bunch = start_step + max_steps
        if (np.max(f["Step#"+str(start_step*step_interval)]['z']) < self._rfq_end):
            print("Particles have not yet reached the end of the RFQ. Abandoning bunch finding.")
            return None

        for i in range(0, max_steps):
            # step(1)
            print("Currently at: {}".format((start_step+i)*step_interval))
            self.measure_bunch(f["Step#"+str((start_step+i)*step_interval)], plot_bunch)
            if (self._bunch_found):
                break

    def measure_bunch(self, step_data, plot_bunch=True):
        global DIHYDROGENMASS

        zvelocities = np.array(step_data['pz']) / (DIHYDROGENMASS)
        if self._bunch_found:
            return

        if not self._velocity_calculated:
            step_zdata = np.array(step_data['z'])

            idx = np.where(np.logical_and(step_zdata>(self._rfq_end-0.01), step_zdata>(self._rfq_end+0.01)))
            self._velocity_array = np.concatenate((self._velocity_array, zvelocities[idx]))

            if (len(self._velocity_array) > self._velocity_min):
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

                if plot_bunch:
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


    def populate_distribution(self, ion, x, y, z, vx, vy, vz):
        self._particledistribution = BunchParticleDistribution(self._rfq_freq, ion=ion, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
        self._distribution_data = self._particledistribution.calculate_emittances()
        
        print(self._distribution_data['raw_data']['EYYPRMS'])


    def emittancePlots(self):

        filename = self._bunch_filename
        if (self._bunch_filename == None):
            if(filename == None):
                fd = FileDialog()
                filename = fd.get_filename()
        else:
            filename = self._bunch_filename

        pickle_in = open(filename, "rb")
        bunch_data = pickle.load(pickle_in)    

        bunch_species = IonSpecies(name="H2_1+", energy_mev=1)
        velocity_beta = np.mean((bunch_data['H2+']['vz']) / const.value("speed of light in vacuum"))
        bunch_species.calculate_from_velocity_beta(beta=velocity_beta)

        self.populate_distribution(bunch_species,
                              x=bunch_data['H2+']['x'],
                              y=bunch_data['H2+']['y'],
                              z=bunch_data['H2+']['z'],
                              vx=bunch_data['H2+']['vx'],
                              vy=bunch_data['H2+']['vy'],
                              vz=bunch_data['H2+']['vz'])

        fig = plt.figure(figsize=(10, 10)) 
        plt.subplots_adjust(left=0.1, right=0.95, hspace=0.3)  
        ax1 = fig.add_subplot(221, xlim=(-0.02, 0.02), ylim=(-0.4, 0.4))
        ax2 = fig.add_subplot(222, xlim=(-0.02, 0.02), ylim=(-0.4, 0.4))
        ax3 = fig.add_subplot(223)#, ylim=(-1e-16, 6e-15))
        ax4 = fig.add_subplot(224, xlim=(0, 5), ylim=(0, 5))
        ax4.axis('off')

        ax1.set_title("X' by X")
        ax2.set_title("Y' by Y")
        ax3.set_title("E by Phase")

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

        zscale, zunits = 1, 'rad'
        ticks_z = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/zscale))
        ax3.xaxis.set_major_formatter(ticks_z)
        ax3.set_xlabel("Phase " + '(' + zunits + ')')


        energy_array = 0.5 * 3.34615e-27 * (np.square(bunch_data['H2+']['vz']))
        mean_energy = np.mean(energy_array)
        mean_vz = np.mean(bunch_data['H2+']['vz'])
        wavelength = mean_vz / self._rfq_freq


        z_array = bunch_data['H2+']['z'] - np.mean(bunch_data['H2+']['z'])
        z_phase_array = z_array * 2 * np.pi / wavelength 


        ax1.plot(bunch_data['H2+']['x'], bunch_data['H2+']['xp'], 'bo', ms=0.3)
        ax2.plot(bunch_data['H2+']['y'], bunch_data['H2+']['yp'], 'bo', ms=0.3)
        ax3.plot(z_phase_array, energy_array, 'bo', ms=0.3)
        


        ax4.text(0, 4, 'H2+')
        ax4.text(0, 3.7, 'Number of particles: {}'.format(len(bunch_data['H2+']['x'])))
        ax4.text(0, 3.3, "Mean energy: {} keV".format(mean_energy*6.242e18 / 1e3 ))
        ax4.text(0, 2.9, "EXXPRMS: {} {}".format(self._distribution_data['data'][13][1], self._distribution_data['data'][13][2]))
        ax4.text(0, 2.6, "EYYPRMS: {} {}".format(self._distribution_data['data'][14][1], self._distribution_data['data'][14][2]))
        ax4.text(0, 2.3, "EXYPRMS: {} {}".format(self._distribution_data['data'][15][1], self._distribution_data['data'][15][2]))
        ax4.text(0, 2, "EYXPRMS: {} {}".format(self._distribution_data['data'][16][1], self._distribution_data['data'][16][2]))
        ax4.text(0, 1.7, "Long emittance: {} {}".format(self._distribution_data['data'][17][1] * 1e3, 'KeV-rad'))
                

        ax4.text(0, 1.2, "EXXPRMS norm: {} {}".format(self._distribution_data['data'][18][1], self._distribution_data['data'][18][2]))
        ax4.text(0, 0.9, "EYYPRMS norm: {} {}".format(self._distribution_data['data'][19][1], self._distribution_data['data'][19][2]))
        ax4.text(0, 0.6, "EXYPRMS norm: {} {}".format(self._distribution_data['data'][20][1], self._distribution_data['data'][20][2]))
        ax4.text(0, 0.3, "EYXPRMS norm: {} {}".format(self._distribution_data['data'][21][1], self._distribution_data['data'][21][2]))
        ax4.text(0, 0, "Long norm emittance: {} {}".format(self._distribution_data['data'][22][1] * 1e3, 'KeV-rad'))
        

        plt.show() 


    def make_3d_bunch_plot(self, filename=None):
        from mpl_toolkits.mplot3d import Axes3D

        if (self._bunch_filename == None):
            if(filename == None):
                fd = FileDialog()
                filename = fd.get_filename()
        else:
            filename = self._bunch_filename

        pickle_in = open(filename, "rb")
        bunch_data = pickle.load(pickle_in)

        fig = plt.figure(figsize=(13,9))
        ax = Axes3D(fig)

        ax.set_zlabel("Y (m)")
        ax.set_ylabel("X (m)")
        ax.set_xlabel("Z (m)")
        ax.set_title("Bunch Particles")

        global DIHYDROGENMASS
        c = 0.5 * DIHYDROGENMASS * (np.square(np.array(bunch_data['H2+']['vz'])) * 6.242e15)

        plot = ax.scatter(bunch_data['H2+']['z'], bunch_data['H2+']['x'], bunch_data['H2+']['y'], s=1, c=c, cmap='plasma')
        
        clb = fig.colorbar(plot, ax=ax)
        clb.set_label('Energy (keV)')

        plt.show()

def main():
    # emittancePlots()
    # find_bunch(2000, 1700)

    postprocessor = BunchPostProcessor('H2_1+', 32.8e6)
    # postprocessor.find_bunch(2000, 900, 1.4, step_interval=10)
    postprocessor.emittancePlots()
    # postprocessor.make_3d_bunch_plot()

if __name__ == '__main__':
    main()
