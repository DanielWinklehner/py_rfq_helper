# Post processing tools for PyRFQ bunch data
# Jared Hwang July 2019

import numpy as np
from scipy import constants as const
from scipy import stats
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
import hdbscan  

__author__ = "Jared Hwang"
__doc__ = """Post processing utilities for bunches made using PyRFQ module"""


# Some constants
clight = const.value("speed of light in vacuum")  # (m/s)
amu_kg = const.value("atomic mass constant")  # (kg)
echarge = const.value("elementary charge")

DIHYDROGENMASS = 3.34615e-27

# Class for handling bunch related post processing actions
class BunchPostProcessor(object):
    def __init__(self, rfq_freq, bunch_filename=None, sim_data_filename=None):
        # bunch_filename: filename of the pickle dump with the bunch data in it
        # sim_data_filename: filename of the hdf5 dump with the simulation data in it

        self._rfq_end = None              # end of the rfq
        self._rfq_freq = rfq_freq             # rfq frequency

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

        self._z_ekin = None
        self._z_phase = None

    def find_bunch(self, max_steps, start_step, rfq_end, velocity_min_sample=7500, bunch_species_name=None, filename=None, step_interval=1, plot_bunch_name=None):
        # Finds and dumps a bunch given hdf5 simulation dump

        # max_steps: max # steps code should try to find a bunch
        # start_step: step at which bunch finding should begin
        # rfq_end: (m) value of the end of the rfq
        # velocity_min_sample: minunum number of velocity samples to calculate wavelength
        # bunch_species_name: name (string) of the species that should be used to find the bunch
        # filename: hdf5 dump filename
        # step_interval: interval at which the original simulation was run at (how many steps in between saved steps)
        # plot_bunch_name: name of the species to plot after the bunch has been found. If None, will not plot
        # 
        # if bunch_species_name is None, bunch will be found using all species.
        #

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
        
        # Creating the list of species to measure a bunch from
        # If passed in species name is None, then assume measure all species.
        species_list = f['SpeciesList']
        species_dict = {}
        if (bunch_species_name is None):
            for key in species_list:
                species_dict[key] = tuple(species_list[key])
        else:
            species_dict[bunch_species_name] = tuple(species_list[bunch_species_name])


        steps_only_list = list(f.keys())
        steps_only_list.remove("SpeciesList")
        step_array_int = np.array([int(elem[5:]) for elem in np.array(steps_only_list)])

        # ensure start_step has data
        if (start_step not in step_array_int):
            if (start_step < step_array_int.min()):
                print("Data has not started to be collected at that starting step yet. Exiting.")
                exit()
            if (start_step > step_array_int.max()):
                print("Requested start step is past last data collected step. Exiting")
                exit()
            else:
                print("Requested step is within collected data steps but not congruent with step interval. Finding nearest step as starting point")
                idx = (np.abs(step_array_int - start_step)).argmin()
                start_step = step_array_int[idx]
                print("New start step: {}".format(start_step))

        if (np.max(f["Step#"+str(start_step)]['z']) < self._rfq_end):
            print("Particles have not yet reached the end of the RFQ. Abandoning bunch finding.")
            return None

        for i in range(0, max_steps):
            # Only calls measure_bunch for steps with data in them
            print("Currently at: {}".format((start_step+(i*step_interval))))
            self.measure_bunch(f["Step#"+str((start_step)+i*step_interval)], velocity_min_sample=velocity_min_sample, species_dict=species_dict, plot_bunch_name=plot_bunch_name)
            if (self._bunch_found):
                break


    def plot_bunch(self, xdata, ydata, velocity_data=None, mass=None):
        # Plots the particles down the rfq with the found bunch highlighted
        # xdata: data on the x axis
        # ydata: data on the y axis
        # velocity_data: z velocity of particles
        # mass: mass of particles
        # if mass == None, energy will not be displayed

        fig, ax = plt.subplots(figsize = (20, 8))
        ax.axvline(x=self._zclose, color='green')
        ax.axvline(x=self._zfar, color='green')

        # particlesall, = plt.plot([], [], 'bo', ms=0.5, color='blue')
        # particlesbunch, = plt.plot([], [], 'bo', ms=0.5, color='red')
        ax.set_xlim((-0.014, 1.6))
        ax.set_ylim((-0.025, 0.025))
        ax.set_xlabel('Z (m)')

        xscale, xunits = 1e-2, 'cm'
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/xscale))
        ax.yaxis.set_major_formatter(ticks_x)
        ax.set_ylabel('X (' + xunits + ')')
        ax.set_title('X by Z position down beam')

        # particlesall.set_data(step_data['z'], step_data['x'])
        # particlesbunch.set_data(bunch_particles['H2+']["z"], bunch_particles['H2+']["x"])
        # print(np.max(bunch_particles['H2+']['z']))

        fig.tight_layout(pad=5)
        if mass == None:
            particlesall = ax.scatter(np.array(xdata), np.array(ydata), s=0.5)
        else:
            # Energy calculation 
            # NOT RELATIVISTIC RIGHT NOW
            c = 0.5 * mass * (np.square(np.array(velocity_data))) * 6.242e15
            particlesall = ax.scatter(np.array(xdata), np.array(ydata), s=0.5, c=c, cmap='plasma')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='1%', pad=0.05)
            clb = fig.colorbar(particlesall, cax=cax)
            clb.set_label('Energy (keV)', rotation=-90)
        
        plt.show()


    def measure_bunch(self, step_data, velocity_min_sample=7500, species_dict=None, plot_bunch_name=None):
        # Measures the particles to find bunch. Should not be called by user
        # step_data from the dump
        # species_dict: dictionary of species names
        # plot_bunch_name: name of species to plot. If none, will not plot

        if self._bunch_found:
            return

        zvelocities = np.array(step_data['vz'])
        if not self._velocity_calculated: 
            # Gather velocities at end of RFQ to calculate wavelength
            idx = []
            if (len(species_dict.keys()) == 1): #only one species
                step_zdata = np.array(step_data['z'])
                step_mdata = np.array(step_data['m'])
                step_qdata = np.array(step_data['q'])

                # Extract indices where particles are within range of the end of the rfq, 
                # and the mass and charge match the specified species
                key = list(species_dict.keys())[0]
                desired_mass = species_dict[key][0]
                desired_charge = species_dict[key][1]
                for i in range(0, len(step_zdata)):
                    if step_zdata[i] < (self._rfq_end-0.01):
                        continue   
                    if step_zdata[i] > (self._rfq_end+0.01):
                        continue
                    if step_mdata[i] != desired_mass:
                        continue
                    if step_qdata[i] != desired_charge:
                        continue
                    idx.append(i)
                self._velocity_array = np.concatenate((self._velocity_array, zvelocities[idx]))

            else: #all species
                step_zdata = np.array(step_data['z'])
                idx = np.where(np.logical_and(step_zdata>(self._rfq_end-0.01), step_zdata<(self._rfq_end+0.01)))            
                self._velocity_array = np.concatenate((self._velocity_array, zvelocities[idx]))

            if (len(self._velocity_array) > velocity_min_sample): # if number of samples collected > defined minimum
                average_velocity = np.mean(self._velocity_array) # calculate wavelength
                wavelength = average_velocity / self._rfq_freq
                self._velocity_calculated = True
                self._zfar = self._zclose + wavelength
                return

        if self._velocity_calculated: # move on to looking for a bunch
            z_positions = []
            if (len(species_dict.keys()) == 1): # if only looking for bunch of one species

                step_zdata = np.array(step_data['z'])
                step_mdata = np.array(step_data['m'])
                step_qdata = np.array(step_data['q'])

                key = list(species_dict.keys())[0]
                desired_mass = species_dict[key][0]
                desired_charge = species_dict[key][1]
                idx = []
                for i in range(0, len(step_zdata)):    # collect particles that are within wavelength, have the right mass
                    if step_zdata[i] < (self._zclose): # and the right charge for specified species
                        continue
                    if step_zdata[i] > (self._zfar):
                        continue
                    if step_mdata[i] != desired_mass:
                        continue
                    if step_qdata[i] != desired_charge:
                        continue
                            
                    idx.append(i)
                z_positions = step_zdata[idx]
            
            else: # gather all particles
                zpart = np.array(step_data['z'])
                z_positions = [item for item in zpart if (self._zclose < item < self._zfar)]


            print("Result: {},  Desired:  {}".format(np.around(np.mean(z_positions), decimals=3), np.around((self._zfar + self._zclose) / 2, decimals=3)))
            if (np.around(np.mean(z_positions), decimals=3) == (np.around(((self._zfar - self._zclose) / 2) + self._zclose, decimals=3))):
                # If the mean z position of the particles within one wavelength is the same as the exact center (rounded to 3 decimals)
                # a bunch has been found

                self._bunch_found = True
                
                step_zdata = np.array(step_data['z'])

                bunchparticles_indices = np.where(np.logical_and(step_zdata>(self._zclose), step_zdata<(self._zfar)))

                bunch_particles_x = np.array(step_data['x'])[bunchparticles_indices]
                bunch_particles_y = np.array(step_data['y'])[bunchparticles_indices]
                bunch_particles_z = np.array(step_data['z'])[bunchparticles_indices]
                bunch_particles_vx = np.array(step_data['vx'])[bunchparticles_indices]
                bunch_particles_vy = np.array(step_data['vy'])[bunchparticles_indices]
                bunch_particles_vz = np.array(step_data['vz'])[bunchparticles_indices]
                bunch_particles_xp = bunch_particles_vx / bunch_particles_vz
                bunch_particles_yp = bunch_particles_vy / bunch_particles_vz

                bunch_particles_m = np.array(step_data['m'])[bunchparticles_indices]
                bunch_particles_q = np.array(step_data['q'])[bunchparticles_indices]

                bunch_particles = {}
                for key in species_dict.keys(): # label them by species
                    bunch_particles[key] = {}

                    idx = []
                    step_mdata = np.array(step_data['m'])
                    step_qdata = np.array(step_data['q'])
                    desired_mass = species_dict[key][0]
                    desired_charge = species_dict[key][1]
                    for i in range(0, len(bunchparticles_indices[0])):
                        if bunch_particles_m[i] == desired_mass:
                            if bunch_particles_q[i] == desired_charge:
                                idx.append(i)

                    bunch_particles[key]['x'] = bunch_particles_x[idx]
                    bunch_particles[key]['y'] = bunch_particles_y[idx]
                    bunch_particles[key]['z'] = bunch_particles_z[idx]

                    bunch_particles[key]['vx'] = bunch_particles_vx[idx]
                    bunch_particles[key]['vy'] = bunch_particles_vy[idx]
                    bunch_particles[key]['vz'] = bunch_particles_vz[idx]
                    bunch_particles[key]['xp'] = bunch_particles_xp[idx]
                    bunch_particles[key]['yp'] = bunch_particles_yp[idx]

                i = 0
                while os.path.exists("bunch_particles.%s.dump" % i):
                    i += 1

                pickle.dump(bunch_particles, open("bunch_particles.%s.dump" % i, "wb"))

                print("Bunch found.")

                if plot_bunch_name != None: # plot the desired species
                    step_zdata = np.array(step_data['z'])
                    step_mdata = np.array(step_data['m'])
                    step_qdata = np.array(step_data['q'])

                    key = list(species_dict.keys())[0]
                    desired_mass = species_dict[plot_bunch_name][0]
                    desired_charge = species_dict[plot_bunch_name][1]
                    idx = []
                    for i in range(0, len(step_zdata)):
                        if step_mdata[i] != desired_mass:
                            continue
                        if step_qdata[i] != desired_charge:
                            continue
                                
                        idx.append(i)

                    self.plot_bunch(step_zdata[idx], np.array(step_data['x'])[idx], velocity_data=np.array(step_data['vz'])[idx], mass=desired_mass)

    def test_cluster_detection(self, min_cluster_size, ion, z, vz):
        # only plots bunch data with clusters highlighted, and main bunch shown to test if the
        # cluster detection algorithm is working and if the user parameter "min_cluster_size" is
        # appropriate
        z_beta_rel = vz / clight
        z_gamma_rel = 1.0 / np.array(np.sqrt(1.0 - z_beta_rel ** 2.0))
        z_ekin = (z_gamma_rel - 1.0) * ion.mass_mev() * 1e3 # Per particle

        wavelength = np.mean(vz) / self._rfq_freq
        z_phase = 360 * (z - np.mean(z)) / wavelength

        data = np.array(list(zip(z_phase, z_ekin)))

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)

        clusterer.fit(data)
        hdb_labels = clusterer.labels_
        hdb_unique_labels = set(clusterer.labels_)
        print("Num clusters found: {}".format(np.max(hdb_labels)+1))
        hdb_colors = plt.cm.Spectral(np.linspace(0, 1, len(hdb_unique_labels)))
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        escale, eunits = 1, 'MeV'
        ticks_e = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*escale))
        ax1.yaxis.set_major_formatter(ticks_e)
        ax1.set_ylabel("E " + '(' + eunits + ')')
        ax2.yaxis.set_major_formatter(ticks_e)
        ax2.set_ylabel("E " + '(' + eunits + ')')

        zscale, zunits = 1, 'deg'
        ticks_z = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/zscale))
        ax1.xaxis.set_major_formatter(ticks_z)
        ax1.set_xlabel("Phase " + '(' + zunits + ')')
        ax2.xaxis.set_major_formatter(ticks_z)
        ax2.set_xlabel("Phase " + '(' + zunits + ')')

        labels_no_noise = hdb_labels[np.where(hdb_labels >= 0)]
        biggest_cluster = stats.mode(labels_no_noise)[0][0]

        for k, col in zip(hdb_unique_labels, hdb_colors):
            if k == -1:
                # Black used for noise.
                col = 'k'

            ax1.plot(data[hdb_labels == k, 0], data[hdb_labels == k, 1], '.', markerfacecolor=col, markersize=3)

            if k == biggest_cluster:
                ax2.plot(data[hdb_labels == k, 0], data[hdb_labels == k, 1], '.', markerfacecolor=col, markersize=3)

        ax1.set_title('E by Phase -- Clusters found: {}.'.format(np.max(hdb_labels)+1))
        ax2.set_title('E by Phase -- Only bunch')
        plt.show()


    # Finds clusters and separates low energy particles from high energy
    def find_clusters(self, ion, x, y, z, vx, vy, vz, min_cluster_size=20):
        z_beta_rel = vz / clight
        z_gamma_rel = 1.0 / np.array(np.sqrt(1.0 - z_beta_rel ** 2.0))
        z_ekin = (z_gamma_rel - 1.0) * ion.mass_mev() * 1e3 # Per particle

        wavelength = np.mean(vz) / self._rfq_freq
        z_phase = 360 * (z - np.mean(z)) / wavelength

        data = np.array(list(zip(z_phase, z_ekin)))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)

        clusterer.fit(data)
        hdb_labels = clusterer.labels_
        hdb_unique_labels = set(clusterer.labels_)
        print("Num clusters found: {}".format(np.max(hdb_labels)+1))
  
        labels_no_noise = hdb_labels[np.where(hdb_labels >= 0)]
        # TODO #
        biggest_cluster = stats.mode(labels_no_noise)[0][0]  # assumes the main bunch is the cluster with the most 
                                                             # particles. Apply better restriction later???

        cluster_indices = np.array(list(range(0, len(hdb_labels))))[hdb_labels == biggest_cluster]
        non_cluster_indices = np.array(list(range(0, len(hdb_labels))))[hdb_labels != biggest_cluster]

        return cluster_indices, non_cluster_indices

    def populate_distribution(self, ion, x, y, z, vx, vy, vz):
        # creates and calculates emittances for the bunch
        self._particledistribution = BunchParticleDistribution(self._rfq_freq, ion=ion, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
        self._distribution_data = self._particledistribution.calculate_emittances()

    def emittancePlots(self, species_name, auto_bunch_detection=False, min_cluster_size=20, test_cluster=True):
        # plots the X-X', Y-Y', E by Phase, and a table of emittances and particle #'s
        # species_name: string indicating desired species for emittance data
        # auto_bunch_detection: automatically find and extract clustered particles
        # min_cluster_size: user parameter for cluster finding algorithm
        # test_cluster: run the cluster finding algorithm first to check if it works well with user parameter

        filename = self._bunch_filename
        if (self._bunch_filename == None):
            if(filename == None):
                fd = FileDialog()
                filename = fd.get_filename()
        else:
            filename = self._bunch_filename

        pickle_in = open(filename, "rb")
        bunch_data = pickle.load(pickle_in)    

        bunch_species = IonSpecies(name=species_name, energy_mev=1)
        velocity_beta = np.mean((bunch_data[species_name]['vz']) / const.value("speed of light in vacuum"))
        bunch_species.calculate_from_velocity_beta(beta=velocity_beta)

        cluster_idx = np.array(list(range(0, len(bunch_data[species_name]['x']))))
        non_cluster_idx = []

        if(auto_bunch_detection): 
            print("=======================================================")
            print("If automatic bunch detection is ON, it is HIGHLY recommended to first run with test_cluster=True to check how well the algorithm worked, and to tweak the cluster detection parameter 'min_cluster_size' if necessary.")
            print("=======================================================")
            if test_cluster:
                self.test_cluster_detection(50, bunch_species, z=bunch_data['H2_1+']['z'], vz=bunch_data['H2_1+']['vz'])

            cluster_idx, non_cluster_idx = self.find_clusters(bunch_species,
                                          x=bunch_data[species_name]['x'],
                                          y=bunch_data[species_name]['y'],
                                          z=bunch_data[species_name]['z'],
                                          vx=bunch_data[species_name]['vx'],
                                          vy=bunch_data[species_name]['vy'],
                                          vz=bunch_data[species_name]['vz'],
                                          min_cluster_size=min_cluster_size)


        self.populate_distribution(bunch_species,
                              x=bunch_data[species_name]['x'][cluster_idx],
                              y=bunch_data[species_name]['y'][cluster_idx],
                              z=bunch_data[species_name]['z'][cluster_idx],
                              vx=bunch_data[species_name]['vx'][cluster_idx],
                              vy=bunch_data[species_name]['vy'][cluster_idx],
                              vz=bunch_data[species_name]['vz'][cluster_idx])

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
        zscale, zunits = 1, 'deg'
        ticks_z = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/zscale))
        ax3.xaxis.set_major_formatter(ticks_z)
        ax3.set_xlabel("Phase " + '(' + zunits + ')')


        energy_array = 0.5 * 3.34615e-27 * (np.square(bunch_data[species_name]['vz']))
        mean_energy_all = np.mean(energy_array)
        mean_energy_bunch = np.mean(energy_array[cluster_idx])

        mean_vz = np.mean(bunch_data[species_name]['vz'])
        wavelength = mean_vz / self._rfq_freq
        z_array = bunch_data[species_name]['z'] - np.mean(bunch_data[species_name]['z'])
        z_phase_array = z_array * 2 * 180 / wavelength 

        # Plot the non-cluster data points in black
        ax1.plot(bunch_data[species_name]['x'][non_cluster_idx], bunch_data[species_name]['xp'][non_cluster_idx], 'bo', c='black', ms=0.3, label='Non Bunch')
        ax2.plot(bunch_data[species_name]['y'][non_cluster_idx], bunch_data[species_name]['yp'][non_cluster_idx], 'bo', c='black', ms=0.3)
        ax3.plot(z_phase_array[non_cluster_idx], energy_array[non_cluster_idx], 'bo', c='black', ms=0.3, label='Non Bunch')
        
        # Plot the cluster data points in red
        ax1.plot(bunch_data[species_name]['x'][cluster_idx], bunch_data[species_name]['xp'][cluster_idx], 'bo', c='red', ms=0.3, label='Bunch')
        ax2.plot(bunch_data[species_name]['y'][cluster_idx], bunch_data[species_name]['yp'][cluster_idx], 'bo', c='red', ms=0.3)
        ax3.plot(z_phase_array[cluster_idx], energy_array[cluster_idx], 'bo', c='red', ms=0.3, label='Bunch')
        
        if (auto_bunch_detection):
            ax1.legend(loc='upper left', markerscale=15)


        table_array = [[species_name, ''],
                       ['Total # particles', len(bunch_data[species_name]['x'])],
                       ['Bunch # particles', '{} ({:.1f}%)'.format(len(cluster_idx), 100 * len(cluster_idx) / len(bunch_data[species_name]['x']))],
                       ['Total mean energy', '{:.3f} keV'.format(mean_energy_all*6.242e18 / 1e3)],
                       ['Bunch mean energy', '{:.3f} keV'.format(mean_energy_bunch*6.242e18 / 1e3)],
                       ['',''],
                       ["X-X' norm. 1 rms emittance",'{:.3f} {}'.format(self._distribution_data['data'][18][1], self._distribution_data['data'][18][2])],
                       ["Y-Y' norm. 1 rms emittance",'{:.3f} {}'.format(self._distribution_data['data'][19][1], self._distribution_data['data'][19][2])],
                       ["X-Y' norm. 1 rms emittance",'{:.3f} {}'.format(self._distribution_data['data'][20][1], self._distribution_data['data'][20][2])],
                       ["Y-X' norm. 1 rms emittance",'{:.3f} {}'.format(self._distribution_data['data'][21][1], self._distribution_data['data'][21][2])],
                       ["Logitudinal 1 rms emittance", '{:.3f} {}'.format(self._distribution_data['data'][17][1] * 1e3, 'KeV-deg')]]

        table = ax4.table(table_array, loc='best')
        # table.auto_set_font_size(True)
        table.scale(1, 1.5)

        plt.show() 


    def make_3d_bunch_plot(self, filename=None, make_energy_plot=True):
        # make a 3d plot of the bunch particles
        # DEPRECATED, LEAVING IN FOR REFERENCE AND LEGACY(?)

        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d import proj3d

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

        ### Scaling
        x_scale=3
        y_scale=1
        z_scale=1

        scale=np.diag([x_scale, y_scale, z_scale, 1.0])
        scale=scale*(1.0/scale.max())
        scale[3,3]=1.0

        def short_proj():
          return np.dot(Axes3D.get_proj(ax), scale)

        ax.get_proj=short_proj
        ### Scaling

        if (make_energy_plot):
            global DIHYDROGENMASS

            # Non relativistic and hardcoded for h2+
            # TODO
            c = 0.5 * DIHYDROGENMASS * (np.square(np.array(bunch_data['H2_1+']['vz'])) * 6.242e15)
            plot = ax.scatter(bunch_data['H2_1+']['z'], bunch_data['H2_1+']['x'], bunch_data['H2_1+']['y'], s=1, c=c, cmap='plasma')
            
            clb = fig.colorbar(plot, ax=ax)
            clb.set_label('Energy (keV)')
        else:
            plot = ax.scatter(bunch_data['H2_1+']['z'], bunch_data['H2_1+']['x'], bunch_data['H2_1+']['y'], s=1)

        plt.show()

def main():
    postprocessor = BunchPostProcessor(32.8e6)

    postprocessor.find_bunch(20000, 18011, 1.4, velocity_min_sample=75000, step_interval=10, plot_bunch_name='H2_1+')
    # postprocessor.emittancePlots('H2_1+', auto_bunch_detection=True, min_cluster_size=50, test_cluster=True)
    # postprocessor.make_3d_bunch_plot(make_energy_plot=False)

if __name__ == '__main__':
    main()
