# Py RFQ Helper

### Usage

#### Creating the RFQ

* Create the RFQ object
    `rfq = PyRFQ(filename=filename, from_cells=False, twoterm=True, boundarymethod=False)`
  * Filename refers to field filename which takes the following format
    `x, y, z, ex, ey, ez`

* Set the following parameters
```python
   rfq.simple_rods    = True  # simple rods
   rfq.vane_radius     = None  # radius of simple rods
   rfq.vane_distance  = None  # vane center distance from central axis
   rfq.rf_freq        = None  # RF Frequency
   rfq.zstart         = 0.0   # start of the RFQ
   rfq.sim_end_buffer = 0.0   # added distance to the end of the RFQ beyond vanes
   rfq.resolution     = 0.002 # in meters
```
  * Optional parameters
     ```python
     rfq.endplates                     = False
     rfq.endplates_outer_diameter      = 0.2 # in meters
     rfq.endplates_inner_diameter      = 0.1 
     rfq.endplates_distance_from_vanes = 0.1
     rfq.endplates_thickness           = 0.1
     ```
* Following variables are also available 
   ```python 
   rfq._conductors  # WARP conductors to construct the vanes
   rfq._field       # Field object
   rfq._sim_end     # End of the RFQ simulation
   rfq._length      # length of the simulation
   rfq._fieldzmax   # Z maximum of the field
   ```

### RFQ Setup and Installation into Simulation

* Setup RFQ
  `rfq.setup()`

* Install the RFQ
  `rfq.install(field_scale_factor=1)`

  * Allows for scaling factor to be applied to the imported field

### Diagnostics and Utils

#### Utils class
* Create a PyRfqUtils object, passing it the RFQ object and a list of the WARP beam species

  `utils = PyRfqUtils(rfq, [h2_beam, proton_beam]` 

##### Particle saving
* At any time, the particles within the simulation at a particular step can be saved into an hdf5 data output file
  * For serial:
    `utils.write_hdf5_data(step_number, beamlist=None)`
  * Parallel:
    `utils.write_hdf5_data_p(step_number, beamlist=None)`

  Beamlist is a list of the species what are to be saved.


##### Bunch Finding
* The helper can automatically isolate and gather data of a bunch right outside of the RFQ
  * During the simulation:
    * Serial
      `utils.find_bunch(h2_beam, max_steps)`
    * Parallel
      `utils.find_bunch_p(h2_beam, max_steps=10000)`
  * Using a saved hdf5 dataset
    * Create a `BunchPostProcessor()` object from bunchpostprocessing.py, passing it the RF frequency
    * Call
      `postprocessor.find_bunch(max steps, starting step, end of the rfq, minimum number of velocity samples, step data gathering interval, name of species)`

##### RFQ Visualization
* During the simulation, a real time plot can be shown of the rms of the beam 
  * Setup PyQtGraph
    `app = pg.mkQApp()`
  * Setup the rms plot
    `utils.rms_plot_setup(title="X and Y RMS (twice rms) vs Z", labels={'left':('X, Y', 'm'), 'bottom':('Z', 'm')}, xrange=[,], yrange=[,])`
  * Call every step
    `utils.plot_rms()`
* Something similar can be done with the particle positions as well, but it is not recommended as it causes heavy slowdown
  * Setup PyQtGraph as above
  * Setup the particle plots
    `utils.particle_plot_setup(title='X and Y Particles vs Z', labels={'left':('X, Y', 'm'), 'bottom':('Z', 'm')})`
  * Call every step
    `utils.plot_particles()`

* A video of the RFQ can also be made with the hdf5 particle data output
  * Settings to produce a video and descriptions can be found in PyRFQBeamVideoSettings.txt
  * Run particle_videos.py
    `python particle_videos.py`
  * The program will gather the settings from the settings file and output an mp4 titled with the date and time
  * This may take some time


##### Emittances and Phase portraits

* Once there exists a bunch data output file, either created during the simulation or after, one can plot the phase portraits X-X', Y-Y', and the Energy by Phase plot using the BunchPostProcessor class.
  * Produce the emittance plots
    `postprocessor.emittancePlots('speciesname', auto_bunch_detection=True, min_cluster_size=50, test_cluster=True)`
  * Species name refers to the species that should have the emittances calculated for it
  * If `auto_bunch_detection == True`, then the code will use HDBSCAN to automatically extract the clustered particles from the low energy particles. `min_cluster_size` is the user parameter for the algorithm. 
  * If `auto_bunch_detection` is on, then it is HIGHLY recommended to also set test_cluster to true, as it will show you a plot of the clusters found, and which one it has identified to be the bunch. By doing so, the user can tweak the min_cluster_size to create a more fitting extraction.

### APPENDIX and details

#### HDF5 Output format
~~~text
f = {    
         'SpeciesList':    {
                            'H2_1+': [mass, charge, charge_state, A, Z]
                            'P_1+': [mass, charge, charge_state, A, Z]
                            ...
                           }            
         'Step#1':         {
                            'ENERGY': [energy]
                            'm'     : [mass]
                            'px'    : [x momenta]
                            'py'    : [y momenta]
                            'pz'    : [z momenta]
                            'q'     : [charge]
                            'vx'    : [x velocities]
                            'vy'    : [y velocities]
                            'vz'    : [z velocities]
                            'x'     : [x positions]
                            'y'     : [y positions]
                            'z'     : [z positions]
                           }   
         'Step#(num)':     ...
    }
~~~

#### Bunch Particle Pickle format
~~~text
pickle = {
            'H2_1+': {
                      'gaminv': [gamma inverse]
                      'r': [r positions]
                      'rp': [r transverse normalized velocity]
                      'theta': [theta positions]
                      'ux': [x momenta/mass]
                      'uy': [y momenta/mass]
                      'uz': [z momenta/mass]
                      'vx': [x velocities]
                      'vy': [y velocities]
                      'vz': [z velocities]
                      'x': [x positions] 
                      'xp': [x transverse normalized velocities]
                      'y': [y positions]
                      'yp': [y transverse normalized velocities]
                      'z': [z positions]
                     }
            'P_1+': {
                        ...
                    }
            ...
         }
~~~

#### Particle Videos catalogued data structure
~~~test
data_dict = {
                'SpeciesList': ['H2_1+', 'P_1+', ...]
                'Step#1':  {
                                'H2_1+': {
                                            'x': [x positions]
                                            'y': [y positions]
                                            'z': [z positions]
                                         }
                                'P_1+':  {
                                            'x': [x positions]
                                            'y': [y positions]
                                            'z': [z positions]
                                         }
                                ...
                           }
                'Step#...': {
                                ...
                            }
            }
~~~