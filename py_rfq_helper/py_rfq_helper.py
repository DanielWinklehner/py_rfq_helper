# RFQ class. Contains parameters and guidelines to create an RFQ in an already
# existent warp simulation. Contains an instance of a FieldLoader class from field.py.
# 
# Usage: create an instance of an RFQ with the filename of either cell parameters 
#        or field data, and a boolean to indicate if the file contains the former
#        or the latter. 
#        User must also instantiate the values "vane_distance" (vane distance 
#        from axis), and rf_freq (frequency field modulation in Hz). If the bool
#        simple_rods is set, the user must instantiate the value "vane_radius" to
#        be the radius of the rod vanes. 
#        The values "zstart" (start of the rfq), "sim_start" (start of the
#        simulation, and "sim_end_buffer" (extra room beyond the rfq) can be set
#        if necessary, and are otherwise instantiated at 0.0.
#       

from warp import *
from field import *

class RFQ(object):
    def __init__(self,
                 filename=None,
                 from_cells=False,
                 ):
        # Takes in a filename with Cell parameters or field
        # Vane radius set to some value if using simple rod approximation
        # Vane distance
        # Start position of the RFQ

        # User Parameters
        #     Passed in via constructor
        self._from_cells    = from_cells
        self._filename      = filename

        #     Must be set outside of object creation
        self.simple_rods   = True
        self.vane_radius   = None
        self.vane_distance = None
        self.rf_freq       = None
        self.zstart        = 0.0
        self.sim_start     = 0.0
        self.sim_end_buffer = 0.0


        # "Private" variables
        self._conductors    = None
        self._field         = FieldLoader()
        self._sim_end       = 0.0

        # Debugging
        self._ray           = [] #debugging

    def install(self):
        # Parameters: None
        # Returns: None
        # Installs the field and conductors into Warp
        self.setup()

    def setup(self):
        # Parameters: None

        if (not self.vane_radius) and (self.simple_rods):
            print("Please specify the vane radius (vane_radius) for the simple rod structure. Exiting.")
            exit(1)
        if (not self.vane_distance):
            print("Please specify the vane distance (vane_distance) from axis. Exiting.")
            exit(1)
        if (not self.rf_freq):
            print("The RF frequency (rf_freq) must be specified. Exiting")
            exit(1)

        if self._from_cells:
            self._field.load_field_from_cells(self._filename)
        else:
            self._field.load_field_from_file(self._filename)


        self._sim_end = self._field._zmax + self.sim_end_buffer

        self.import_field()
        
        
        self.plot_efield()

        self.create_vanes()

    def import_field(self):
        # import_field
        # Parameters: none
        # Returns: none
        # Loads the appropriate field into Warp simulation.

        def fieldscaling(time):
            val = np.cos(time * 2 * np.pi * self.rf_freq)
            print(time, val)
            self._ray.append(val)
            return val

        egrd = addnewegrddataset(ex=self._field._ex, 
                         ey=self._field._ey,
                         ez=self._field._ez,
                         dx=self._field._dx,
                         dy=self._field._dy,
                         zlength=self._field._z_length)
        
        print("=====================================")
        print("xmin: " + str(self._field._xmin))
        print("ymin {}".format(self._field._ymin))
        print("nx {} ny {} nz {}".format(self._field._nx,self._field._ny,self._field._nz))
        print("dx {} dy {} dz {}".format(self._field._dx,self._field._dy,self._field._dz))
        print("======================================")
        
        addnewegrd(id=egrd, zs=0, xs=self._field._xmin, ys=self._field._ymin, ze=self._field._z_length, func=fieldscaling)


    def create_vanes(self):
        # create_vanes
        # Parameters: None
        # Returns: None
        # Creates the conducting objects in the warp simulation.
        # Vanes and outer tube.

        length = self._field._z_length
        zcent  = (self._field._z_length / 2.0) + abs(self.zstart)

        print("zmin {}  zmax {}".format(self._field._zmax, self._field._zmin))
        print("simend {} simstart {}".format(self._sim_end, self.sim_start))

        outer_shell = ZCylinderOut(self.vane_distance + 0.05, (self._sim_end - self.sim_start), zcent=(self._sim_end + self.sim_start)/2)

        rod1 = ZCylinder(self.vane_radius, length, zcent=zcent, xcent=self.vane_distance)
        rod2 = ZCylinder(self.vane_radius, length, zcent=zcent, xcent=-self.vane_distance) 
        rod3 = ZCylinder(self.vane_radius, length, zcent=zcent, ycent=self.vane_distance)
        rod4 = ZCylinder(self.vane_radius, length, zcent=zcent, ycent=-self.vane_distance)
        total_conductors = outer_shell + rod1 + rod2 + rod3 + rod4
        
        installconductor(total_conductors)
        scraper = ParticleScraper(total_conductors)

        self._conductors = total_conductors

    def plot_efield(self):
        # Plots the e field along the z axis
        # Parameters: None
        # Returns: None

        plotegrd(component="z", iy=self._field._ny, ix=self._field._nx)
        fma()

        plotegrd(component="x", ix=self._field._nx, iy=self._field._ny)
        fma()

        plotegrd(component="y", iy=self._field._ny, ix=self._field._nx)
        fma()

        
        # plotegrd(component="x", iz=50)
        # fma()
        #plotegrd(component="y", iz=50)
        #fma()
    
