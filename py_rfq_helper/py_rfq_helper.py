from warp import *
from field import *

class RFQ(object):
    def __init__(self,
                 filename=None,
                 vane_radius=0.0,
                 vane_distance=0.0,
                 zstart=0.0,
                 rf_freq=0.0):
        # RFQ Object
        # Takes in a filename with Cell parameters or field
        # Vane radius set to some value if using simple rod approximation
        # Vane distance
        # Start position of the RFQ

        self._filename      = filename
        self._field         = FieldLoader()
        self._vane_radius   = vane_radius
        self._vane_distance = vane_distance
        self._zstart        = zstart
        self._conductors    = None
        self._rf_freq       = rf_freq
        self._ray = []

        self.setup()

    def setup(self):
        # Parameters: None
        #self._field.load_field_from_file(self._filename)
        self._field.load_field_from_cells(self._filename)
        self.import_field()
        self.create_vanes()



    def import_field(self):
        # import_field
        # Parameters: none
        # Returns: none
        # Loads the appropriate field into Warp simulation.

        def fieldscaling(time):
            val = np.sin(time * 2 * np.pi * self._rf_freq)
            self._ray.append(val)
            return val

        egrd = addnewegrddataset(ex=self._field._ex, 
                         ey=self._field._ey,
                         ez=self._field._ez,
                         dx=self._field._dx,
                         dy=self._field._dz,
                         zlength=self._field._z_length)

        addnewegrd(id=egrd, zs=0, ze=self._field._z_length, func=fieldscaling)

    def create_vanes(self):
        # create_vanes
        # Parameters: None
        # Returns: None
        # Creates the conducting objects in the warp simulation.
        # Vanes and outer tube.

        length = self._field._z_length
        zcent  = (self._field._z_length / 2.0) + abs(self._zstart)

        outer_shell = ZCylinderOut(self._vane_distance + 0.05, length + abs(self._zstart), zcent=(length + self._zstart)/2)

        rod1 = ZCylinder(self._vane_radius, length, zcent=zcent, xcent=self._vane_distance)
        rod2 = ZCylinder(self._vane_radius, length, zcent=zcent, xcent=-self._vane_distance) 
        rod3 = ZCylinder(self._vane_radius, length, zcent=zcent, ycent=self._vane_distance)
        rod4 = ZCylinder(self._vane_radius, length, zcent=zcent, ycent=-self._vane_distance)
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

        plotegrd(component="x", iz=self._field._nz, iy=self._field._ny)
        fma()

        plotegrd(component="y", iz=self._field._nz, ix=self._field._nx)
        fma()