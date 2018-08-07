from warp import *
import numpy as np

global _lwithnewconductorgeneration
_lwithnewconductorgeneration = True


class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def sub(self, v):
        return Vec3(self.x - v.x,
                    self.y - v.y,
                    self.z - v.z)

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z
    
    def cross(self, v):
        return Vec3(self.y * v.z - self.z * v.y,
                    self.z * v.x - self.x * v.z,
                    self.x * v.y - self.y * v.x)
    
    def length(self):
        return np.sqrt(self.x * self.x +
                       self.y * self.y +
                       self.z * self.z)
    
    def normalize(self):
        l = self.length()
        return Vec3(self.x / l, self.y / l, self.z / l)
    
    
class Ray:
    def __init__(self, orig=None, direction=None):
        self.orig = orig
        self.direction = direction
        
        
def ray_triangle_intersect(r, v0, v1, v2):
    v0v1 = v1.sub(v0)
    v0v2 = v2.sub(v0)
    pvec = r.direction.cross(v0v2)
    
    det = v0v1.dot(pvec)
    
    if det < 0.000001:
        return float('-inf')
    
    invDet = 1.0 / det
    tvec = r.orig.sub(v0)
    u = tvec.dot(pvec) * invDet
    
    if u < 0 or u > 1:
        return float('-inf')
    
    qvec = tvec.cross(v0v1)
    v = r.direction.dot(qvec) * invDet
    
    if v < 0 or u + v > 1:
        return float('-inf')
    
    return v0v2.dot(qvec) * invDet

                                                                                                                                                                                                            
class WARPRFQConductor(Assembly):
    """
    Conductor defined through a gmsh mesh, using the BEMPP format
    which in turn uses dune-grid (https://www.dune-project.org/).
    - mesh: gmsh mesh (either from file or bempp generated) TODO: fromfile
    - voltage=0: conductor voltage
    - xcent=0.,ycent=0.,zcent=0.: origin of CAD object
    - condid='next': conductor id, must be integer, or can be 'next' in
                     which case a unique ID is chosen
    """
    def __init__(self, mesh,
                 voltage=0.,
                 xcent=0.,ycent=0.,zcent=0.,
                 condid='next',**kw):
        
        assert _lwithnewconductorgeneration,\
          'WARPRFQconductor can only be used with the new conductor generation method'
        kwlist = []
        Assembly.__init__(self,voltage,xcent,ycent,zcent,condid,kwlist,
                          self.conductorf,self.conductord,self.intercept,
                          self.conductorfnew,
                          kw=kw)

        # --- Store the mesh that holds the vertex, line and triangle data --- #
        self._mesh = mesh

        _verts = self._mesh.leaf_view.vertices
        _surfs = self._mesh.leaf_view.elements[:,:10].T

        print(_surfs)
        print()
        print(_verts.shape)
        print()
        
        for _surf in _surfs:
            print(_verts[:, _surf])
        
        # --- The extent is not used with the new method, but define it anyway.
        self.createextent([-largepos,-largepos,-largepos],
                          [+largepos,+largepos,+largepos])

    def plot_gmsh(self):
        self._mesh.plot()
        
    def conductorf(self):
        raise Exception('This function should never be called')

    def conductord(self,xcent,ycent,zcent,n,x,y,z,distance):
       
        if xcent != 0.: x = x - xcent
        if ycent != 0.: y = y - ycent
        if zcent != 0.: z = z - zcent

        #distance[:] = None

    def intercept(self,xcent,ycent,zcent,
                  n,x,y,z,vx,vy,vz,xi,yi,zi,itheta,iphi):
        raise Exception('CADconductor intercept not yet implemented')

    def conductorfnew(self,xcent,ycent,zcent,intercepts,fuzz):

        _xmin = intercepts.xmmin + xcent
        _ymin = intercepts.ymmin + ycent
        _zmin = intercepts.zmmin + zcent

        _dx = intercepts.dx
        _dy = intercepts.dy
        _dz = intercepts.dz

        _nx = intercepts.nx
        _ny = intercepts.ny
        _nz = intercepts.nz
        
        _grid = np.meshgrid(np.linspace(_xmin, _xmin + (_nx - 1) * _dx, _nx, endpoint=True),
                            np.linspace(_ymin, _ymin + (_ny - 1) * _dy, _ny, endpoint=True),
                            np.linspace(_zmin, _zmin + (_nz - 1) * _dz, _nz, endpoint=True),
                            indexing="ij")  # TODO: Is this the right indexing?


        _grid_x, _grid_y, _grid_z = _grid

        del _grid

        _grid_x = _grid_x.flatten()
        _grid_y = _grid_y.flatten()
        _grid_z = _grid_z.flatten()

        _verts = self._mesh.leaf_view.vertices
        _surfs = self._mesh.leaf_view.elements

        """
        xi,yi,zi = CADmodule.CADgetintercepts(filename,
                                              intercepts.xmmin+xcent,
                                              intercepts.ymmin+ycent,
                                              intercepts.zmmin+zcent,
                                              intercepts.dx,
                                              intercepts.dy,
                                              intercepts.dz,
                                              intercepts.nx,
                                              intercepts.ny,
                                              intercepts.nz)
        """
        intercepts.nxicpt = xi.shape[0]
        intercepts.nyicpt = yi.shape[0]
        intercepts.nzicpt = zi.shape[0]
        intercepts.gchange()
        intercepts.xintercepts[...] = xi
        intercepts.yintercepts[...] = yi
        intercepts.zintercepts[...] = zi

if __name__ == "__main__":
    wrc = WARPRFQConductor(mesh=None)
