"""
RMS:
  pzxrms: Plots RMS X versus Z
  pzyrms: Plots RMS Y versus Z
  pzzrms: Plots RMS Z versus Z
  pzrrms: Plots RMS R versus Z
  pzxprms: Plots RMS X' versus Z
  pzyprms: Plots RMS Y' versus Z
  pzvxrms: Plots true RMS Vx versus Z
  pzvyrms: Plots true RMS Vy versus Z
  pzvzrms: Plots true RMS Vz versus Z
  pzepsx: Plots X-X' emittance versus Z
  pzepsy: Plots Y-Y' emittance versus Z
  pzepsz: Plots Z-Z' emittance versus Z
  pzepsr: Plots R-R' emittance versus Z
  pzepsnx: Plots X-X' normalized emittance versus Z
  pzepsny: Plots Y-Y' normalized emittance versus Z
  pzepsnz: Plots Z-Z' normalized emittance versus Z
  pzepsnr: Plots R-R' normalized emittance versus Z
  pzepsg: Plots generalized emittance versus Z
  pzepsh: Plots generalized emittance versus Z
  pzepsng: Plots generalized normalized emittance versus Z
  pzepsnh: Plots generalized normalized emittance versus Z
  pzxxpslope: Plots slope of x-x' phase space versus Z
  pzyypslope: Plots slope of y-y' phase space versus Z
Envelope:
  pzenvx: Plots beam X envelope (twice Xrms) versus Z
  pzenvy: Plots beam Y envelope (twice Yrms) versus Z
  pzxedge: Plots beam X envelope (twice Xrms) versus Z
  pzxpedge: Plots beam X' envelope versus Z
  pzyedge: Plots beam Y envelope (twice Yrms) versus Z
  pzypedge: Plots beam Y' envelope versus Z
  pzredge: Plots beam R envelope (root 2 Rrms) versus Z
  pzxedges: Plots beam X edges (centroid +- twice Xrms) versus Z
  pzyedges: Plots beam Y edges (centroid +- twice Yrms) versus Z
  pzredges: Plots beam R edges (+- root 2 Rrms) versus Z
  pzenvxp: Plots beam X' envelope (2*xxpbar/xrms) versus Z
  pzenvyp: Plots beam Y' envelope (2*yypbar/yrms) versus Z
Means:
  pzxbar: Plots mean X coordinate versus Z
  pzybar: Plots mean Y coordinate versus Z
  pzzbar: Plots mean axial location versus Z
  pzxpbar: Plots mean X' versus Z
  pzypbar: Plots mean Y' versus Z
  pzvxbar: Plots mean Vx versus Z
  pzvybar: Plots mean Vy versus Z
  pzvzbar: Plots mean Vz versus Z
  pzxybar: Plots mean product of X  and Y  versus Z
  pzxypbar: Plots mean product of X  and Y' versus Z
  pzyxpbar: Plots mean product of Y  and X' versus Z
  pzxpypbar: Plots mean product of X' and Y' versus Z
  pzxvybar: Plots mean product of X  and Vy versus Z
  pzyvxbar: Plots mean product of Y  and Vx versus Z
  pzvxvybar: Plots mean product of Vx and Vy versus Z
  pzxsqbar: Plots mean X-squared versus Z
  pzysqbar: Plots mean Y-squared versus Z
  pzzsqbar: Plots mean Z-squared versus Z
  pzxpsqbar: Plots mean X' squared versus Z
  pzypsqbar: Plots mean Y' squared versus Z
  pzvxsqbar: Plots mean Vx squared versus Z
  pzvysqbar: Plots mean Vy squared versus Z
  pzvzsqbar: Plots mean Vz squared versus Z
  pzxxpbar: Plots mean product of X and X' versus Z
  pzyypbar: Plots mean product of Y and Y' versus Z
  pzxvxbar: Plots mean product of X and Vx versus Z
  pzyvybar: Plots mean product of Y and Vy versus Z
  pzzvzbar: Plots mean product of Z and Vz versus Z
  pzxvzbar: Plots mean product of X and Vz versus Z
  pzyvzbar: Plots mean product of Y and Vz versus Z
  pzvxvzbar: Plots mean product of Vx and Vz versus Z
  pzvyvzbar: Plots mean product of Vy and Vz versus Z
Miscellaneous:
  pzcurr: Plots beam current versus Z
  pzlchg: Plots line charge versus Z
  pzvzofz: Plots mean axial velocity versus Z
  pzrhomid: Plots charge dens. on axis versus Z
  pzrhomax: Plots charge dens. max-over-X,Y versus Z
  pzrhoax: Plots charge density on axis versus Z
  pzphiax: Plots electrostatic potential on axis versus Z
  pzegap: Plots gap electric field versus Z
  pzezax: Plots Z electric field on axis versus Z
  pznpsim: Plots no. of simulation particles versus Z
  pzpnum: Plots no. of physical particles versus Z
  pzppcell: Plots no. of simulation particles per cell versus Z
"""

from warp import *
import __main__


def pzplotsdoc():
    from . import pzplots
    print(pzplots.__doc__)

def setzdiagsflag(flag):
    "Turns on or off the various z diagnostics"
    w3d.lsrhoax3d = flag
    w3d.lgtlchg3d = flag
    w3d.lgetvzofz = flag
    w3d.lgetese3d = flag
    w3d.lsphiax3d = flag
    w3d.lsezax3d  = flag
    w3d.lsetcurr  = flag

###########################################################################
def extractvar(name,varsuffix=None,pkg='top',ff=None):
    """
  Helper function which, given a name, returns the appropriate data. Note that
  name could actually be the variable itself, in which case, it is just
  returned.
    """
    if isinstance(name,str):
        # --- if varsuffix is specified, try to evaluate the name with the
        # --- suffix. If ok, return the result, otherwise, default to the
        # --- fortran variable in the specified package.
        if varsuffix is not None:
            vname = name + str(varsuffix)
            try:    result = ff.read(vname)
            except: result = None
            if result is not None: return result
            try:    result = __main__.__dict__[vname]
            except: result = None
            if result is not None: return result
        try:    result = ff.read(name+'@'+pkg)
        except: result = None
        if result is not None: return result
        return getattr(packageobject(pkg),name)
    else:
        return name

def _extractvarkw(name,kw,pkg='top'):
    return _extractvar(name,kw.get('varsuffix',None),pkg=pkg)

def gettitler(js):
    if js == -1: return "All species"
    else:        return "Species %d"%js

