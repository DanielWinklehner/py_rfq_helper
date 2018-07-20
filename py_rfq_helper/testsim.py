from warp import *
from py_rfq_helper import *

#FILENAME  = "vecc_rfq_004_py.dat"
#FILENAME  = "PARMTEQOUT.TXT"
FILENAME  = "Parm_50_63cells.dat"

VANE_RAD  = 2 * cm
VANE_DIST = 11 * cm

NX     = 16
NY     = 16
NZ     = 512
PRWALL = 0.2
D_T    = 1e-10
RF_FREQ = 3.28e7
Z_START = -0.15

setup()

w3d.solvergeom = w3d.XYZgeom

w3d.xmmax =  PRWALL
w3d.xmmin = -PRWALL
w3d.nx    =  NX

w3d.ymmax =  PRWALL
w3d.ymmin = -PRWALL
w3d.ny    =  NY

w3d.zmmax =  1.538
w3d.zmmin =  Z_START
w3d.nz    =  NZ

w3d.bound0   = dirichlet
w3d.boundnz  = neumann
w3d.boundxy  = neumann
# ---   for particles
top.pbound0  = absorb
top.pboundnz = absorb
top.prwall   = PRWALL

top.dt = D_T

solver = MultiGrid3D()
registersolver(solver)




top.npinject = 15
top.inject   = 1
# top.vinject  = 15 * kV # Only needed if top.inject != 1
w3d.l_inj_rz = False
top.zinject  = Z_START 
top.injctspc = 1000000




rfq = RFQ(filename=FILENAME, vane_radius=VANE_RAD, vane_distance=VANE_DIST, zstart=Z_START, rf_freq=RF_FREQ)
#rfq.plot_efield()

#exit(1)

beam = Species(type=Dihydrogen, charge_state=+1, name="H2+")
top.lrelativ = False
top.zbeam = 0.0
top.pgroup.nps = 0

beam.ekin = 15.*kV      # ion kinetic energy [eV] [eV]
beam.ibeam = 10 * mA  # compensated beam current [A]
beam.emitx = 1e-6  # beam x-emittance, rms edge [m-rad]
beam.emity = 1e-6  # beam y-emittance, rms edge [m-rad]
beam.vthz = 0.0  # axial velocity spread [m/sec]

# Beam centroid and envelope initial conditions
beam.x0 = 0.0  # initial x-centroid xc = <x> [m]
beam.y0 = 0.0  # initial y-centroid yc = <y> [m]
beam.xp0 = 0.0  # initial x-centroid angle xc' = <x'> = d<x>/ds [rad]
beam.yp0 = 0.0  # initial y-centroid angle yc' = <y'> = d<y>/ds [rad]
beam.a0 = 5 * mm  # initial x-envelope edge a = 2*sqrt(<(x-xc)^2>) [m]
beam.b0 = 5 * mm  # initial y-envelope edge b = 2*sqrt(<(y-yc)^2>) [m]
beam.ap0 = 0 # initial x-envelope angle ap = a' = d a/ds [rad]
beam.bp0 = 0  # initial y-envelope angle bp = b' = d b/ds [rad]

# This routine will calculate vbeam and other quantities.
derivqty()

package("w3d")
generate()


def plotparticles(view=1):
    rfq._conductors.draw()
    pfzx(plotsg=0, cond=0, titles=False, view=view)
    ppzx(titles=False, view=view)
    limits(w3d.zmminglobal, w3d.zmmaxglobal)
    ptitles("", "Z (m)", "X (m)")
    # rfq.plot_efield()


def beamplots():
    fma()
    plotparticles()
    refresh()

@callfromafterstep
def makeplots():
    if top.it%5 == 0:
        beamplots()

step(50000)
hcp()

# import matplotlib.pyplot as plt
# plt.plot(rfq._ray)
# plt.show()
