from warp import *
from py_rfq_helper import *
import time

#FILENAME  = "input/vecc_rfq_004_py.dat"
#FILENAME  = "input/PARMTEQOUT.TXT"
FILENAME  = "input/Parm_50_63cells.dat"
#FILENAME  = "input/fieldoutput.txt"
#FILENAME   = "input/fieldw015width.dat"

VANE_RAD  = 2 * cm
VANE_DIST = 11 * cm

NX     = 16
NY     = 16
NZ     = 512
PRWALL = 0.2
D_T    = 1e-9
RF_FREQ = 32.8e6
Z_START = 0.0 #the start of the rfq
SIM_START = -0.15
setup()

w3d.solvergeom = w3d.XYZgeom

w3d.xmmax =  PRWALL
w3d.xmmin = -PRWALL
w3d.nx    =  NX

w3d.ymmax =  PRWALL
w3d.ymmin = -PRWALL
w3d.ny    =  NY

w3d.zmmax =  1.456 + 0.2
w3d.zmmin =  SIM_START
w3d.nz    =  NZ

w3d.bound0   = neumann
w3d.boundnz  = neumann
w3d.boundxy  = neumann
# ---   for particles
top.pbound0  = absorb
top.pboundnz = absorb
top.prwall   = PRWALL

top.dt = D_T

refinedsolver = MRBlock3D()
registersolver(refinedsolver)
#solver = MultiGrid3D()
#registersolver(solver)

top.npinject = 50
top.inject   = 1
w3d.l_inj_rz = False
top.zinject  = SIM_START 
w3d.zmmin = SIM_START
top.injctspc = 1000000




##########################################
# RFQ creation and initialization of parameters

rfq = RFQ(filename=FILENAME, from_cells=True, twoterm=True)
rfq.vane_radius       = VANE_RAD
rfq.vane_distance     = VANE_DIST
rfq.zstart            = Z_START
rfq.rf_freq           = RF_FREQ
rfq.sim_start         = SIM_START
rfq.sim_end_buffer    = 0.2
rfq.resolution        = 0.002

rfq.xy_limits         = [-0.015, 0.015, -0.015, 0.015]
rfq.tt_voltage        = 50.0e3
rfq.tt_a_init         = 0.038802
rfq.setup()

rfq.add_cell(cell_type="TCS",

            aperture=0.011255045027294745,
            modulation=1.6686390559337798,
            length=0.0427)
# 0.10972618296477678

rfq.add_cell(cell_type="DCS",
            aperture=0.015017826368066015,
            modulation=1.0,
            length=0.13)



rfq.install()

############################################



childmesh = refinedsolver.addchild(mins=[rfq._field._xmin, rfq._field._ymin, SIM_START], 
                             maxs=[rfq._field._xmax, rfq._field._ymax, w3d.zmmax],
                             refinement=[2,2,2])

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

winon(1, suffix='YZ')
winon(2, suffix="X'X")
winon(3, suffix="Y'Y")
winon()

def plotXZparticles(view=1):

    plsys(view)

    plg([-PRWALL,PRWALL],[0,0], color=red)
    plg([-PRWALL,PRWALL],[rfq._field._zmax, rfq._field._zmax], color=red)

    rfq._conductors.draw()
    # pfzx(plotsg=0, cond=0, titles=False, view=view)
    ppzx(titles=False, view=view)
    limits(w3d.zmminglobal, w3d.zmmaxglobal)
    ptitles("", "Z (m)", "X (m)")

def plotYZparticles(view=1):
    plsys(view)

    plg([-PRWALL,PRWALL],[0,0], color=red)
    plg([-PRWALL,PRWALL],[rfq._field._zmax, rfq._field._zmax], color=red)
    
    rfq._conductors.draw()
    # pfzy(plotsg=0, cond=0, titles=False, view=view)
    ppzy(titles=False, view=view)
    limits(w3d.zmminglobal, w3d.zmmaxglobal)
    ptitles("", "Z (m)", "Y (m)")

def plotXphase(view=1):
    plsys(view)
    beam.ppxxp()

def plotYphase(view=1):
    plsys(view)
    beam.ppyyp()

def beamplots():
    window(0)
    fma()
    plotXZparticles()
    refresh()

    window(1)
    fma()
    plotYZparticles()
    refresh()

    window(2)
    fma()
    plotXphase()
    refresh()

    window(3)
    fma()
    plotYphase()
    refresh()

@callfromafterstep
def makeplots():
    if top.it%1 == 0:
        beamplots()
        # window(0)
        # rfq.plot_efield()
        # refresh()
        

starttime = time.time()

step(1000)
hcp()

endtime = time.time()

print("Elapsed time for simulation: {}".format(endtime-starttime))

part_x = beam.getx()
part_y = beam.gety()
part_z = beam.getz()

print(len(part_x))

with open("particleoutput.dump", 'w') as outfile:
    outfile.write("x, y, z\n")
    for x, y, z in zip(part_x, part_y, part_z):
        outfile.write("{:.4e}   {:.4e}   {:.4e}\n".format(x, y, z))



# import matplotlib.pyplot as plt
# print(rfq._ray)
# plt.plot(rfq._ray)
# plt.show()