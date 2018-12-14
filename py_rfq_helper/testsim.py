from warp import *
from py_rfq_helper import *
from py_rfq_designer import *
from py_rfq_utils import *
import bisect
import time
import pprint

# FILENAME  = "input/vecc_rfq_004_py.dat"
# FILENAME  = "input/PARMTEQOUT.TXT"
# FILENAME  = "input/Parm_50_63cells.dat"
# FILENAME  = "input/fieldoutput.txt"
FILENAME  = "input/fieldw015width.dat"

VANE_RAD  = 1 * cm
VANE_DIST = 2.5 * cm
# VANE_DIST = 2.5 * cm

NX     = 16
NY     = 16
NZ     = 512
# NX = 8
# NY = 8
# NZ = 256
PRWALL = 0.04
D_T    = 1e-9
RF_FREQ = 32.8e6
Z_START = 0.0 #the start of the rfq
SIM_START = -0.1
setup()

w3d.solvergeom = w3d.XYZgeom

w3d.xmmax =  PRWALL
w3d.xmmin = -PRWALL
w3d.nx    =  NX

w3d.ymmax =  PRWALL
w3d.ymmin = -PRWALL
w3d.ny    =  NY


w3d.zmmax =  1.456 + 0.5
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
# solver = MultiGrid3D()
# registersolver(solver)

top.npinject = 50
top.inject   = 1
w3d.l_inj_rz = False
top.zinject  = SIM_START 
w3d.zmmin = SIM_START
top.injctspc = 1000000




###############################################
# RFQ creation and initialization of parameters

rfq = PyRFQ(filename=FILENAME, from_cells=False, twoterm=True, boundarymethod=False)
rfq.vane_radius       = VANE_RAD
rfq.vane_distance     = VANE_DIST
rfq.zstart            = Z_START
rfq.rf_freq           = RF_FREQ
rfq.sim_start         = SIM_START
rfq.sim_end_buffer    = 0.5
rfq.resolution        = 0.002
rfq.endplates = False

rfq.xy_limits         = [-0.03, 0.03, -0.03, 0.03]
rfq.z_limits          = [0, 1.4]
rfq._voltage          = 22e3
rfq.tt_a_init         = 0.038802

# rfq.add_endplates  = True
# rfq.cyl_id         = 0.1
# rfq.grid_res_bempp = 0.005 
# rfq.pot_shift      = 3.0 * 22000.0
rfq.ignore_rms     = False


rfq.simple_rods    = True

rfq.setup()

# rfq.add_cell(cell_type="TCS",

#             aperture=0.011255045027294745,
#             modulation=1.6686390559337798,
#             length=0.0427)
# 0.10972618296477678

# rfq.add_cell(cell_type="DCS",
#             aperture=0.015017826368066015,
#             modulation=1.0,
#             length=0.13)



rfq.install()

############################################


# import matplotlib.pyplot as plt
# plt.axvline(x=vertical_xmin * 100, color='r')
# plt.axvline(x=vertical_xmax* 100, color='r')
# plt.axhline(y=lateral_ymin* 100)
# plt.axhline(y=lateral_ymax* 100)

# plt.axhline(y=ymin_north* 100, color='r')
# plt.axhline(y=ymax_north* 100, color='r')
# plt.axhline(y=ymax_south* 100)
# plt.axhline(y=ymin_south* 100)

# plt.axvline(x=xmin_west* 100)
# plt.axvline(x=xmax_west* 100)
# plt.axvline(x=xmin_east* 100)
# plt.axvline(x=xmax_east* 100)

# plt.show()
# exit()

beam = Species(type=Dihydrogen, charge_state=+1, name="H2+")
top.lrelativ = False
top.zbeam = 0.0
top.pgroup.nps = 0

beam.ekin  = 15.*kV      # ion kinetic energy [eV] [eV]
beam.ibeam = 10 * mA  # compensated beam current [A]
beam.emitx = 1e-6  # beam x-emittance, rms edge [m-rad]
beam.emity = 1e-6  # beam y-emittance, rms edge [m-rad]
beam.vthz  = 0.0  # axial velocity spread [m/s ec]

# Beam centroid and envelope initial conditions
beam.x0  = 0.0  # initial x-centroid xc = <x> [m]
beam.y0  = 0.0  # initial y-centroid yc = <y> [m]
beam.xp0 = 0.0  # initial x-centroid angle xc' = <x'> = d<x>/ds [rad]
beam.yp0 = 0.0  # initial y-centroid angle yc' = <y'> = d<y>/ds [rad]
beam.a0  = 5 * mm  # initial x-envelope edge a = 2*sqrt(<(x-xc)^2>) [m]
beam.b0  = 5 * mm  # initial y-envelope edge b = 2*sqrt(<(y-yc)^2>) [m]
beam.ap0 = -0.06 # initial x-envelope angle ap = a' = d a/ds [rad]
beam.bp0 = -0.06  # initial y-envelope angle bp = b' = d b/ds [rad]


utils = PyRfqUtils(rfq, beam)

boundaries = utils.find_vane_mesh_boundaries(NX, SIM_START, w3d.zmmax, -PRWALL, PRWALL, VANE_DIST, VANE_RAD)


northvane_mesh = refinedsolver.addchild(mins=boundaries["northmins"],
								  		maxs=boundaries["northmaxs"],
								  		refinement=[4,4,4])

southvane_mesh = refinedsolver.addchild(mins=boundaries["southmins"],
										maxs=boundaries["southmaxs"],
										refinement=[4,4,4])

westvane_mesh  = refinedsolver.addchild(mins=boundaries["westmins"],
										maxs=boundaries["westmaxs"],
										refinement=[4,4,4])

eastvane_mesh  = refinedsolver.addchild(mins=boundaries["eastmins"],
										maxs=boundaries["eastmaxs"],
										refinement=[4,4,4])


# Mesh refinement for the center of the beam
childmesh = refinedsolver.addchild(mins=[-VANE_DIST+VANE_RAD, -VANE_DIST+VANE_RAD, SIM_START], 
                             	   maxs=[ VANE_DIST-VANE_RAD,  VANE_DIST-VANE_RAD, w3d.zmmax],
                             	   refinement=[4,4,4])



# This routine will calculate vbeam and other quantities.`
derivqty()

package("w3d")
generate()



@callfromafterstep
def callutils():
    global utils
    utils.make_plots(1)

starttime = time.time()

step(500)
hcp()

endtime = time.time()

print("Elapsed time for simulation: {} seconds".format(endtime-starttime))

bunch = utils.find_bunch(max_steps=10000)
utils.make_plots()

utils.plot_rms_graph(SIM_START, 2)


part_x = beam.getx()
part_y = beam.gety()
part_z = beam.getz()


i = 0
while os.path.exists("particle.%s.dump" % i):
    i += 1

with open("particle.%s.dump" % i, 'w') as outfile:
    outfile.write("x, y, z\n")
    for x, y, z in zip(part_x, part_y, part_z):
        outfile.write("{:.4e}   {:.4e}   {:.4e}\n".format(x, y, z))




# import matplotlib.pyplot as plt
# print(rfq._ray)
# plt.plot(rfq._ray)
# plt.show()
