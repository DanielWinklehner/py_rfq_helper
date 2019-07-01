from warp import *
from py_rfq_helper.py_rfq_helper import *
from py_rfq_helper.py_rfq_designer import *
from py_rfq_helper.py_rfq_utils import *
import bisect
import time
import pprint
from dans_pymodules import IonSpecies, ParticleDistribution, FileDialog
import numpy as np
import scipy.constants as const
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtCore import QThread
from random import sample

def main():

    # FILENAME  = "input/vecc_rfq_004_py.dat"
    # FILENAME  = "input/PARMTEQOUT.TXT"
    # FILENAME  = "input/Parm_50_63cells.dat"
    # FILENAME  = "input/fieldoutput.txt"
    FIELD_FILENAME  = "input/fieldw015width.dat"


    # Initialization of basic RFQ parameters
    VANE_RAD   = 1 * cm    # radius of vane cylinder
    VANE_DIST  = 2.5 * cm  # distance of vane center to central axis
    NX, NY, NZ = 8, 8, 256
    PRWALL     = 0.04
    D_T        = 1e-9
    RF_FREQ    = 32.8e6
    Z_START    = 0.0  #the start of the rfq
    SIM_START  = -0.1

    # setup() # Warp setup function

    ## Warp parameter specifications for simulation
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

    # refinedsolver = MRBlock3D() # Refined mesh solver
    # registersolver(refinedsolver)
    solver = MultiGrid3D()    # Non-refined mesh solver
    registersolver(solver)

    top.npinject = 50
    top.inject   = 1
    w3d.l_inj_rz = False
    top.zinject  = SIM_START 
    w3d.zmmin    = SIM_START
    top.injctspc = 1000000



    ## RFQ specification and declaration
    rfq = PyRFQ(filename=FIELD_FILENAME, from_cells=False, twoterm=True, boundarymethod=False)
    rfq.vane_radius    = VANE_RAD
    rfq.vane_distance  = VANE_DIST
    rfq.zstart         = Z_START
    rfq.rf_freq        = RF_FREQ
    rfq.sim_start      = SIM_START
    rfq.sim_end_buffer = 0.5
    rfq.resolution     = 0.002
    rfq.endplates      = False

    rfq.xy_limits = [-0.03, 0.03, -0.03, 0.03]
    rfq.z_limits  = [0, 1.4]
    rfq._voltage  = 22e3
    rfq.tt_a_init = 0.038802

    # rfq.add_endplates  = True
    # rfq.cyl_id         = 0.1
    # rfq.grid_res_bempp = 0.005 
    # rfq.pot_shift      = 3.0 * 22000.0
    rfq.ignore_rms  = False
    rfq.simple_rods = True

    rfq.setup()
    rfq.install()



    # pp = Species(type=Proton, charge_state=pd[0].ion.z(), name=pd[0].ion.name())
    # beam = Species(type=Dihydrogen, charge_state=pd[1].ion.z(), name=pd[1].ion.name())
    beam = Species(type=Dihydrogen, charge_state=+1, name="H2+")
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
    beam.ap0 = -0.03 # initial x-envelope angle ap = a' = d a/ds [rad]
    beam.bp0 = -0.03 # initial y-envelope angle bp = b' = d b/ds [rad]


    top.lrelativ = False
    top.zbeam = 0.0
    top.pgroup.nps = 0


    utils = PyRfqUtils(rfq, beam)

    # boundaries = utils.find_vane_mesh_boundaries(NX, SIM_START, w3d.zmmax, -PRWALL, PRWALL, VANE_DIST, VANE_RAD)


    # northvane_mesh = refinedsolver.addchild(mins=boundaries["northmins"],
    #                                         maxs=boundaries["northmaxs"],
    #                                         refinement=[4,4,4])

    # southvane_mesh = refinedsolver.addchild(mins=boundaries["southmins"],
    #                                         maxs=boundaries["southmaxs"],
    #                                         refinement=[4,4,4])

    # westvane_mesh  = refinedsolver.addchild(mins=boundaries["westmins"],
    #                                         maxs=boundaries["westmaxs"],
    #                                         refinement=[4,4,4])

    # eastvane_mesh  = refinedsolver.addchild(mins=boundaries["eastmins"],
    #                                         maxs=boundaries["eastmaxs"],
    #                                         refinement=[4,4,4])


    # # Mesh refinement for the center of the beam
    # childmesh = refinedsolver.addchild(mins=[-VANE_DIST+VANE_RAD, -VANE_DIST+VANE_RAD, SIM_START], 
    #                                    maxs=[ VANE_DIST-VANE_RAD,  VANE_DIST-VANE_RAD, w3d.zmmax],
    #                                    refinement=[4,4,4])


    derivqty()

    package("w3d")
    generate()

    # PyQtgraph setup
    app = pg.mkQApp()

    view = pg.PlotWidget()
    view.show()
    scatter = pg.ScatterPlotItem(pen=pg.mkPen(width=1, color='g'), symbol='s', size=0.25)
    scatter_y = pg.ScatterPlotItem(pen=pg.mkPen(width=1, color='r'), symbol='s', size=0.25)
    view.setRange(xRange=[-0.1,0.75], yRange=[-0.005,0.005])
    view.addItem(scatter)
    view.addItem(scatter_y)

    @callfromafterstep
    def makeplots():
        if top.it%2 == 0:
            utils.beamplots()

    @callfromafterstep
    def plotpyqt():
        particle_display_factor = 0.25
        if top.it%10 == 1:
            x_by_z_particles = list(zip(beam.getz(),beam.getx()))
            factored_x_by_z = sample(x_by_z_particles, int(len(x_by_z_particles)*particle_display_factor))
            scatter.setData(pos=factored_x_by_z)
            QtGui.QApplication.processEvents()


    starttime = time.time()
    step(1000)
    hcp()
    endtime = time.time()  
    print("Elapsed time for simulation: {} seconds".format(endtime-starttime))

    sys.exit(app.exec_()) 



if __name__ == '__main__':
    main()