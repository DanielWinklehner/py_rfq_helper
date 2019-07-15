from warp import *
from py_rfq_helper.py_rfq_helper import *
from py_rfq_helper.py_rfq_designer import *
from py_rfq_helper.py_rfq_utils import *
import bisect
import time
import pprint
from dans_pymodules import IonSpecies, ParticleDistribution, FileDialog, MyColors
import numpy as np
import scipy.constants as const
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtCore import QThread
from random import sample
from my_pzplots import *

colors = MyColors()

def load_from_ibsimu(filename):
    # Some constants
    clight = const.value("speed of light in vacuum")  # (m/s)
    amu_kg = const.value("atomic mass constant")  # (kg)
    amu_mev = const.value("atomic mass constant energy equivalent in MeV")  # MeV
    echarge = const.value("elementary charge")

    # IBSimu particle file: I, M (kg), t, x (m), vx (m/s), y (m), vy (m/s), z (m), vz (m/s)
    with open(filename) as infile:
        lines = infile.readlines()

    npart = len(lines)

    current = np.empty(npart)
    mass = np.empty(npart)
    x = np.empty(npart)
    y = np.empty(npart)
    z = np.empty(npart)
    vx = np.empty(npart)
    vy = np.empty(npart)
    vz = np.empty(npart)

    for i, line in enumerate(lines):
        current[i], mass[i], _, x[i], vx[i], y[i], vy[i], z[i], vz[i] = [float(item) for item in line.strip().split()]

    masses = np.sort(np.unique(mass))  # mass in MeV, sorted in ascending order (protons before h2+)

    particle_distributions = []

    for i, m in enumerate(masses):

        m_mev = m / amu_kg * amu_mev

        species_indices = np.where((mass == m) & (vz > 5.0e5))

        ion = IonSpecies("Species {}".format(i + 1),
                         mass_mev=m_mev,
                         a=m_mev / amu_mev,
                         z=np.round(m_mev / amu_mev, 0),
                         q=1.0,
                         current=np.sum(current[species_indices]),
                         energy_mev=1)  # Note: Set energy to 1 for now, will be recalculated when calling emittance

        particle_distributions.append(
            ParticleDistribution(ion=ion,
                                 x=x[species_indices],
                                 y=y[species_indices],
                                 z=z[species_indices],
                                 vx=vx[species_indices],
                                 vy=vy[species_indices],
                                 vz=vz[species_indices]
                                 ))

        # plt.scatter(x[species_indices], y[species_indices], s=0.5)
        # plt.show()
        # plt.scatter(x[species_indices], vx[species_indices]/vz[species_indices], s=0.5)
        # plt.show()

        particle_distributions[-1].calculate_emittances()


    return particle_distributions, current, mass, x, vx, y, vy, z, vz





def main():
    # FIELD_FILENAME  = "input/vecc_rfq_004_py.dat"
    # FILENAME  = "input/PARMTEQOUT.TXT"
    # FILENAME  = "input/Parm_50_63cells.dat"
    # FILENAME  = "input/fieldoutput.txt"
    FIELD_FILENAME  = "input/fieldw015width.dat"


    # Initialization of basic RFQ parameters
    VANE_RAD   = 1 * cm    # radius of vane cylinder
    VANE_DIST  = 2.5 * cm  # distance of vane center to central axis
    NX, NY, NZ = 16, 16, 512
    PRWALL     = 0.04
    D_T        = 1e-9
    RF_FREQ    = 32.8e6
    # Z_START    = 0.0  #the start of the rfq
    # SIM_START  = -0.1

    Z_START = 0.4
    SIM_START = 0.3

    setup() # Warp setup function

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

    top.npinject = 0
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
    # beam = Species(type=Dihydrogen, charge_state=+1, name="H2+", color=red)
    # beam.ekin  = 15.*kV      # ion kinetic energy [eV] [eV]
    # beam.ibeam = 10 * mA  # compensated beam current [A]
    # beam.emitx = 1e-6  # beam x-emittance, rms edge [m-rad]
    # beam.emity = 1e-6  # beam y-emittance, rms edge [m-rad]
    # beam.vthz  = 0.0  # axial velocity spread [m/s ec]


    particle_dist, current, mass, x, vx, y, vy, z, vz= load_from_ibsimu('./input/particle_out_461mm_n5kv_10ma_20KV.txt')
    h2_beam = Species(type=Dihydrogen, charge_state=+1, name="H2+")
    proton_beam = Species(type=Proton, charge_state=+1, name="P", color=red)
    # print("LEN X: ", len(x))
    # add_particles(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, lallindomain=True)
    top.ainject = 0.05
    top.binject = 0.05 
    # beam.ekin  = 15.*kV      # ion kinetic energy [eV] [eV]
    # beam.ibeam = 10 * mA  # compensated beam current [A]
    # beam.emitx = 1e-6  # beam x-emittance, rms edge [m-rad]
    # beam.emity = 1e-6  # beam y-emittance, rms edge [m-rad]
    # beam.vthz  = 0.0  # axial velocity spread [m/s ec]

    
    
    

    def createmybeam():
        idx = np.random.choice(np.arange(len(x)), 1000, replace=False)
        mass_inject = mass[idx]
        x_inject = x[idx]
        y_inject = y[idx]
        z_inject = z[idx]
        vx_inject = vx[idx]
        vy_inject = vy[idx]
        vz_inject = vz[idx]

        h2_indices = np.where(mass_inject > 1.7e-27)[0]
        h2_beam.addparticles(x=x_inject[h2_indices],
                             y=y_inject[h2_indices],
                             z=z_inject[h2_indices],
                             vx=vx_inject[h2_indices],
                             vy=vy_injec[h2_indicest],
                             vz=vz_inject[h2_indices])

        proton_indices = np.where(mass_inject < 1.7e-27)[0]
        proton_beam.addparticles(x=x_inject[proton_indices],
                             y=y_inject[proton_indices],
                             z=z_inject[proton_indices],
                             vx=vx_inject[proton_indices],
                             vy=vy_injec[proton_indicest],
                             vz=vz_inject[proton_indices])
    
    installuserinjection(createmybeam)

    # Beam centroid and envelope initial conditions
    h2_beam.x0  = 0.0  # initial x-centroid xc = <x> [m]
    h2_beam.y0  = 0.0  # initial y-centroid yc = <y> [m]
    h2_beam.xp0 = 0.0  # initial x-centroid angle xc' = <x'> = d<x>/ds [rad]
    h2_beam.yp0 = 0.0  # initial y-centroid angle yc' = <y'> = d<y>/ds [rad]
    h2_beam.a0  = 5 * mm  # initial x-envelope edge a = 2*sqrt(<(x-xc)^2>) [m]
    h2_beam.b0  = 5 * mm  # initial y-envelope edge b = 2*sqrt(<(y-yc)^2>) [m]
    h2_beam.ap0 = -0.03 # initial x-envelope angle ap = a' = d a/ds [rad]
    h2_beam.bp0 = -0.03 # initial y-envelope angle bp = b' = d b/ds [rad]

    proton_beam.x0  = 0.0  # initial x-centroid xc = <x> [m]
    proton_beam.y0  = 0.0  # initial y-centroid yc = <y> [m]
    proton_beam.xp0 = 0.0  # initial x-centroid angle xc' = <x'> = d<x>/ds [rad]
    proton_beam.yp0 = 0.0  # initial y-centroid angle yc' = <y'> = d<y>/ds [rad]
    proton_beam.a0  = 5 * mm  # initial x-envelope edge a = 2*sqrt(<(x-xc)^2>) [m]
    proton_beam.b0  = 5 * mm  # initial y-envelope edge b = 2*sqrt(<(y-yc)^2>) [m]
    proton_beam.ap0 = -0.03 # initial x-envelope angle ap = a' = d a/ds [rad]
    proton_beam.bp0 = -0.03 # initial y-envelope angle bp = b' = d b/ds [rad]


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

    # rms_x.plot(pen=pg.mkPen(width=1, color='g'), size=1)

    # colors = MyColors()

    utils.rms_plot_setup(title="X and Y RMS (twice rms) vs Z", labels={'left':('X, Y', 'm'), 'bottom':('Z', 'm')})
    
    @callfromafterstep
    def makeplots():
        if top.it%2 == 0:
            # utils.plot_rms()
            utils.beamplots()
            # window()
            # limits(-0.1, 1, -0.01, 0.01)
            # pzxedges(color='blue')
            # pzyedges(color='red')
            # fma()
            # refresh() 


    # utils.particle_plot_setup(title="X and Y Particles vs Z", labels={'left':('X, Y', 'm'), 'bottom':('Z', 'm')})

    # @callfromafterstep
    # def plotpyqt():
    #     if top.it%5 == 1:
    #         utils.plot_particles(factor=0.2)

    # @callfromafterstep
    # def output_particles():
    #     utils.write_particle_data(top.it, 2)


    starttime = time.time()
    step(1500)
    hcp()
    endtime = time.time()  
    print("Elapsed time for simulation: {} seconds".format(endtime-starttime))
        
    # sys.exit(app.exec_()) 





if __name__ == '__main__':
    main()