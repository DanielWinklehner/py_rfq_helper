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
from mpi4py import MPI
from my_pzplots import *

__author__ = "Jared Hwang"
__doc__ = """Example PyRFQ Simulation"""


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


    return current, mass, x, vx, y, vy, z, vz





def main():
    FIELD_FILENAME = 'input/2019_07_23_PyRFQ_Test_Field_Voltage_25kV_W_In15keV_W_Out_60keV.txt'


    # Initialization of basic RFQ parameters
    VANE_RAD   = 1 * cm    # radius of vane cylinder
    VANE_DIST  = 2.5 * cm  # distance of vane center to central axis
    NX, NY, NZ = 16, 16, 512
    PRWALL     = 0.04
    D_T        = 1e-9
    RF_FREQ    = 32.8e6
    Z_START    = 0.01  #the start of the rfq
    SIM_START  = -0.014

    setup() # Warp setup function

    ## Warp parameter specifications for simulation
    w3d.solvergeom = w3d.XYZgeom
    
    w3d.xmmax =  PRWALL
    w3d.xmmin = -PRWALL
    w3d.nx    =  NX

    w3d.ymmax =  PRWALL
    w3d.ymmin = -PRWALL
    w3d.ny    =  NY

    w3d.zmmax =  1.456 + 0.3
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
    rfq = PyRFQ(filename=FIELD_FILENAME, from_cells=False, twoterm=False, boundarymethod=False)
    rfq.vane_radius    = VANE_RAD
    rfq.vane_distance  = VANE_DIST
    rfq.zstart         = Z_START
    rfq.rf_freq        = RF_FREQ
    rfq.sim_start      = SIM_START
    rfq.sim_end_buffer = 0.5
    rfq.resolution     = 0.002
    rfq.endplates      = False
    rfq.field_scaling_factor = 2

    rfq.xy_limits = [-0.03, 0.03, -0.03, 0.03]
    rfq.z_limits  = [0, 1.5]
    rfq._voltage  = 22e3
    rfq.tt_a_init = 0.038802

    # rfq.add_endplates  = True
    # rfq.cyl_id         = 0.1
    # rfq.grid_res_bempp = 0.005 
    # rfq.pot_shift      = 3.0 * 22000.0
    # rfq.ignore_rms  = False
    rfq.simple_rods = True

    rfq.setup()
    rfq.install()


    ##################################### WARP BEAM
    # beam = Species(type=Dihydrogen, charge_state=pd[1].ion.z(), name=pd[1].ion.name())
    # beam = Species(type=Dihydrogen, charge_state=+1, name="H2+", color=red)

    # beam.ekin  = 15.*kV      # ion kinetic energy [eV] [eV]
    # beam.ibeam = 10 * mA  # compensated beam current [A]
    # beam.emitx = 1.8e-6  # beam x-emittance, rms edge [m-rad]
    # beam.emity = 1.8e-6  # beam y-emittance, rms edge [m-rad]
    # beam.vthz  = 0.0  # axial velocity spread [m/s ec]
    
    # # Beam centroid and envelope initial conditions

    # twiss_emitx = 1.8e-6 /6
    # twiss_emity = 1.8e-6 /6
    # twiss_alphax = 1.9896856 #dimensionless
    # twiss_alphay = 1.9896856
    # twiss_alphaz = 0
    # twiss_betax = 13.241259 *cm / mm # cm/mrad
    # twiss_betay = 13.241259 *cm / mm
    # twiss_betaz = 45
    # twiss_gammax = (1 + twiss_alphax**2) / twiss_betax
    # twiss_gammay = (1 + twiss_alphay**2) / twiss_betay

    # beamxangle = -sqrt(twiss_emitx * twiss_gammax)
    # beamx = sqrt(twiss_emitx * twiss_betax)
    # beamyangle = -sqrt(twiss_emity * twiss_gammay)
    # beamy = sqrt(twiss_emity * twiss_betay)

    # beam.x0  = 0.0  # initial x-centroid xc = <x> [m]
    # beam.y0  = 0.0  # initial y-centroid yc = <y> [m]
    # beam.xp0 = 0.0  # initial x-centroid angle xc' = <x'> = d<x>/ds [rad]
    # beam.yp0 = 0.0  # initial y-centroid angle yc' = <y'> = d<y>/ds [rad]
    # beam.a0  = beamx
    # beam.b0  = beamy
    # beam.ap0 = beamxangle
    # beam.bp0 = beamyangle


    # beam.a0  = 5 * mm  # initial x-envelope edge a = 2*sqrt(<(x-xc)^2>) [m]
    # beam.b0  = 5 * mm  # initial y-envelope edge b = 2*sqrt(<(y-yc)^2>) [m]
    # beam.ap0 = -0.06 # initial x-envelope angle ap = a' = d a/ds [rad]
    # beam.bp0 = -0.06  # initial y-envelope angle bp = b' = d b/ds [rad]
    ###########################################################


    ##################################### PARTICLE DISTRIBUTION
    current, mass, x, vx, y, vy, z, vz = load_from_ibsimu('./input/particle_out_461mm_n5kv_10ma_20KV.txt')
    all_particles = np.array(list(zip(current, mass, x, vx, y, vy, z, vz)))

    h2_list = all_particles[np.where(mass > mass.min())]
    proton_list = all_particles[np.where(mass == mass.min())]

    h2_num = len(h2_list)
    proton_num = len(proton_list)

    h2_current = sum([i for i, _, _, _, _, _, _, _ in h2_list])
    proton_current = sum([i for i, _, _, _, _, _, _, _ in proton_list])

    h2_beam = Species(type=Dihydrogen, charge_state=+1, name="H2_1+", color=blue)
    proton_beam = Species(type=Proton, charge_state=+1, name="P", color=red)

    top.ainject = 0.05
    top.binject = 0.05 
    h2_beam.ibeam = h2_current
    proton_beam.ibeam = proton_current
    h2_beam.ekin = 15.*kV
    proton_beam.ekin = 15.*kV

    # beam.ekin  = 15.*kV      # ion kinetic energy [eV] [eV]
    # beam.ibeam = 10 * mA  # compensated beam current [A]
    # beam.emitx = 1e-6  # beam x-emittance, rms edge [m-rad]
    # beam.emity = 1e-6  # beam y-emittance, rms edge [m-rad]
    # beam.vthz  = 0.0  # axial velocity spread [m/s ec]

    # injection required flags
    w3d.l_inj_user_particles_v = true
    top.linj_enormcl = false
    top.linj_efromgrid = true

    h2_beam_id = 0
    proton_beam_id = 1

    # Adding in amount of particles proportional to total number in distribution
    total_parts_per_step = 500
    h2_per_step = int(total_parts_per_step * (h2_num / (h2_num + proton_num)))
    prot_per_step = int(total_parts_per_step * (proton_num / (h2_num + proton_num)))

    def injectionsource():
        if (w3d.inj_js == h2_beam_id):
            nump = h2_per_step
            w3d.npgrp = nump
            gchange('Setpwork3d')
            
            idx = np.random.choice(np.arange(len(h2_list)), h2_per_step, replace=False)
            h2_inject = h2_list[idx]
            _, _, h2_x, h2_vx, h2_y, h2_vy, h2_z, h2_vz = list(zip(*h2_inject))
            w3d.xt[:] = h2_x
            w3d.yt[:] = h2_y
            # w3d.zt[:] = np.full((len(h2_x)), 0)
            w3d.uxt[:] = h2_vx
            w3d.uyt[:] = h2_vy
            w3d.uzt[:] = h2_vz

        elif (w3d.inj_js == proton_beam_id):
            nump = prot_per_step
            w3d.npgrp = nump
            gchange('Setpwork3d')

            idx = np.random.choice(np.arange(len(proton_list)), prot_per_step, replace=False)
            proton_inject = proton_list[idx]
            _, _, p_x, p_vx, p_y, p_vy, p_z, p_vz = list(zip(*proton_inject))
            w3d.xt[:] = p_x
            w3d.yt[:] = p_y
            # w3d.zt[:] = np.full((len(p_x)), 0)
            w3d.uxt[:] = p_vx
            w3d.uyt[:] = p_vy
            w3d.uzt[:] = p_vz

    installuserparticlesinjection(injectionsource)


    # Beam centroid and envelope initial conditions
    h2_beam.x0  = 0.0  # initial x-centroid xc = <x> [m]
    h2_beam.y0  = 0.0  # initial y-centroid yc = <y> [m]
    h2_beam.xp0 = 0.0  # initial x-centroid angle xc' = <x'> = d<x>/ds [rad]
    h2_beam.yp0 = 0.0  # initial y-centroid angle yc' = <y'> = d<y>/ds [rad]
    h2_beam.a0  = 5 * mm  # initial x-envelope edge a = 2*sqrt(<(x-xc)^2>) [m]
    h2_beam.b0  = 5 * mm  # initial y-envelope edge b = 2*sqrt(<(y-yc)^2>) [m]
    h2_beam.ap0 = -0.03 # initial x-envelope angle ap = a' = d a/ds [rad]
    h2_beam.bp0 = -0.03 # initial y-envelope angle bp = b' = d b/ds [rad]
    h2_beam.ekin = 15 * kV

    proton_beam.x0  = 0.0  # initial x-centroid xc = <x> [m]
    proton_beam.y0  = 0.0  # initial y-centroid yc = <y> [m]
    proton_beam.xp0 = 0.0  # initial x-centroid angle xc' = <x'> = d<x>/ds [rad]
    proton_beam.yp0 = 0.0  # initial y-centroid angle yc' = <y'> = d<y>/ds [rad]
    proton_beam.a0  = 5 * mm  # initial x-envelope edge a = 2*sqrt(<(x-xc)^2>) [m]
    proton_beam.b0  = 5 * mm  # initial y-envelope edge b = 2*sqrt(<(y-yc)^2>) [m]
    proton_beam.ap0 = -0.03 # initial x-envelope angle ap = a' = d a/ds [rad]
    proton_beam.bp0 = -0.03 # initial y-envelope angle bp = b' = d b/ds [rad]
    proton_beam.ekin = 15 * kV
    ############################################################################


    top.lrelativ = False
    top.zbeam = 0.0
    top.pgroup.nps = 0

    utils = PyRfqUtils(rfq, [h2_beam, proton_beam])


    ##################################### MESH REFINEMENT
    # note: after testing, using mesh refinement appeared to be slower than not using it, ergo commented out
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
    #########################################

    derivqty()

    package("w3d")
    generate()

    # WARP built in plotting 
    # @callfromafterstep
    # def makeplots():
    #     if top.it > 19900:
    #         if top.it%10 == 0:
    #             # utils.plot_rms()
    #             utils.beamplots()
    #             # print(h2_beam.getux())
    #             # window()
    #             # limits(-0.1, 1, -0.01, 0.01)
    #             # pzxedges(color='blue')
    #             # pzyedges(color='red')
    #             # fma()
    #             # refresh() 

    ################################# PyQTGraph RMS plotting
    # # PyQtgraph setup
    # app = pg.mkQApp()

    # # Setup the rms plot
    # utils.rms_plot_setup(title="X and Y RMS (twice rms) vs Z", labels={'left':('X, Y', 'm'), 'bottom':('Z', 'm')}, 
    #                      xrange=[-0.1, 1.6], yrange=[-0.015, 0.015])

    # ## setup the particle plots. Not recommended, slows down simulation immensely    
    # # utils.particle_plot_setup(title="X and Y Particles vs Z", labels={'left':('X, Y', 'm'), 'bottom':('Z', 'm')})

    # @callfromafterstep
    # def plotpyqt():
    #     if top.it%2 == 1:
    #         utils.plot_rms()

    STEP_NUM = 2000
    PARTICLE_OUTPUT_STARTSTEP = 1800
    PARTICLE_OUTPUT_FRAME_FREQ = 2

    @callfromafterstep
    def output_particles():
        if top.it > PARTICLE_OUTPUT_STARTSTEP:
            if top.it%PARTICLE_OUTPUT_FRAME_FREQ == 0:
                utils.write_hdf5_data_p(top.it, [h2_beam])


    starttime = time.time()
    step(STEP_NUM)
    hcp()
    endtime = time.time()  
    print("Elapsed time for simulation: {} seconds".format(endtime-starttime))
        
    # bunch = utils.find_bunch_p(h2_beam, max_steps=10000)


if __name__ == '__main__':
    main()