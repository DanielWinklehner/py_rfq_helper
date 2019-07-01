import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from random import sample

def plotparticles(z_particles, x_particles, int topit, scatter):
    cdef float particle_display_factor = 1
    if topit%10 == 1:
        x_by_z_particles = list(zip(z_particles, x_particles))
        factored_x_by_z = sample(x_by_z_particles, int(len(x_by_z_particles)*particle_display_factor))
        scatter.setData(pos=factored_x_by_z)
        QtGui.QApplication.processEvents()

def imatestfunction():
    print('im a print function')