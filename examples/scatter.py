# Parse h5py data from the PyPA py_rfq_helper module and produces
# an animated plot of the particles through the beam line.
# Jared Hwang July 2019

import numpy as np
from scipy.spatial.distance import pdist, squareform
from dans_pymodules import FileDialog, MyColors
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import datetime
import h5py
import sys

colors = MyColors()

def side_by_side_plot(data, zlim=(-0.1,2),vert_lim=(-0.01,0.01),
                      point_size=0.5, xcolor=colors[5], ycolor=colors[6],
                      yscale=1e-2, yunits='cm', frames=2000, figsize=(20,10)):

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(211, xlim=zlim, ylim=vert_lim)
    ax2 = fig.add_subplot(212, xlim=zlim, ylim=vert_lim)
    particlesx, = ax1.plot([], [], 'bo', ms=point_size, color=xcolor)
    particlesy, = ax2.plot([], [], 'bo', ms=point_size, color=ycolor)

    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/yscale))
    ax1.yaxis.set_major_formatter(ticks_y)
    ax2.yaxis.set_major_formatter(ticks_y)

    fig.suptitle('Particles position vs Z', fontsize=16)
    plt.xlabel('Z (m)')

    ax1.set_ylabel('X ' + '(' + yunits + ')')
    ax2.set_ylabel('Y ' + '(' + yunits + ')')

    def init():
        ax1.set_xlim(zlim)
        ax1.set_ylim(vert_lim)
        ax2.set_xlim(zlim)
        ax2.set_ylim(vert_lim)
        return particlesx, particlesy,

    def animate(i):
        print(i)
        if str(i) in data.keys():
            x_part, y_part, z_part = zip(*data[str(i)])
            particlesx.set_data(z_part, x_part)
            particlesy.set_data(z_part, y_part)
            return particlesx, particlesy,

        return particlesx, particlesy,

    ani = animation.FuncAnimation(fig, animate, frames=frames,
                                  interval=5, blit=True, init_func=init)
    # plt.show()

    date = datetime.datetime.today()
    filename = date.strftime('%Y-%m-%dT%H.%M') + "_simulation_video.mp4"
    ani.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])


def overlaid_plot(data, zlim=(-0.1,2),vert_lim=(-0.01,0.01),
                  point_size=0.5, xcolor=colors[5], ycolor=colors[6],
                  yscale=1e-2, yunits='cm', frames=2000, figsize=(20,8)):

    fig, ax = plt.subplots(figsize=figsize)
    particlesx, = plt.plot([], [], 'bo', ms=point_size, color=xcolor)
    particlesy, = plt.plot([], [], 'bo', ms=point_size, color=ycolor)

    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/yscale))
    ax.yaxis.set_major_formatter(ticks_y)

    plt.title('Particles position vs Z')
    plt.xlabel('Z (m)')
    plt.ylabel('(' + yunits + ')')


    def init():
        ax.set_xlim(zlim)
        ax.set_ylim(vert_lim)
        return particlesx, particlesy,

    def animate(i):
        print(i)
        if str(i) in data.keys():
            x_part, y_part, z_part = zip(*data[str(i)])
            particlesx.set_data(z_part, x_part)
            particlesy.set_data(z_part, y_part)
            return particlesx, particlesy,

        return particlesx, particlesy,

    ani = animation.FuncAnimation(fig, animate, frames=frames,
                                  interval=5, blit=True, init_func=init)
    # plt.show()

    date = datetime.datetime.today()
    filename = date.strftime('%Y-%m-%dT%H.%M') + "_simulation_video.mp4"
    ani.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])


def produce_particle_mp4(overlaid=False, darkmode=True, filename=None, zlim=(-0.1,2),vert_lim=(-0.01,0.01),
                  point_size=0.5, xcolor=colors[5], ycolor=colors[6],
                  yscale=1e-2, yunits='cm', frames=2000, figsize=(20,8)):
    
    if filename == None:
        fd = FileDialog()
        filename = fd.get_filename()

    f = h5py.File(filename, 'r')
    data = f['particle_data']

    if darkmode:
        plt.style.use('dark_background')

    if overlaid:
        overlaid_plot(data, zlim=zlim,vert_lim=vert_lim,
                  point_size=point_size, xcolor=xcolor, ycolor=ycolor,
                  yscale=yscale, yunits=yunits, frames=frames, figsize=figsize)
    else:
        side_by_side_plot(data, zlim=zlim,vert_lim=vert_lim,
                  point_size=point_size, xcolor=xcolor, ycolor=ycolor,
                  yscale=yscale, yunits=yunits, frames=frames, figsize=figsize)

def main():
    overlaid = False
    darkmode = True
    if "-o" in sys.argv:
        overlaid = True
        sys.argv.remove('-o')
    if "-l" in sys.argv:
        darkmode = False
        sys.argv.remove('-l')

    filename = None
    if (len(sys.argv) > 1):
        filename = sys.argv.pop()

    produce_particle_mp4(overlaid=overlaid, darkmode=darkmode, filename=filename)


if __name__ == '__main__':
    main()