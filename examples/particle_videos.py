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
import time
import h5py
import sys
import random
import re
import math
import pprint

__author__ = "Jared Hwang"
__doc__ = """MP4 creator using particle data"""


colors = MyColors()

# Read file data and sort it into dictionary for easy access later
# Sorts into dictionary categorized by species (identified with mass and charge)
# TODO: currently catalogues ALL data. Should make faster by only cataloging data from startstep to #frames
def catalogue_data(f, num_species=-1):
    # if num species == -1, it is not specified by user

    print("cataloguing...")

    data_dict = {}

    if(num_species != -1): # finds number of species
        num_species = num_species
    else:
        num_species = len(list(f['SpeciesList']))

    species_id_list = np.arange(num_species)

    if num_species == 1:
        for stepnum in f.keys():
            if stepnum == 'SpeciesList':
                continue
            print(stepnum)
            data_dict[stepnum] = {}
            speciesname = list(f['SpeciesList'])[0]
            data_dict[stepnum][speciesname] = {}
            data_dict[stepnum][speciesname]['x'] = f[stepnum]['x']
            data_dict[stepnum][speciesname]['y'] = f[stepnum]['y']
            data_dict[stepnum][speciesname]['z'] = f[stepnum]['z']

    else:
        print('There is more than one species!')
        for key in f.keys():
            print(key)
            data_dict[key] = {}
            if key == 'SpeciesList':
                continue

            x = f[key]['x']
            y = f[key]['y']
            z = f[key]['z']
            qm = list(zip((f[key]['q']), (f[key]['m'])))

            i = 0
            for species in list_of_unique:
                idx = [j for j, spec in enumerate(qm) if np.array_equal(spec, species)]

                keyname = "Species:" + str(i)
                data_dict[key][keyname] = {}
                data_dict[key][keyname]['x'] = x[idx]
                data_dict[key][keyname]['y'] = y[idx]
                data_dict[key][keyname]['z'] = z[idx]
                i += 1

    data_dict['SpeciesIdList'] = sorted(list(f['SpeciesList'].keys()))


    return data_dict


# Plots X by Z on the top plot, and Y by Z on the bottom plot
# Functions with multiple species
def side_by_side_plot(data, start_step=0, step_interval=1, zlim=(-0.1,2),vert_lim=(-0.01,0.01),
                      point_size=0.5, xcolor=colors[5], ycolor=colors[6],
                      yscale=1e-2, yunits='cm', frames=2000, figsize=(20,10), fps=30):
    # data: catalogued data from catalogue_data()
    # start_step: starting step (from warp) to start video
    # step_interval: interval at which original simulation saved data

    num_species = len(data['SpeciesIdList'])

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(211, xlim=zlim, ylim=vert_lim)
    ax2 = fig.add_subplot(212, xlim=zlim, ylim=vert_lim)

    # Axes scalers
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/yscale))
    ax1.yaxis.set_major_formatter(ticks_y)
    ax2.yaxis.set_major_formatter(ticks_y)

    fig.suptitle('Particles position vs Z', fontsize=16)
    plt.xlabel('Z (m)')

    ax1.set_ylabel('X ' + '(' + yunits + ')')
    ax2.set_ylabel('Y ' + '(' + yunits + ')')


    plot_array = [] # arrange species in order in array, so when applying colors they are consistent
    for i in range(0, num_species):
        plot_array += (ax1.plot([], [], 'bo', ms=point_size, color=colors[i]))
        plot_array += (ax2.plot([], [], 'bo', ms=point_size, color=colors[i]))


    step_list = [item[5:] for item in list(data.keys())]
    step_list.remove('esIdList')
    step_list_int = np.array([int(elem) for elem in step_list])
    # ensure start_step has data
    if (start_step not in step_list_int):
        if (start_step < step_list_int.min()):
            print("Data has not started to be collected at that starting step yet. Exiting.")
            exit()
        if (start_step > step_list_int.max()):
            print("Requested start step is past last data collected step. Exiting")
            exit()
        else:
            print("Requested step is within collected data steps but not congruent with step interval. Finding nearest step as starting point")
            idx = (np.abs(step_list_int - start_step)).argmin()
            start_step = step_list_int[idx]
            print("New start step: {}".format(start_step))

    # Sets up plots
    def init():
        ax1.set_xlim(zlim)
        ax1.set_ylim(vert_lim)
        ax2.set_xlim(zlim)
        ax2.set_ylim(vert_lim)
        return plot_array

    # Plots a frame of the animation
    def animate(i):
        print("Frame {}".format(i))
        real_i = (i*step_interval) + start_step
        if str(real_i) in step_list:
            print("Plotting Step {}".format(str(real_i)))
            step_str = "Step#" + str(real_i)

            # For every species identified in the cataloguing
            i = 0
            for species in sorted(data[step_str].keys()):
                x = data[step_str][species]['x']
                y = data[step_str][species]['y']
                z = data[step_str][species]['z']

                # Ensures each species is given a different color as defined above
                plot_array[2*i].set_data(z, x)
                (plot_array[2*i + 1]).set_data(z, y)
                i += 1

        return plot_array

    ani = animation.FuncAnimation(fig, animate, frames=frames,
                                  interval=5, blit=True, init_func=init)
    # plt.show()

    date = datetime.datetime.today()
    filename = date.strftime('%Y-%m-%dT%H.%M') + "_simulation_video.mp4"
    ani.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    return

# Plots X by Z and Y by Z on the same subplot
# Does NOT differentiate species by color (yet?)
def overlaid_plot(data, zlim=(-0.1,2),vert_lim=(-0.01,0.01),
                  point_size=0.5, xcolor=colors[5], ycolor=colors[6],
                  yscale=1e-2, yunits='cm', frames=2000, figsize=(20,8), fps=30):
    # DEPRECATED, DOES NOT WORK
    # TODO: add functionality to work with the data cataloguing etc.

    fig, ax = plt.subplots(figsize=figsize)
    particlesx, = plt.plot([], [], 'bo', ms=point_size, color=xcolor)
    particlesy, = plt.plot([], [], 'bo', ms=point_size, color=ycolor)

    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/yscale))
    ax.yaxis.set_major_formatter(ticks_y)

    plt.title('Particles position vs Z')
    plt.xlabel('Z (m)')
    plt.ylabel('(' + yunits + ')')

    step_list = [item[5:] for item in list(data.keys())]

    def init():
        ax.set_xlim(zlim)
        ax.set_ylim(vert_lim)
        return particlesx, particlesy,

    def animate(i):
        print("Frame {}".format(i))
        if str(i) in step_list:
            step_str = "Step#" + str(i)
            x_part, y_part, z_part = list(data[step_str]['x']), \
                                     list(data[step_str]['y']), \
                                     list(data[step_str]['z']),
            particlesx.set_data(z_part, x_part)
            particlesy.set_data(z_part, y_part)

        return particlesx, particlesy,

    ani = animation.FuncAnimation(fig, animate, frames=frames,
                                  interval=5, blit=True, init_func=init)
    # plt.show()

    date = datetime.datetime.today()
    filename = date.strftime('%Y-%m-%dT%H.%M') + "_simulation_video.mp4"
    ani.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])


def make_3d_plot(data, species, start_step=0, step_interval=1, zlim=(-0.1,2),vert_lim=(-0.01,0.01),
                      point_size=0.5, xcolor=colors[5], ycolor=colors[6],
                      yscale=1e-2, yunits='cm', frames=2000, figsize=(20,10), fps=30):
    # makes 3d plot of the beam using the same rules as the side_by_side plot

    from mpl_toolkits.mplot3d import Axes3D
    print("Making 3d video plot")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Particles position vs Z')
    ax.set_zlabel('Y (m)')
    ax.set_ylabel('X (m)')
    ax.set_xlabel('Z (m)')

    ax.set_xlim3d(zlim[0], zlim[1])
    ax.set_ylim3d(vert_lim[0], vert_lim[1])
    ax.set_zlim3d(vert_lim[0], vert_lim[1])


    ### Scaling
    x_scale=3
    y_scale=1
    z_scale=1

    scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=1.0

    def short_proj():
      return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj=short_proj
    ### Scaling


    step_list = [item[5:] for item in list(data.keys())]
    step_list.remove('esIdList')
    step_list_int = np.array([int(elem) for elem in step_list])
    # ensure start_step has data
    if (start_step not in step_list_int):
        if (start_step < step_list_int.min()):
            print("Data has not started to be collected at that starting step yet. Exiting.")
            exit()
        if (start_step > step_list_int.max()):
            print("Requested start step is past last data collected step. Exiting")
            exit()
        else:
            print("Requested step is within collected data steps but not congruent with step interval. Finding nearest step as starting point")
            idx = (np.abs(step_list_int - start_step)).argmin()
            start_step = step_list_int[idx]
            print("New start step: {}".format(start_step))


    particles, = ax.plot([], [], [], 'bo', markersize=point_size)
    # Sets up plots
    def init():
        return particles,

    # Plots a frame of the animation
    def animate(i):
        print("Frame {}".format(i))
        real_i = (i*step_interval) + start_step
        if str(real_i) in step_list:
            print("Plotting Step {}".format(str(real_i)))
            print(species)
            step_str = "Step#" + str(real_i)

            x = data[step_str][species]['x']
            y = data[step_str][species]['y']
            z = data[step_str][species]['z']

            # xaxis = z, zaxis = y, yaxis = x
            print('got arrays')
            particles.set_data(z, x)
            print('set data')
            particles.set_3d_properties(y)
            print('set 3d')
        return particles,

    ani = animation.FuncAnimation(fig, animate, frames=frames,
                                  interval=5, blit=True, init_func=init)
    date = datetime.datetime.today()
    filename = date.strftime('%Y-%m-%dT%H.%M') + "_simulation_video.mp4"
    ani.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    return


def produce_particle_mp4(plot_3d=False, overlaid=False, darkmode=True, filename=None, start_step=0, step_interval=1, zlim=(-0.1,2),vert_lim=(-0.01,0.01),
                  point_size=0.5, xcolor=colors[5], ycolor=colors[6],
                  yscale=1e-2, yunits='cm', frames=50, figsize=(20,8), fps=30, num_species=-1):

    if filename == None:
        fd = FileDialog()
        filename = fd.get_filename()

    f = h5py.File(filename, 'r')

    if darkmode:
        plt.style.use('dark_background')

    if plot_3d:
        make_3d_plot(catalogue_data(f, num_species=num_species), 'H2_1+', start_step=start_step, step_interval=step_interval, zlim=zlim,vert_lim=vert_lim,
                  point_size=point_size, xcolor=xcolor, ycolor=ycolor,
                  yscale=yscale, yunits=yunits, frames=frames, figsize=figsize, fps=fps)
    elif overlaid:
        overlaid_plot(f, zlim=zlim,vert_lim=vert_lim,
                  point_size=point_size, xcolor=xcolor, ycolor=ycolor,
                  yscale=yscale, yunits=yunits, frames=frames, figsize=figsize, fps=fps)
    else:
        side_by_side_plot(catalogue_data(f, num_species=num_species), start_step=start_step, step_interval=step_interval, zlim=zlim,vert_lim=vert_lim,
                  point_size=point_size, xcolor=xcolor, ycolor=ycolor,
                  yscale=yscale, yunits=yunits, frames=frames, figsize=figsize, fps=fps)



def main():
    overlaid = False
    darkmode = True
    if "-o" in sys.argv:
        overlaid = True
        sys.argv.remove('-o')
    if "-l" in sys.argv:
        darkmode = False
        sys.argv.remove('-l')

    # Video and plot settings are contained in file called PyRFQPlotSettings.txt for now
    with open('PyRFQBeamVideoSettings.txt') as fp:
        lines = fp.readlines()
        print(lines)
    # Extract value from settings list
    floatregex = '[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?' # courtesy of https://www.regular-expressions.info/floatingpoint.html
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        # line = line.replace('\n', '')
        if line[0] == '#':
            continue
        elif 'DARKMODE' in line:
            darkmode = int(re.search(floatregex, line).group())
        elif 'THREED' in line:
            plot_3d = int(re.search(floatregex, line).group())
        elif 'OVERLAID' in line:
            overlaid = int(re.search(floatregex, line).group())
        elif 'STARTSTEP' in line:
            startstep = int(re.search(floatregex, line).group())
        elif 'STEPINTERVAL' in line:
            step_interval = int(re.search(floatregex, line).group())
        elif 'ZMIN' in line:
            zmin = float(re.search(floatregex, line).group())
        elif 'ZMAX' in line:
            zmax = float(re.search(floatregex, line).group())
        elif 'VERTMIN' in line:
            vertmin = float(re.search(floatregex, line).group())
        elif 'VERTMAX' in line:
            vertmax = float(re.search(floatregex, line).group())
        elif 'POINTSIZE' in line:
            pointsize = float(re.search(floatregex, line).group())
        elif 'XCOLOR' in line:
            xcolor = line.replace(' ', '')
            xcolor = xcolor.replace('XCOLOR:', '')
        elif 'YCOLOR' in line:
            ycolor = line.replace(' ', '')
            ycolor = ycolor.replace('YCOLOR:', '')
        elif 'YSCALE' in line:
            yscale = float(re.search(floatregex, line).group())
        elif 'YUNITS' in line:
            yunits = line.replace(' ', '')
            yunits = yunits.replace('YUNITS:', '')
        elif 'FRAMES' in line:
            frames = int(re.search(floatregex, line).group())
        elif 'FIGSIZEX' in line:
            figsizex = int(re.search(floatregex, line).group())
        elif 'FIGSIZEY' in line:
            figsizey = int(re.search(floatregex, line).group())
        elif 'NUMSPECIES' in line:
            num_species = int(re.search(floatregex, line).group())
        elif 'FPS' in line:
            fps = int(re.search(floatregex, line).group())

    filename = None
    if (len(sys.argv) > 1):
        filename = sys.argv.pop()

    starttime = time.time()
    produce_particle_mp4(plot_3d=plot_3d, overlaid=overlaid, darkmode=darkmode, filename=filename,
                         start_step=startstep, step_interval=step_interval, zlim=(zmin,zmax),vert_lim=(vertmin,vertmax),
                         point_size=pointsize, xcolor=xcolor, ycolor=ycolor,
                         yscale=yscale, yunits=yunits, frames=frames, figsize=(figsizex, figsizey), num_species=num_species, fps=fps)
    endtime = time.time()

    print("Elapsed time for animating: {} seconds".format(endtime-starttime))



if __name__ == '__main__':
    main()
