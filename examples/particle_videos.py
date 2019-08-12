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

__author__ = "Jared Hwang"
__doc__ = """MP4 creator using particle data"""


colors = MyColors()

# Read file data and sort it into dictionary for easy access later
# Sorts into dictionary categorized by species (identified with mass and charge)
def catalogue_data(f):

    print("cataloguing...")

    data_dict = {}

    part_data = f[random.choice(list(f.keys()))]

    list_of_unique = np.unique(list(zip(part_data['q'], part_data['m'])), axis=0)

    num_species = len(list_of_unique)

    species_id_list = np.arange(num_species)

    if num_species == 1:
        for key in f.keys():
            print(key)
            data_dict[key] = {}
            keyname = 'Species:0'
            data_dict[key][keyname] = {}
            data_dict[key][keyname]['x'] = f[key]['x']
            data_dict[key][keyname]['y'] = f[key]['y']
            data_dict[key][keyname]['z'] = f[key]['z']

    else:
        print('There is more than one species!')
        for key in f.keys():
            print(key)
            data_dict[key] = {}

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

    data_dict['SpeciesIdList'] = species_id_list

    return data_dict


# Plots X by Z on the top plot, and Y by Z on the bottom plot
# Functions with multiple species
def side_by_side_plot(data, zlim=(-0.1,2),vert_lim=(-0.01,0.01),
                      point_size=0.5, xcolor=colors[5], ycolor=colors[6],
                      yscale=1e-2, yunits='cm', frames=2000, figsize=(20,10), fps=30):

    num_species = len(data['SpeciesIdList'])

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(211, xlim=zlim, ylim=vert_lim)
    ax2 = fig.add_subplot(212, xlim=zlim, ylim=vert_lim)
    
    plot_array = []
    for i in range(0, num_species):
        plot_array += (ax1.plot([], [], 'bo', ms=point_size, color=colors[i]))
        plot_array += (ax2.plot([], [], 'bo', ms=point_size, color=colors[i]))

    # Axes scalers
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/yscale))
    ax1.yaxis.set_major_formatter(ticks_y)
    ax2.yaxis.set_major_formatter(ticks_y)

    fig.suptitle('Particles position vs Z', fontsize=16)
    plt.xlabel('Z (m)')

    ax1.set_ylabel('X ' + '(' + yunits + ')')
    ax2.set_ylabel('Y ' + '(' + yunits + ')')

    step_list = [item[5:] for item in list(data.keys())]

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
        if str(i) in step_list:
            step_str = "Step#" + str(i)

            # For every species identified in the cataloguing
            for species in data['SpeciesIdList']:
                x = data[step_str]['Species:'+str(species)]['x']
                y = data[step_str]['Species:'+str(species)]['y']
                z = data[step_str]['Species:'+str(species)]['z']

                # Ensures each species is given a different color as defined above
                plot_array[2*species].set_data(z, x)
                (plot_array[2*species + 1]).set_data(z, y)
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


def produce_particle_mp4(overlaid=False, darkmode=True, filename=None, zlim=(-0.1,2),vert_lim=(-0.01,0.01),
                  point_size=0.5, xcolor=colors[5], ycolor=colors[6],
                  yscale=1e-2, yunits='cm', frames=50, figsize=(20,8), fps=30):
    
    if filename == None:
        fd = FileDialog()
        filename = fd.get_filename()

    f = h5py.File(filename, 'r')

    if darkmode:
        plt.style.use('dark_background')

    if overlaid:
        overlaid_plot(f, zlim=zlim,vert_lim=vert_lim,
                  point_size=point_size, xcolor=xcolor, ycolor=ycolor,
                  yscale=yscale, yunits=yunits, frames=frames, figsize=figsize, fps=fps)
    else:
        side_by_side_plot(catalogue_data(f), zlim=zlim,vert_lim=vert_lim,
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
    with open('PyRFQPlotSettings.txt') as fp:
        lines = fp.readlines()

    # Extract value from settings list
    floatregex = '[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?' # courtesy of https://www.regular-expressions.info/floatingpoint.html
    for line in lines:
        line = line.replace('\n', '')
        if 'DARKMODE' in line:
            darkmode = int(re.search(floatregex, line).group())
        elif 'OVERLAID' in line:
            overlaid = int(re.search(floatregex, line).group())
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
        elif 'FPS' in line:
            fps = int(re.search(floatregex,line).group())

    filename = None
    if (len(sys.argv) > 1):
        filename = sys.argv.pop()

    starttime = time.time()
    produce_particle_mp4(overlaid=overlaid, darkmode=darkmode, filename=filename,
                         zlim=(zmin,zmax),vert_lim=(vertmin,vertmax),
                         point_size=pointsize, xcolor=xcolor, ycolor=ycolor,
                         yscale=yscale, yunits=yunits, frames=frames, figsize=(figsizex, figsizey), fps=fps)
    endtime = time.time()  

    print("Elapsed time for animating: {} seconds".format(endtime-starttime))



if __name__ == '__main__':
    main()