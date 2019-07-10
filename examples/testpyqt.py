#!/usr/bin/python3
#
# Scatter plot example using pyqtgraph with PyQT5
#
# Install instructions for Mac:
#   brew install pyqt
#   pip3 install pyqt5 pyqtgraph
#   python3 pyqtgraph_pyqt5.py

import sys

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

app = pg.mkQApp()


view = pg.PlotWidget()
view.show()
# plot_widget.setRange(xRange=[-0.1,0.75], yRange=[-0.005,0.005])
# rms_x.plot([1, 2, 3], [1, 2, 3], pen=pg.mkPen(width=1, color='g'))
view = pg.PlotWidget()
view.show()
x_top_rms = pg.PlotDataItem(pen=pg.mkPen(width=1, color='g'))
# x_bottom_rms = pg.PlotDataItem(pen=pg.mkPen(width=1, color='g'))
view.addItem(x_top_rms)
# view.addItem(x_bottom_rms)
x_top_rms.setData([1, 2, 3], [1, 2, 3])
QtGui.QApplication.processEvents()


sys.exit(app.exec_()) 
