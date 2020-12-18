# Post processing tools for PyRFQ bunch data
# Jared Hwang July 2019

from py_rfq_helper.post_processing import *

postprocessor = BunchPostProcessor(32.8e6)
# postprocessor.find_bunch(20000, 18011, 1.4, velocity_min_sample=75000, step_interval=10, plot_bunch_name='H2_1+')
# postprocessor.emittancePlots('H2_1+', auto_bunch_detection=True, min_cluster_size=50, test_cluster=True)
# postprocessor.make_3d_bunch_plot(make_energy_plot=False)
