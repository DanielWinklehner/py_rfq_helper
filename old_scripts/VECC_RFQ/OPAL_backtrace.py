from dans_pymodules import *
import os
import matplotlib.pyplot as plt
import platform

colors = MyColors()

# H2+ with 15 keV total energy
ion = IonSpecies('H2_1+', 1.0)
ion.calculate_from_energy_mev(0.015 / ion.a())

if platform.node() == "Mailuefterl":
    folder = r"D:\Daniel\Dropbox (MIT)\Projects" \
             r"\RFQ Direct Injection\RFQ_Tests\VECC-Design\rfq_vecc_004"
elif platform.node() == "TARDIS":
    folder = r"D:\Dropbox (MIT)\Projects" \
             r"\RFQ Direct Injection\RFQ_Tests\VECC-Design\rfq_vecc_004"
else:
    folder = r"C:\Users\Daniel Winklehner\Dropbox (MIT)\Projects" \
             r"\RFQ Direct Injection\RFQ_Tests\VECC-Design\rfq_vecc_004"

output_folder = os.path.join(folder, "output_with_end2")

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# end_particles_fn = "rfq_vecc_004_step4078.dat"
# start_particles_fn = "rfq_vecc_004_step0000.dat"

end2_particles_fn = "rfq_vecc_004_phi-5deg_step4200.dat"
end1_particles_fn = "rfq_vecc_004_phi-5deg_step4078.dat"
start_particles_fn = "rfq_vecc_004_phi-5deg_step0000.dat"
start_particles_fn = os.path.join(folder, start_particles_fn)
end1_particles_fn = os.path.join(folder, end1_particles_fn)
end2_particles_fn = os.path.join(folder, end2_particles_fn)

# --- Load start data --- #
with open(start_particles_fn, 'r') as infile:
    print("\nHeader:\n{}".format(infile.readline()))

    data = []

    for line in infile.readlines():

        data.append(tuple([float(item) for item in line.strip().split()]))

mydtype = [("step", int), ("ID", int),
           ("x", float), ("y", float), ("z", float),
           ("px", float), ("py", float), ("pz", float)
           ]

start_data = np.array(data, dtype=mydtype)

# --- Load end1 data (directly at exit) --- #
with open(end1_particles_fn, 'r') as infile:
    print("\nHeader:\n{}".format(infile.readline()))

    data = []

    for line in infile.readlines():

        data.append(tuple([float(item) for item in line.strip().split()]))

mydtype = [("step", int), ("ID", int),
           ("x", float), ("y", float), ("z", float),
           ("px", float), ("py", float), ("pz", float)
           ]

end1_data = np.array(data, dtype=mydtype)

# --- Load end2 data (a little further so that neighboring bunch has full energy too)--- #
with open(end2_particles_fn, 'r') as infile:
    print("\nHeader:\n{}".format(infile.readline()))

    data = []

    for line in infile.readlines():

        data.append(tuple([float(item) for item in line.strip().split()]))

mydtype = [("step", int), ("ID", int),
           ("x", float), ("y", float), ("z", float),
           ("px", float), ("py", float), ("pz", float)
           ]

end2_data = np.array(data, dtype=mydtype)

# Change momentum (be*ga) to total energy (keV)
for key in ["px", "py", "pz"]:
    start_data[key] = 1.0e3 * (np.sqrt(start_data[key] ** 2.0 + 1.0) - 1.0) * ion.mass_mev()
    end1_data[key] = 1.0e3 * (np.sqrt(end1_data[key] ** 2.0 + 1.0) - 1.0) * ion.mass_mev()
    end2_data[key] = 1.0e3 * (np.sqrt(end2_data[key] ** 2.0 + 1.0) - 1.0) * ion.mass_mev()

# Temp: Plot end data to find energy cut values #
# plt.scatter(end1_data["z"], end1_data["pz"], s=0.25, c=colors[0])
# plt.xlabel(r"z (m)")
# plt.ylabel(r"E (keV)")
# plt.show()
# exit()

# --- Apply cuts --- #
bunch_parameters = {"Main": {"pz_min": 64.0,
                             "z_min": 1.29,
                             "z_max": 1.34,
                             "plot_fn": None},
                    "Left": {"pz_min": 45.0,
                             "z_min": 1.21,
                             "z_max": 1.26,
                             "plot_fn": None},
                    "Right": {"pz_min": 64.0,
                              "z_min": 1.37,
                              "z_max": 1.43,
                              "plot_fn": None}}

# Plot the backtracked particles from the central three bunches
for key, params in bunch_parameters.items():
    params["plot_fn"] = os.path.join(output_folder, key+"Bunch.png")

    # Apply cut
    idx = np.where(
        (end1_data["z"] < params["z_max"]) & (params["z_min"] < end1_data["z"]) & (end1_data["pz"] > params["pz_min"]))

    # --- Find cut particles in start distribution --- #
    idx2 = []
    desired_ids = end1_data["ID"][idx]
    for i, particle in enumerate(start_data):
        if particle["ID"] in desired_ids:
            idx2.append(i)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # --- Do some plotting --- #
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 6)

    ax[0].scatter(start_data["z"], start_data["y"], s=0.25, c=colors[0])
    ax[0].scatter(start_data["z"][idx2], start_data["y"][idx2], s=0.25, c=colors[1])
    ax[0].set_xlabel("z (m)")
    ax[0].set_ylabel("y (m)")
    ax[0].set_title("Step = 0000")

    ax[1].scatter(end1_data["z"], end1_data["pz"], s=0.25, c=colors[0])
    ax[1].scatter(end1_data["z"][idx], end1_data["pz"][idx], s=0.25, c=colors[1])
    ax[1].set_xlabel("z (m)")
    ax[1].set_ylabel(r"E (keV)")
    ax[1].set_title("Step = 4078")

    # Print out the energy spread of the bunch
    e_cut = end1_data["pz"][idx]
    e_mean = np.mean(e_cut)
    e_std = np.std(e_cut)
    e_max = max(e_cut)
    e_min = min(e_cut)
    delta_e = (e_max - e_min) * 0.5

    # ion.calculate_from_momentum_betagamma(momentum_bega=pz_mean)
    # e_mean = 1000.0 * ion.energy_mev() * ion.a()
    # ion.calculate_from_momentum_betagamma(momentum_bega=pz_std)
    # e_std = 1000.0 * ion.energy_mev() * ion.a()
    # ion.calculate_from_momentum_betagamma(momentum_bega=delta_pz)
    # e_delta = 1000.0 * ion.energy_mev() * ion.a()

    numpart_2percent = len(np.where(np.abs(end1_data["pz"][idx] - e_mean) <= 0.02 * e_mean)[0])

    print(r"Mean energy of {} bunch = {} keV".format(key, e_mean))
    print(r"Sigma E = {} keV, (Sigma E)/E = {:.2f} %".format(e_std, 100.0 * e_std/e_mean))
    print(r"Delta E max = {} keV, (Delta E)/E = {:.2f} %".format(delta_e, 100.0 * delta_e/e_mean))
    print("Percent particles between +- 2% Energy = {:.2f}".format(100.0 * numpart_2percent / len(end1_data["z"][idx])))
    print("")

    plt.tight_layout()

    fig.savefig(params["plot_fn"], dpi=400)

    plt.close(fig)

# --- Do some analysis on the data --- #
params = bunch_parameters["Main"]

# Apply cut
idx = np.where(
    (end1_data["z"] < params["z_max"]) & (params["z_min"] < end1_data["z"]) & (end1_data["pz"] > params["pz_min"]))

# --- Find cut particles in start distribution --- #
idx2 = []
desired_ids = end1_data["ID"][idx]
for i, particle in enumerate(start_data):
    if particle["ID"] in desired_ids:
        idx2.append(i)

main_bunch_start = start_data[idx2]
z_mean = np.mean(main_bunch_start["z"])

print("Main bunch has start center z = {} m".format(z_mean))

bunch_length = 0.0366777295335042
z_max_bl = z_mean + 0.5 * bunch_length
z_min_bl = z_mean - 0.5 * bunch_length

# --- Cut the data new --- #
subset_start_idx = np.where((start_data["z"] < z_max_bl) & (z_min_bl < start_data["z"]))
subset_start = start_data[subset_start_idx]

subset_end1_idx = []
desired_ids = subset_start["ID"]
for i, particle in enumerate(end1_data):
    if particle["ID"] in desired_ids:
        subset_end1_idx.append(i)

subset_end1 = end1_data[subset_end1_idx]

subset_end2_idx = []
desired_ids = subset_start["ID"]
for i, particle in enumerate(end2_data):
    if particle["ID"] in desired_ids:
        subset_end2_idx.append(i)

subset_end2 = end2_data[subset_end2_idx]

print("Using {} particles from initial distribution from z = {} m to z = {} m".format(len(subset_start),
                                                                                      z_min_bl, z_max_bl))
trans_effic = float(len(subset_end1)) / len(subset_start)
print("Transmission efficiency: {} percent".format(100.0 * trans_effic))
accel_effic = float(len(subset_end1[np.where(subset_end1["pz"] > bunch_parameters["Left"]["pz_min"])])) / len(
    subset_end1)
print("Acceleration Efficiency: {} percent".format(100.0 * accel_effic))
print("Overall Efficiency: {} percent".format(100.0 * trans_effic * accel_effic))

# Plot the desired subset
plot_fn = os.path.join(output_folder, "CutOneBetaLambda.pdf")

plt.rc('font', size=14)
fig, ax = plt.subplots(2, 2)
fig.set_size_inches(12, 12)

idx_lost = []
for i, particle in enumerate(subset_start):
    if particle["ID"] not in subset_end2["ID"]:
        idx_lost.append(i)

# Load the RFQ parameters and diplay in upper left panel
# noinspection PyTypeChecker
_data = []
with open(os.path.join(folder, "Parm_50.dat"), "r") as infile:
    for line in infile:
        _data.append(tuple([float(item) for item in line.strip().split()]))

mydtype = [("cell no", int),
           ("energy", float),
           ("phase", float),
           ("aperture", float),
           ("modulation", float),
           ("focusing factor", float),
           ("cell length", float),
           ("cumulative length", float)]

parameters = np.array(_data, dtype=mydtype)

pl1 = ax[0, 0].plot(parameters["cell no"], 1000.0 * parameters["aperture"],
                    c=colors[0], linewidth=2, label="Aperture")
pl2 = ax[0, 0].plot(parameters["cell no"], 10.0 * parameters["modulation"],
                    c=colors[1], linewidth=2, label="Modulation")
pl3 = ax[0, 0].plot(parameters["cell no"], parameters["focusing factor"],
                    c=colors[2], linewidth=2, label="Focusing")

ax[0, 0].set_xlabel(r"Cell number")
ax[0, 0].set_ylabel(r"Aperture (mm), Focusing Factor, Modulation (x10)")
ax[0, 0].set_title(r"RFQ Parameters")
ax[0, 0].set_ylim(0, 30)
ax[0, 0].set_xlim(0, 64)

twinax = ax[0, 0].twinx()
pl4 = twinax.plot(parameters["cell no"], parameters["phase"],
                  c=colors[3], linewidth=2, label="Phase")

twinax.set_xlim(0, 64)
twinax.set_ylim(-100, -20)
twinax.set_ylabel("Phase (deg)")

allpl = pl1 + pl2 + pl3 + pl4
alllabels = [l.get_label() for l in allpl]

ax[0, 0].legend(allpl, alllabels, loc=1)

ax[0, 1].scatter(1000.0 * subset_start["z"], 1000.0 * subset_start["y"], s=0.25, c=colors[0])
ax[0, 1].scatter(1000.0 * subset_start["z"][idx_lost], 1000.0 * subset_start["y"][idx_lost], s=0.25, c=colors[1])
ax[0, 1].set_xlabel(r"z (mm)")
ax[0, 1].set_ylabel(r"y (mm)")
ax[0, 1].set_ylim(-10, 10)
ax[0, 1].set_title(r"Input Beam (side view)")
ax[0, 1].set_xlim(-50, -10)

ax[1, 0].scatter(1000.0 * subset_end2["z"], subset_end2["pz"], s=0.25, c=colors[0])
ax[1, 0].set_xlabel(r"z (mm)")
ax[1, 0].set_ylabel(r"E (keV)")
ax[1, 0].set_title(r"Output Beam (position-energy space)")
ax[1, 0].set_xlim(1000, 1600)

ax[1, 1].hist(subset_end2["pz"], bins=20)
ax[1, 1].set_xlabel(r"E (keV)")
ax[1, 1].set_ylabel(r"Counts")
ax[1, 1].set_title(r"Histogram of output energies")
ax[1, 1].set_xlim(0, 100)
ax[1, 1].set_ylim(0, 5000)

plt.tight_layout()

fig.savefig(plot_fn, dpi=400)

plt.close(fig)
