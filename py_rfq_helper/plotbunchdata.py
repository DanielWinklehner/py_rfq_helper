import matplotlib.pyplot as plt
import pickle
import numpy as np

bunch = pickle.load(open("bunch_particles.2.dump", "rb"))

z = bunch["z"]
vz = bunch["vz"]
y = bunch["y"]
x = bunch["x"]
xp = bunch["xp"]
yp = bunch["yp"]
idx = np.where(vz > 2.5e6)
#idx = np.where(vz > 0)

plt.scatter(z[idx] - np.mean(z[idx]), vz[idx], 15, marker='.')
plt.xlabel("Z")
plt.ylabel("Vz")
plt.title("Vz vs. Z")
# plt.gca().set_aspect("equal")

# plt.scatter(x[idx], xp[idx], 10, marker='.')
# plt.scatter(y[idx], yp[idx], 10, marker='.', c="red")

# plt.hist(z, 25)

plt.show()