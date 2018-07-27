import numpy as np
import matplotlib.pyplot as plt
import pickle

from dans_pymodules import FileDialog

# fn = None
fn = "C:/Users/Daniel Winklehner/Documents/pot_out.field"

if fn is None:
    fd = FileDialog()
    fn = fd.get_filename()
    print(fn)

with open(fn, "rb") as infile:
    mypot = pickle.load(infile)

print(mypot.shape)

# plt.imshow(mypot[:, :, 350].T, extent=(-0.015, 0.015, -0.015, 0.015))
# plt.xlabel("x (m)")
# plt.ylabel("y (m)")
# plt.title("Potential (V)")
# plt.colorbar()
# plt.show()

plt.imshow(mypot[7, :, :], extent=(-0.1, 1.35, -0.015, 0.015))
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Potential (V)")
plt.colorbar()
plt.show()
