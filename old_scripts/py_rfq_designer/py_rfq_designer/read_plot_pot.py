import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d
from dans_pymodules import FileDialog
import numpy.ma as ma


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    # if x.ndim != 1:
    #     raise ValueError, "smooth only accepts 1 dimension arrays."
    #
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


# fn = None
# fn = "C:/Users/Daniel Winklehner/Documents/ef_phi2.field"
fn = "ef_phi.field"

if fn is None:
    fd = FileDialog()
    fn = fd.get_filename()
    print(fn)

with open(fn, "rb") as infile:
    mydata = pickle.load(infile)

mypot = mydata["phi"]
mymask = mydata["mask"]
nx, ny, nz = mydata["n"]
dx, dy, dz = mydata["d"]

# This will be replaced with limits stored in file
xmin, xmax = mydata["limits"][0]
ymin, ymax = mydata["limits"][1]
zmin, zmax = mydata["limits"][2]

print("Loaded data with n = ", mydata["n"], ", d = ", mydata["d"])
print("Limits are ", xmin, xmax, ymin, ymax, zmin, zmax)

z_idx_s = [int(0.5 * nz)]
# z_idx_s = range(126)
for z_idx in z_idx_s:
    z_pos = zmin + z_idx * dz
    _pot = mypot[:, :, z_idx].T
    _msk = mymask[:, :, z_idx].T

    masked_pot = ma.masked_array(_pot, mask=_msk)

    plt.imshow(masked_pot, extent=(xmin, xmax, ymin, ymax))
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Potential (V) at z = {:.4f} m".format(z_pos))
    plt.colorbar()
    plt.show()

x_idx = int(nx / 2.0)
z_win_s_idx = 0  # 1000
z_win_s = zmin + dz * z_win_s_idx
z_win_e_idx = nz - 1
z_win_e = zmin + dz * z_win_e_idx
z_win_len = z_win_e_idx - z_win_s_idx + 1
z_win = np.linspace(z_win_s, z_win_e, z_win_len)

print("Slicing plots along x axis at [{}] ({} m)".format(x_idx, xmin + dx * x_idx))
print("Truncating plots along z axis from [{}] ({} m) to [{}] ({} m)".format(z_win_s_idx, z_win_s,
                                                                             z_win_e_idx, z_win_e))

_pot = mypot[x_idx, :, z_win_s_idx:z_win_e_idx + 1]
_msk = mymask[x_idx, :, z_win_s_idx:z_win_e_idx + 1]

masked_pot = ma.masked_array(_pot, mask=_msk)

plt.imshow(masked_pot, extent=(z_win_s, z_win_e, ymin, ymax))
plt.xlabel("z (m)")
plt.ylabel("y (m)")
plt.title("Potential (V)")
plt.colorbar()
plt.show()

print("Slicing plots along y axis at [{}] ({} m)".format(x_idx, xmin + dx * x_idx))
print("Truncating plots along z axis from [{}] ({} m) to [{}] ({} m)".format(z_win_s_idx, z_win_s,
                                                                             z_win_e_idx, z_win_e))

_pot = mypot[:, x_idx, z_win_s_idx:z_win_e_idx + 1]
_msk = mymask[:, x_idx, z_win_s_idx:z_win_e_idx + 1]

masked_pot = ma.masked_array(_pot, mask=_msk)

plt.imshow(masked_pot, extent=(z_win_s, z_win_e, xmin, xmax))
plt.xlabel("z (m)")
plt.ylabel("x (m)")
plt.title("Potential (V)")
plt.colorbar()
plt.show()

# Try masked array for numpy gradient
masked_pot = ma.masked_array(mypot, mask=mymask)

# Regular gradient, masking after
ex, ey, ez = np.gradient(mypot, dx)
ex = ma.masked_array(ex, mask=mymask)
ey = ma.masked_array(ey, mask=mymask)
ez = ma.masked_array(ez, mask=mymask)

# gradient of masked potential
exm, eym, ezm = np.gradient(masked_pot, dx)

# Ex: Slice and reduce plot size for visibility
_ex = ex[x_idx, :, z_win_s_idx:z_win_e_idx + 1]
_exm = exm[x_idx, :, z_win_s_idx:z_win_e_idx + 1]
_ex_1d = ex[x_idx, x_idx, z_win_s_idx:z_win_e_idx + 1]
_exm_1d = exm[x_idx, x_idx, z_win_s_idx:z_win_e_idx + 1]

_, ax = plt.subplots(2, 1)
p1, p2 = ax

plt.sca(p1)
plt.imshow(_ex, extent=(z_win_s, z_win_e, ymin, ymax))
plt.xlabel("z (m)")
plt.ylabel("y (m)")
plt.title("Ex (V/m)")
plt.colorbar()
plt.twinx()
plt.ylabel("Ex(z) (V/m)")
plt.plot(z_win, _ex_1d)

plt.sca(p2)
plt.imshow(_exm, extent=(z_win_s, z_win_e, ymin, ymax))
plt.xlabel("z (m)")
plt.ylabel("y (m)")
plt.title("Ex (V/m)")
plt.colorbar()
plt.twinx()
plt.ylabel("Ex(z) (V/m)")
plt.plot(z_win, _exm_1d)

plt.show()

# Ex: Slice and reduce plot size for visibility
_ey = ey[x_idx, :, z_win_s_idx:z_win_e_idx + 1]
_eym = eym[x_idx, :, z_win_s_idx:z_win_e_idx + 1]
_ey_1d = ey[x_idx, x_idx, z_win_s_idx:z_win_e_idx + 1]
_eym_1d = eym[x_idx, x_idx, z_win_s_idx:z_win_e_idx + 1]

_, ax = plt.subplots(2, 1)
p1, p2 = ax

plt.sca(p1)
plt.imshow(_ey, extent=(z_win_s, z_win_e, ymin, ymax))
plt.xlabel("z (m)")
plt.ylabel("y (m)")
plt.title("Ey (V/m)")
plt.colorbar()
plt.twinx()
plt.ylabel("Ey(z) (V/m)")
plt.plot(z_win, _ey_1d)

plt.sca(p2)
plt.imshow(_eym, extent=(z_win_s, z_win_e, ymin, ymax))
plt.xlabel("z (m)")
plt.ylabel("y (m)")
plt.title("Ey (V/m)")
plt.colorbar()
plt.twinx()
plt.ylabel("Ey(z) (V/m)")
plt.plot(z_win, _eym_1d)

plt.show()

# Ez: Slice and reduce plot size for visibility
_ez = ez[x_idx, :, z_win_s_idx:z_win_e_idx + 1]
_ezm = ezm[x_idx, :, z_win_s_idx:z_win_e_idx + 1]
_ez_1d = ez[x_idx, x_idx, z_win_s_idx:z_win_e_idx + 1]
_ezm_1d = ezm[x_idx, x_idx, z_win_s_idx:z_win_e_idx + 1]

_, ax = plt.subplots(2, 1)
p1, p2 = ax

plt.sca(p1)
plt.imshow(_ez, extent=(z_win_s, z_win_e, ymin, ymax))
plt.xlabel("z (m)")
plt.ylabel("y (m)")
plt.title("Ez (V/m)")
plt.colorbar()
plt.twinx()
plt.ylabel("Ez(z) (V/m)")
plt.plot(z_win, _ez_1d)

plt.sca(p2)
plt.imshow(_ezm, extent=(z_win_s, z_win_e, ymin, ymax))
plt.xlabel("z (m)")
plt.ylabel("y (m)")
plt.title("Ez (V/m)")
plt.colorbar()
plt.twinx()
plt.ylabel("Ez(z) (V/m)")
plt.plot(z_win, _ezm_1d)

plt.show()
