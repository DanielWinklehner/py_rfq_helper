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

print(mypot.shape)

z_idx = 50
_pot = mypot[:, :, z_idx].T
_msk = mymask[:, :, z_idx].T

masked_pot = ma.masked_array(_pot, mask=_msk)

plt.imshow(masked_pot, extent=(-0.02, 0.02, -0.02, 0.02))
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Potential (V)")
plt.colorbar()
plt.show()

x_idx = 15

_pot = mypot[x_idx, :, 534:]
_msk = mymask[x_idx, :, 534:]

masked_pot = ma.masked_array(_pot, mask=_msk)

# plt.imshow(masked_pot, extent=(-0.1, 1.35, -0.02, 0.02))
plt.imshow(masked_pot, extent=(1.0, 1.35, -0.02, 0.02))
plt.xlabel("z (m)")
plt.ylabel("y (m)")
plt.title("Potential (V)")
plt.colorbar()
plt.show()

# Try masked array for numpy gradient
masked_pot = ma.masked_array(mypot, mask=mymask)

# Regulargradient, masking after
ex, ey, ez = np.gradient(mypot, 0.002)
ex = ma.masked_array(ex, mask=mymask)
ey = ma.masked_array(ey, mask=mymask)
ez = ma.masked_array(ez, mask=mymask)

# gradient of masked potential
exm, eym, ezm = np.gradient(masked_pot, 0.002)

# Ex: Slice and reduce plot size for visibility
_ex = ex[x_idx, :, 534:]
_exm = exm[x_idx, :, 534:]
_ex_1d = ex[x_idx, x_idx, 534:]
_exm_1d = exm[x_idx, x_idx, 534:]

fig, ax = plt.subplots(2, 1)
p1, p2 = ax

plt.sca(p1)
plt.imshow(_ex, extent=(1.0, 1.35, -0.02, 0.02))
plt.xlabel("z (m)")
plt.ylabel("y (m)")
plt.title("Ex (V/m)")
plt.colorbar()
plt.twinx()
plt.ylabel("Ex(z) (V/m)")
plt.plot(np.linspace(1.0, 1.35, 192), _ex_1d)

plt.sca(p2)
plt.imshow(_exm, extent=(1.0, 1.35, -0.02, 0.02))
plt.xlabel("z (m)")
plt.ylabel("y (m)")
plt.title("Ex (V/m)")
plt.colorbar()
plt.twinx()
plt.ylabel("Ex(z) (V/m)")
plt.plot(np.linspace(1.0, 1.35, 192), _exm_1d)

plt.show()

# Ex: Slice and reduce plot size for visibility
_ey = ey[x_idx, :, 534:]
_eym = eym[x_idx, :, 534:]
_ey_1d = ey[x_idx, x_idx, 534:]
_eym_1d = eym[x_idx, x_idx, 534:]

fig, ax = plt.subplots(2, 1)
p1, p2 = ax

plt.sca(p1)
plt.imshow(_ey, extent=(1.0, 1.35, -0.02, 0.02))
plt.xlabel("z (m)")
plt.ylabel("y (m)")
plt.title("Ey (V/m)")
plt.colorbar()
plt.twinx()
plt.ylabel("Ey(z) (V/m)")
plt.plot(np.linspace(1.0, 1.35, 192), _ey_1d)

plt.sca(p2)
plt.imshow(_eym, extent=(1.0, 1.35, -0.02, 0.02))
plt.xlabel("z (m)")
plt.ylabel("y (m)")
plt.title("Ey (V/m)")
plt.colorbar()
plt.twinx()
plt.ylabel("Ey(z) (V/m)")
plt.plot(np.linspace(1.0, 1.35, 192), _eym_1d)

plt.show()

# Ez: Slice and reduce plot size for visibility
_ez = ez[x_idx, :, 534:]
_ezm = ezm[x_idx, :, 534:]
_ez_1d = ez[x_idx, x_idx, 534:]
_ezm_1d = ezm[x_idx, x_idx, 534:]

fig, ax = plt.subplots(2, 1)
p1, p2 = ax

plt.sca(p1)
plt.imshow(_ez, extent=(1.0, 1.35, -0.02, 0.02))
plt.xlabel("z (m)")
plt.ylabel("y (m)")
plt.title("Ez (V/m)")
plt.colorbar()
plt.twinx()
plt.ylabel("Ez(z) (V/m)")
plt.plot(np.linspace(1.0, 1.35, 192), _ez_1d)

plt.sca(p2)
plt.imshow(_ezm, extent=(1.0, 1.35, -0.02, 0.02))
plt.xlabel("z (m)")
plt.ylabel("y (m)")
plt.title("Ez (V/m)")
plt.colorbar()
plt.twinx()
plt.ylabel("Ez(z) (V/m)")
plt.plot(np.linspace(1.0, 1.35, 192), _ezm_1d)

plt.show()

exit()

pot_xy0 = mypot[10, 10, :]
pot_xy0 = smooth(pot_xy0)[5:-5]

print(pot_xy0.shape)

z_orig = np.linspace(-0.1, 1.35, 726, endpoint=True)
print(z_orig)
pot_itp = interp1d(z_orig,
                   pot_xy0, kind='cubic')

z_vals = np.linspace(-0.1, 1.35, 10000)
pot_vals = pot_itp(z_vals)

plt.plot(z_vals, pot_vals)
plt.xlabel("z (m)")
plt.ylabel("Potential (V)")
# plt.title("Potential (V)")
# plt.colorbar()
plt.show()

plt.plot(z_vals, np.gradient(pot_vals, 0.001))
plt.xlabel("z (m)")
plt.ylabel("Ez (V/m)")
plt.show()
