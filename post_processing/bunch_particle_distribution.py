import numpy as np
from scipy import constants as const
import h5py
import matplotlib.pyplot as plt
from temp_particles import IonSpecies
from dans_pymodules import FileDialog, ParticleDistribution
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import pickle
import os


__author__ = "Jared Hwang"
__doc__ = """Inherited version of the Dan's pymodules particle distribution"""

# Some constants
clight = const.value("speed of light in vacuum")  # (m/s)
amu_kg = const.value("atomic mass constant")  # (kg)
echarge = const.value("elementary charge")


class BunchParticleDistribution(ParticleDistribution):
    def __init__(self, rfq_freq, ion=IonSpecies('proton', 1.0),
                 x=np.zeros(1),
                 y=np.zeros(1),
                 z=np.zeros(1),
                 vx=np.zeros(1),
                 vy=np.zeros(1),
                 vz=np.zeros(1),
                 debug=False):
        self._rfq_freq = rfq_freq

        super().__init__(ion=IonSpecies('proton', 1.0),
                 x=x,
                 y=y,
                 z=z,
                 vx=vx,
                 vy=vy,
                 vz=vz,
                 debug=False)

    def calculate_emittances(self):
        # calculate x and y prime (rad)
        xp = self.vx / self.vz
        yp = self.vy / self.vz

        # calculate mean velocity (m/s)
        v = np.array(np.sqrt(self.vx ** 2.0 + self.vy ** 2.0 + self.vz ** 2.0))
        v_mean = np.mean(v)
        # v_std = np.std(v)

        # Relativistic parameters per particle
        beta_rel = v / clight
        gamma_rel = 1.0 / np.array(np.sqrt(1.0 - beta_rel ** 2.0))
        # betagamma_rel = beta_rel * gamma_rel

        # Kinetic Energy
        e_kin = (gamma_rel - 1.0) * self.ion.mass_mev()  # per particle (MeV)
        e_kin_mean = np.mean(e_kin)
        # e_kin_std = np.std(e_kin)

        # Re-calculate the dependent values for the ion object (includes beta and gamma)
        # Cave: function expects energy in MeV/amu
        self.ion.calculate_from_energy_mev(e_kin_mean / self.ion.a())

        # Mean Values:
        x_mean = np.mean(self.x)  # (m)
        y_mean = np.mean(self.y)  # (m)
        xp_mean = np.mean(xp)  # (rad)
        yp_mean = np.mean(yp)  # (rad)
        # xxp_mean = np.mean(self.x * xp)  # (m * rad)
        # yyp_mean = np.mean(self.y * yp)  # (m * rad)
        # xyp_mean = np.mean(self.x * yp)  # (m * rad)
        # yxp_mean = np.mean(self.y * xp)  # (m * rad)

        # Mean Square Values:
        xsq_mean = np.mean(self.x ** 2.0)  # (m^2)
        ysq_mean = np.mean(self.y ** 2.0)  # (m^2)
        xpsq_mean = np.mean(xp ** 2.0)  # (rad^2)
        ypsq_mean = np.mean(yp ** 2.0)  # (rad^2)
        # xxpsq_mean = np.mean((self.x * xp) ** 2.0)  # ((m * rad)^2)
        # yypsq_mean = np.mean((self.y * yp) ** 2.0)  # ((m * rad)^2)
        # xypsq_mean = np.mean((self.x * yp) ** 2.0)  # ((m * rad)^2)
        # yxpsq_mean = np.mean((self.y * xp) ** 2.0)  # ((m * rad)^2)

        # Calculate standard deviations
        x_std = np.sqrt(xsq_mean - x_mean ** 2.0)  # (m)
        y_std = np.sqrt(ysq_mean - y_mean ** 2.0)  # (m)
        xp_std = np.sqrt(xpsq_mean - xp_mean ** 2)  # (rad)
        yp_std = np.sqrt(ypsq_mean - yp_mean ** 2)  # (rad)


        self.xm = 2.0 * x_std
        self.ym = 2.0 * y_std
        self.xpm = 2.0 * xp_std
        self.ypm = 2.0 * yp_std

        xxp_std = np.mean((self.x - x_mean) * (xp - xp_mean))  # (m * rad)
        yyp_std = np.mean((self.y - y_mean) * (yp - yp_mean))  # (m * rad)
        xyp_std = np.mean((self.x - x_mean) * (yp - yp_mean))  # (m * rad)
        yxp_std = np.mean((self.y - y_mean) * (xp - xp_mean))  # (m * rad)

        # Beam edges (m)
        x_min_1rms = x_mean - x_std
        x_min_2rms = x_mean - 2.0 * x_std
        x_min_full = np.min(self.x)

        x_max_1rms = x_mean + x_std
        x_max_2rms = x_mean + 2.0 * x_std
        x_max_full = np.max(self.x)

        y_min_1rms = y_mean - y_std
        y_min_2rms = y_mean - 2.0 * y_std
        y_min_full = np.min(self.y)

        y_max_1rms = y_mean + y_std
        y_max_2rms = y_mean + 2.0 * y_std
        y_max_full = np.max(self.y)

        # RMS Emittances (m-rad)
        e_xxp_1rms = np.sqrt((x_std * xp_std) ** 2.0 - xxp_std ** 2.0)
        e_yyp_1rms = np.sqrt((y_std * yp_std) ** 2.0 - yyp_std ** 2.0)
        e_xyp_1rms = np.sqrt((x_std * yp_std) ** 2.0 - xyp_std ** 2.0)
        e_yxp_1rms = np.sqrt((y_std * xp_std) ** 2.0 - yxp_std ** 2.0)



        # Normalized RMS Emittances (m-rad)
        en_xxp_1rms = e_xxp_1rms * self.ion.beta() * self.ion.gamma()
        en_yyp_1rms = e_yyp_1rms * self.ion.beta() * self.ion.gamma()
        en_xyp_1rms = e_xyp_1rms * self.ion.beta() * self.ion.gamma()
        en_yxp_1rms = e_yxp_1rms * self.ion.beta() * self.ion.gamma()

        # 4-RMS Emittances (pi-m-rad)
        e_xxp_4rms = 4.0 * e_xxp_1rms
        e_yyp_4rms = 4.0 * e_yyp_1rms
        # en_xxp_4rms = 4.0 * en_xxp_1rms  # normalized
        # en_yyp_4rms = 4.0 * en_yyp_1rms  # normalized

        # Twiss Parameters
        twiss_beta_x = (x_std ** 2.0) / e_xxp_1rms
        twiss_gamma_x = (xp_std ** 2.0) / e_xxp_1rms


        # Longitudinal Emittance
        z_beta_rel = self.vz / clight
        z_gamma_rel = 1.0 / np.array(np.sqrt(1.0 - z_beta_rel ** 2.0))
        
        z_ekin = (z_gamma_rel - 1.0) * self.ion.mass_mev() # Per particle
        z_ekin_std = np.std(z_ekin)

        wavelength = np.mean(self.vz) / self._rfq_freq
        
        z_phase = 360 * self.z / wavelength
        z_phase_std = np.std(z_phase)

        EZP_std = np.mean((z_ekin - np.mean(z_ekin)) * (z_phase - np.mean(z_phase)))

        e_elongphase = np.sqrt((z_ekin_std * z_phase_std)**2 - (EZP_std)**2)
        en_elongphase = self.ion.beta() * self.ion.gamma() * e_elongphase


        if xxp_std < 0:

            twiss_alpha_x = np.sqrt(twiss_beta_x * twiss_gamma_x - 1.0)

        else:

            twiss_alpha_x = -np.sqrt(twiss_beta_x * twiss_gamma_x - 1.0)

        twiss_beta_y = (y_std ** 2.0) / e_yyp_1rms
        twiss_gamma_y = (yp_std ** 2.0) / e_yyp_1rms

        if yyp_std < 0:

            twiss_alpha_y = np.sqrt(twiss_beta_y * twiss_gamma_y - 1.0)

        else:

            twiss_alpha_y = -np.sqrt(twiss_beta_y * twiss_gamma_y - 1.0)

        # Number of particles inside 4-RMS emittance ellipse
        e_xxp_4rms_includes = len(np.where(
            twiss_gamma_x * self.x ** 2.0
            + 2.0 * twiss_alpha_x * self.x * xp
            + twiss_beta_x * xp ** 2.0 < e_xxp_4rms)[0])
        e_yyp_4rms_includes = len(np.where(
            twiss_gamma_y * self.y ** 2.0
            + 2.0 * twiss_alpha_y * self.y * yp
            + twiss_beta_y * yp ** 2.0 < e_yyp_4rms)[0])

        e_xxp_4rms_includes_perc = 100.0 * e_xxp_4rms_includes / self.numpart  # percent
        e_yyp_4rms_includes_perc = 100.0 * e_yyp_4rms_includes / self.numpart  # percent

        # Maximum emittance by using the RMS Twiss parameters and the maximum values of x, x', y and y'
        e_xxp_full = twiss_gamma_x * np.max(self.x) ** 2.0 + 2.0 * twiss_alpha_x * np.max(self.x) * np.max(
            xp) + twiss_beta_x * np.max(xp) ** 2.0
        e_yyp_full = twiss_gamma_y * np.max(self.y) ** 2.0 + 2.0 * twiss_alpha_y * np.max(self.y) * np.max(
            yp) + twiss_beta_y * np.max(yp) ** 2.0

        # --- Create a structured array for better postprocessing --- #
        dtype = [('name', np.unicode_, 20), ('value', float), ('unit', np.unicode_, 10)]

        # TODO: Think about units!
        # actual entries
        values = [('# of particles', self.numpart, ''),
                  ('mean velocity', v_mean, 'm/s'),
                  ('mean energy', e_kin_mean, 'MeV'),
                  ('x-dia 1-rms', 2000.0 * x_std, 'mm'),
                  ('y-dia 1-rms', 2000.0 * y_std, 'mm'),
                  ('x-dia 2-rms', 4000.0 * x_std, 'mm'),
                  ('y-dia 2-rms', 4000.0 * y_std, 'mm'),
                  ('x-dia max', 1000.0 * (x_max_full - x_min_full), 'mm'),
                  ('y-dia max', 1000.0 * (y_max_full - y_min_full), 'mm'),
                  ('x centroid offset', 1000.0 * x_mean, 'mm'),
                  ('y centroid offset', 1000.0 * y_mean, 'mm'),
                  ('xp centroid offset', 1000.0 * xp_mean, 'mm'),
                  ('yp centroid offset', 1000.0 * yp_mean, 'mm'),
                  ('e xxp 1-rms', 1.0e6 * e_xxp_1rms, 'mm-mrad'),
                  ('e yyp 1-rms', 1.0e6 * e_yyp_1rms, 'mm-mrad'),
                  ('e xyp 1-rms', 1.0e6 * e_xyp_1rms, 'mm-mrad'),
                  ('e yxp 1-rms', 1.0e6 * e_yxp_1rms, 'mm-mrad'),
                  ('e long Ekin phase', e_elongphase, 'MeV-rad'),
                  ('e norm xxp 1-rms', 1.0e6 * en_xxp_1rms, 'mm-mrad'),
                  ('e norm yyp 1-rms', 1.0e6 * en_yyp_1rms, 'mm-mrad'),
                  ('e norm xyp 1-rms', 1.0e6 * en_xyp_1rms, 'mm-mrad'),
                  ('e norm yxp 1-rms', 1.0e6 * en_yxp_1rms, 'mm-mrad'),
                  ('e norm long Ekin phase', en_elongphase, 'MeV-rad'),
                  ('e xxp 4-rms', 4.0e6 * e_xxp_1rms, 'pi-mm-mrad'),
                  ('e yyp 4-rms', 4.0e6 * e_yyp_1rms, 'pi-mm-mrad'),
                  ('e xyp 4-rms', 4.0e6 * e_xyp_1rms, 'pi-mm-mrad'),
                  ('e yxp 4-rms', 4.0e6 * e_yxp_1rms, 'pi-mm-mrad'),
                  ('e norm xxp 4-rms', 4.0e6 * en_xxp_1rms, 'pi-mm-mrad'),
                  ('e norm yyp 4-rms', 4.0e6 * en_yyp_1rms, 'pi-mm-mrad'),
                  ('e norm xyp 4-rms', 4.0e6 * en_xyp_1rms, 'pi-mm-mrad'),
                  ('e norm yxp 4-rms', 4.0e6 * en_yxp_1rms, 'pi-mm-mrad'),
                  ('e xxp 4rms includes', e_xxp_4rms_includes_perc, 'percent'),
                  ('e yyp 4rms includes', e_yyp_4rms_includes_perc, 'percent'),
                  ('Twiss Alpha X', twiss_alpha_x, ''),
                  ('Twiss Beta X', twiss_beta_x, 'mm/mrad'),
                  ('Twiss Gamma X', twiss_gamma_x, 'mrad/mm'),
                  ('Twiss Alpha Y', twiss_alpha_y, ''),
                  ('Twiss Beta Y', twiss_beta_y, 'mm/mrad'),
                  ('Twiss Gamma Y', twiss_gamma_y, 'mrad/mm')]

        # put together in structured array
        emi_array = np.array(values, dtype=dtype)

        summary = "--- RMS Emittances, Twiss Parameters and Beamsizes ---\n\n"

        for element in emi_array:

            summary += element[0].ljust(22) + "= " + ("%1.5e " % (element[1])).ljust(14) + element[2] + "\n"

        if self.debug:

            print(summary)

        emi = {"Np": self.numpart,
               "VMean": v_mean,
               "TMean": e_kin_mean,
               "EXXPRMS": e_xxp_1rms,
               "EYYPRMS": e_yyp_1rms,
               "EXYPRMS": e_xyp_1rms,
               "EYXPRMS": e_yxp_1rms,
               "EZP": e_elongphase,
               "EXXPNRMS": en_xxp_1rms,
               "EYYPNRMS": en_yyp_1rms,
               "EXYPNRMS": en_xyp_1rms,
               "EYXPNRMS": en_yxp_1rms,
               "EZPN": en_elongphase,
               "X4RMSINCLUDE": e_xxp_4rms_includes_perc,
               "Y4RMSINCLUDE": e_yyp_4rms_includes_perc,
               "XMin1rms": x_min_1rms,
               "XMax1rms": x_max_1rms,
               "YMin1rms": y_min_1rms,
               "YMax1rms": y_max_1rms,
               "XMin2rms": x_min_2rms,
               "XMax2rms": x_max_2rms,
               "YMin2rms": y_min_2rms,
               "YMax2rms": y_max_2rms,
               "XMinFull": x_min_full,
               "XMaxFull": x_max_full,
               "YMinFull": y_min_full,
               "YMaxFull": y_max_full,
               "XAlpha": twiss_alpha_x,
               "XBeta": twiss_beta_x,
               "XGamma": twiss_gamma_x,
               "YAlpha": twiss_alpha_y,
               "YBeta": twiss_beta_y,
               "YGamma": twiss_gamma_y,
               "XMean": x_mean,
               "XPMean": xp_mean,
               "YMean": y_mean,
               "YPMean": yp_mean,
               "EXXPMax": e_xxp_full,
               "EYYPMax": e_yyp_full,
               "Summary": summary,
               "BetaGamma": beta_rel * gamma_rel}

        results = {'data': emi_array,
                   'summary': summary,
                   'raw_data': emi
                   }

        return results