import numpy as np
from scipy import constants as const

# COPIED VERSION OF IONSPECIES FOR DANS PYMODULES AS DANS PYMODULES IS UNDERGOING LARGE STRUCTURAL CHANGES
# CAN BE REPLACED LATER


__author__ = "Daniel Winklehner, Philip Weigel"
__doc__ = "Simple class to hold and calculate particle data like energy, mass, charge, etc."

# Initialize some global constants
amu = const.value("atomic mass constant energy equivalent in MeV")
echarge = const.value("elementary charge")
emass_mev = const.value("electron mass energy equivalent in MeV")
clight = const.value("speed of light in vacuum")

# TODO: How to handle negative charge? -DW

presets = {'proton': {'latex_label': r"$\mathrm{p}^+$",
                      'mass_mev': const.value('proton mass energy equivalent in MeV'),
                      'a': 1.00727647,
                      'z': 1.0,
                      'q': 1.0},
           'H_1-': {'latex_label': r"$\mathrm{H}^1-$",
                    'mass_mev': 1.00837361135 * amu,
                    'a': 1.00837361135,
                    'z': 1.0,
                    'q': 1.0},
           'electron': {'latex_label': r"$\mathrm{e}^-$",
                        'mass_mev': const.value('electron mass energy equivalent in MeV'),
                        'a': const.value('electron mass energy equivalent in MeV')/amu,
                        'z': 0.0,
                        'q': 1.0},
           'H2_1+': {'latex_label': r"$\mathrm{H}_2^+$",
                     'mass_mev': 1876.634889,
                     'a': 2.01510,
                     'z': 2.0,
                     'q': 1.0},
           '4He_2+': {'latex_label': r"$^4\mathrm{He}^{2+}$",
                      'mass_mev': 3727.379378,
                      'a': 4.0026022,
                      'z': 2.0,
                      'q': 2.0}}


class IonSpecies(object):

    def __init__(self,
                 name,
                 energy_mev,
                 label=None,
                 latex_label=None,
                 mass_mev=None,
                 a=None,
                 z=None,
                 q=None,
                 current=1e-36,
                 debug=False):

        """
        Simple ion species class that holds data and can calculate some basic values like rigidity and emergy.
        :param name: Name of the species, can be one of the presets:
            'protons'
            'electrons'
            'H2_1+'
            '4He_2+'
        if it is not a preset, the following four have to be defined as well:
        :param label: A plain text label
        :param latex_label: A text label for plotting, can be in latex shorthand, defaults to name.
        :param mass_mev:
        :param a:
        :param z:
        :param q:
        :param current: beam current (initialized as very small number if not given)
        :param energy_mev: Energy in MeV/c^2 has to always be provided.
        """

        # Check if user wants a preset ion species:
        if name in presets.keys():

            species = presets[name]

            mass_mev = species["mass_mev"]
            if latex_label is None:
                latex_label = species["latex_label"]
            z = species["z"]
            a = species["a"]
            q = species["q"]

            if debug:
                print("Using preset ion species '{}' with label '{}':".format(name, label))
                print("m_0 = {:.2f} MeV/c^2, q = {:.1f} e, E_kin = {:.2f} MeV/amu". format(mass_mev, q, energy_mev))

        # If not, check for missing initial values
        else:

            init_values = [mass_mev, a, z, q]

            if None in init_values:

                print("Sorry, ion species {} was initialized with missing values ('None')!".format(name))
                print("mass_mev = {}, a = {}, z = {}, q = {}". format(mass_mev, a, z, q))

                exit(1)

            else:

                if debug:
                    print("User defined ion species {}:".format(name))
                    print("m_0 = {:.2f} MeV/c^2, q = {:.1f} e, E_kin = {:.2f} MeV".format(mass_mev, q, energy_mev))

        # Initialize values (default for a proton)
        self._name = name
        if latex_label is None:
            self._latex_label = name
        else:
            self._latex_label = latex_label        # A label for this species
        if label is None:
            self._label = name
        else:
            self._label = label  # A label for this species
        self._mass_mev = mass_mev      # Rest Mass (MeV/c^2)
        self._a = a                    # Mass number A of the ion (amu)
        self._z = z                    # Proton number Z of the ion (unitless)
        self._energy_mev = energy_mev  # Initial kinetic energy (MeV/amu)
        self._q = q                    # charge state
        self._current = current        # beam current in A

        # Init other variables, calculate later
        self._gamma = 0.0
        self._beta = 0.0
        self._b_rho = 0.0
        self._mass_kg = 0.0

        # Calculate mass of the particle in kg
        self._mass_kg = self._mass_mev * echarge * 1.0e6 / clight**2.0

        self.calculate_from_energy_mev()

    def __str__(self):
        return "Ion Species {} with label {}:\n" \
               "M_0 = {} MeV/c^2, q = {}, B-Rho = {},\n" \
               "E_kin = {} MeV/amu, beta = {}, gamma = {}\n" \
               "(For OPERA: M_0 = {} * M_electron)".format(self._name,
                                                           self._label,
                                                           self._mass_mev,
                                                           self._q,
                                                           self._b_rho,
                                                           self._energy_mev,
                                                           self._beta,
                                                           self._gamma,
                                                           self._mass_mev / emass_mev)

    def current(self):
        """
        :return: beam current in A
        """
        return self._current

    def cyclotron_frequency(self, b):
        """
        :param b: B-Field in T
        :return: cyclotron frequency in Hz
        """
        return self._q * echarge * b / self._gamma / self._mass_kg / 2.0 / np.pi

    def mass_mev(self):
        return self._mass_mev

    def mass_kg(self):
        return self._mass_kg

    def mass_opera(self):
        return self.mass_mev() / emass_mev

    def energy_mev(self):
        return self._energy_mev

    def q_over_a(self):
        """
        :return: charge state - to - mass number ratio (unitless) 
        """
        return np.abs(self._q / self._a)

    def q_over_m(self):
        """
        :return: charge - to - mass ratio (C/kg) 
        """
        return self._q * echarge / self._mass_kg

    def total_kinetic_energy_mev(self):
        return self._energy_mev * self._a

    def total_kinetic_energy_ev(self):
        return self._energy_mev * self._a * 1.0e6

    def v_m_per_s(self):
        return self._beta * clight

    def v_cm_per_s(self):
        return self._beta * clight * 1.0e2

    def label(self):
        return self._label

    def name(self):
        return self._name

    def b_rho(self):
        return self._b_rho

    def gamma(self):
        return self._gamma

    def beta(self):
        return self._beta

    def q(self):
        return self._q

    def a(self):
        return self._a

    def z(self):
        return self._z

    def calculate_from_momentum_betagamma(self, momentum_bega=None, nonrelativistic=False):

        if nonrelativistic:
            self._gamma = 1.0
            self._beta = momentum_bega
            self._energy_mev = (self._beta * clight)**2.0 * self._mass_kg * 0.5 / echarge * 1e-6 / self._a
            self._b_rho = self._beta * self._gamma * self._mass_mev * 1.0e6 / (self._q * clight)
            return 0

        if momentum_bega is not None:
            self._gamma = np.sqrt(momentum_bega**2.0 + 1.0)
            self._beta = momentum_bega / self._gamma

        self._energy_mev = (self._gamma - 1.0) * self._mass_mev / self._a

        # Calculate B-rho of the particle
        self._b_rho = self._beta * self._gamma * self._mass_mev * 1.0e6 / (self._q * clight)

    def calculate_from_velocity_beta(self, beta=None, nonrelativistic=False):

        if nonrelativistic:
            self._gamma = 1.0
            self._beta = beta
            self._energy_mev = (self._beta * clight)**2.0 * self._mass_kg * 0.5 / echarge * 1e-6 / self._a
            self._b_rho = self._beta * self._gamma * self._mass_mev * 1.0e6 / (self._q * clight)
            return 0

        if beta is not None:
            self._beta = beta

        self._gamma = np.sqrt(1.0 / (1.0 - self._beta ** 2.0))
        self._energy_mev = (self._gamma - 1.0) * self._mass_mev / self._a

        # Calculate B-rho of the particle
        self._b_rho = self._beta * self._gamma * self._mass_mev * 1.0e6 / (self._q * clight)

    def calculate_from_energy_mev(self, energy_mev=None, nonrelativistic=False):

        if nonrelativistic:
            self._energy_mev = energy_mev
            self._gamma = 1.0
            self._beta = np.sqrt(self._energy_mev * self._a * 1e6 * echarge * 2.0 / self._mass_kg) / clight
            self._b_rho = self._beta * self._gamma * self._mass_mev * 1.0e6 / (self._q * clight)
            return 0

        if energy_mev is not None:
            self._energy_mev = energy_mev

        # Calculate relativistic parameters
        self._gamma = self._energy_mev * self._a / self._mass_mev + 1.0
        self._beta = np.sqrt(1.0 - self._gamma**(-2.0))

        # Calculate B-rho of the particle
        self._b_rho = self._beta * self._gamma * self._mass_mev * 1.0e6 / (self._q * clight)

    def calculate_from_b_rho(self, b_rho=None, nonrelativistic=False):

        if nonrelativistic:
            self._b_rho = b_rho
            self._gamma = 1.0
            self._beta = self._b_rho * self._q * clight * 1.0e-6 / self._mass_mev
            self._energy_mev = (self._beta * clight)**2.0 * self._mass_kg * 0.5 / echarge * 1e-6 / self._a
            return 0

        if b_rho is not None:
            self._b_rho = b_rho

        # Calculate relativistic parameters from b_rho
        betagamma = self._b_rho * self._q * clight * 1.0e-6 / self._mass_mev
        self._gamma = np.sqrt(betagamma**2.0 + 1.0)
        self._beta = betagamma / self._gamma

        # Calculate energy_mev from b_rho (cave: energy_mev is per nucleon)
        self._energy_mev = self._mass_mev * (self._gamma - 1.0) / self._a


if __name__ == '__main__':

    print("Testing IonSpecies class:")

    ion = IonSpecies(name="H2_1+", energy_mev=0.03499)

    print(ion)
    print("Accelerating Voltage = {} V".format(1.0e6 * ion.energy_mev() * ion.a() / ion.q()))
