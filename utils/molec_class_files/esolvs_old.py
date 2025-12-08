import numpy as np
import unyt as u
from thermo import *


class EsolvsConstants:
    """Experimental data and other constants"""

    def __init__(
        self,
        name,
        mol_wt,
        Tc,
        rhoc,
        n_atoms,
        smiles_str,
        param_names,
        gaff_params,
        bnds_sig,
        bnds_eps,
        expt_liq_density,
        expt_surf_tens,
        expt_Pvap,
        expt_Hvap,
        expt_vap_density,
        expt_diff_coeff = {},
        uncertainty={},
    ):
        """Initialize the class with experimental data"""

        # FF Properties
        self.name = name
        self.param_names = param_names
        self.gaff_params = gaff_params
        self.bnds_sig = bnds_sig
        self.bnds_eps = bnds_eps
        self.bounds_sig, self.bounds_eps = self._get_sig_eps_bnds()
        # Experimental data
        self.molecular_weight = mol_wt
        self.expt_Tc = Tc
        self.n_atoms = n_atoms
        self.smiles_str = smiles_str
        self.expt_rhoc = self.chem_crit_dens() if rhoc is None else rhoc
        self.expt_liq_density = expt_liq_density
        self.expt_surf_tens = expt_surf_tens
        self.expt_Pvap = expt_Pvap
        self.expt_Hvap = expt_Hvap
        self.expt_vap_density = self.get_vap_density(expt_vap_density)
        self.expt_diff_coeff = expt_diff_coeff
        self.uncertainty = self.process_unc(uncertainty)

        # Check that the experimental data keys are the same for density, Pvap, and surface tension
        assert (
            self.expt_liq_density.keys()
            == self.expt_vap_density.keys()
            == self.expt_surf_tens.keys()
            == self.expt_Pvap.keys()
        )

    def A_kJmol_to_nm_Kkb(self, dict_or_array):

        """Convert nm and kJ/mol to Angstroms and kJ/mol"""
        if isinstance(dict_or_array, np.ndarray):
            data = np.array(data, dtype=float)
            n = len(data) // 2
            sigmas = float((data[:n] * u.Angstrom).in_units(u.nm).value)
            epsilons = float((data[n:] * (u.K * u.kb)).in_units(u.kJ / u.mol).value)
            dict_arr = np.concatenate([sigmas, epsilons])
        else:
            dict_arr = {}
            for key in dict_or_array.keys():
                if key.startswith("sigma_"):
                    dict_arr[key] = float((dict_or_array[key] * u.Angstrom).in_units(u.nm).value)
                elif key.startswith("epsilon_"):
                    dict_arr[key] = float((dict_or_array[key] * u.K *u .kb).in_units(u.kJ / u.mol).value)
        return dict_arr
        
    def chem_crit_dens(self):
        # Create a Chemical object for the compound
        constants = Chemical(self.smiles_str)
        rhoc = constants.rhoc  # kg/m^3
        return rhoc

    def pr_vap_dens(self, T):
        P = float(
            (self.expt_Pvap[T] * u.bar).in_units("Pa")
        )  # Pressure in Pascals (adjust as needed)
        # Create a Chemical object for the compound
        constants = Chemical(self.smiles_str)
        eos = PR(constants.Tc, constants.Pc, constants.omega, T=T, P=P)
        # Use the Peng-Robinson EOS to calculate the vapor density
        molar_dens_g = eos.rho_g  # mol/m^3
        vapor_density = constants.MW * molar_dens_g / 1000  # Convert to kg/m^3
        return vapor_density

    # def ig_vap_dens(self, T):
    #     unit_T = T * u.K
    #     mw = self.molecular_weight * (u.g / u.mol)
    #     R = 8.314 * (u.Pa * u.m**3) / (u.mol * u.K)
    #     vapor_density = float(
    #         ((self.expt_Pvap[T] * u.bar * mw) / (R * unit_T)).in_units("kg/m**3").value
    #     )
    #     return vapor_density

    def get_vap_density(self, expt_vap_density=None):
        """Get the vapor density from the experimental data"""
        if expt_vap_density is not None:
            expt_vap_density = expt_vap_density
        else:
            expt_vap_density = {}
            for T in self.expt_Pvap.keys():
                expt_vap_density[T] = self.pr_vap_dens(T)
        return expt_vap_density

    def process_unc(self, uncertainty):
        for key in [
            "expt_liq_density",
            "expt_vap_density",
            "expt_surf_tens",
            "expt_Pvap",
            "expt_diff_coeff",
            "expt_Hvap",
        ]:
            if key not in uncertainty.keys():
                uncertainty[key] = 0.02 if key == "expt_vap_density" else 0.10
        return uncertainty

    def _get_sig_eps_bnds(self):
        """Bounds on sigma and epsilon in units of nm and kJ/mol"""

        bounds_sigma = (np.asarray(self.bnds_sig) * u.Angstrom).in_units(u.nm).value
        bounds_epsilon = (
            (np.asarray(self.bnds_eps) * u.K * u.kb).in_units("kJ/mol").value
        )

        return bounds_sigma, bounds_epsilon

    @property
    def param_bounds(self):
        """Bounds on sigma and epsilon in units of nm and kJ/mol"""

        bounds_sigma = (np.asarray(self.bnds_sig) * u.Angstrom).in_units(u.nm).value

        bounds_epsilon = (
            (np.asarray(self.bnds_eps) * u.K * u.kb).in_units("kJ/mol").value
        )

        bounds = np.vstack((bounds_sigma, bounds_epsilon))

        return bounds

    @property
    def n_params(self):
        """Number of adjustable parameters"""
        return len(self.param_names)

    def temperature_bounds(self, prop_name="expt_surf_tens"):
        """Bounds on temperature in units of K"""

        # Get the minimum and maximum temperature from the experimental data
        if prop_name == "expt_liq_density":
            prop = self.expt_liq_density
        elif prop_name == "expt_vap_density":
            prop = self.expt_vap_density
        elif prop_name == "expt_surf_tens":
            prop = self.expt_surf_tens
        elif prop_name == "expt_Pvap":
            prop = self.expt_Pvap
        elif prop_name == "expt_Hvap":
            prop = self.expt_Hvap
        elif prop_name == "expt_diff_coeff":
            prop = self.expt_diff_coeff
        else:
            raise ValueError("Invalid property name")

        # Get the temperature bounds
        lower_bound = np.min(list(prop.keys()))
        upper_bound = np.max(list(prop.keys()))
        bounds = np.asarray([lower_bound, upper_bound], dtype=np.float32)
        return bounds

    @property
    def liq_density_bounds(self):
        """Bounds on liquid density in units of kg/m^3"""

        lower_bound = np.min(list(self.expt_liq_density.values()))
        upper_bound = np.max(list(self.expt_liq_density.values()))
        bounds = np.asarray([lower_bound, upper_bound], dtype=np.float32)
        return bounds
    
    @property
    def vap_density_bounds(self):
        """Bounds on vapor density in units of kg/m^3"""

        lower_bound = np.min(list(self.expt_vap_density.values()))
        upper_bound = np.max(list(self.expt_vap_density.values()))
        bounds = np.asarray([lower_bound, upper_bound], dtype=np.float32)
        return bounds

    @property
    def surf_tens_bounds(self):
        """Bounds on vapor density in units of kg/m^3"""

        lower_bound = np.min(list(self.expt_surf_tens.values()))
        upper_bound = np.max(list(self.expt_surf_tens.values()))
        bounds = np.asarray([lower_bound, upper_bound], dtype=np.float32)
        return bounds

    @property
    def Pvap_bounds(self):
        """Bounds on vapor pressure in units of bar"""

        lower_bound = np.min(list(self.expt_Pvap.values()))
        upper_bound = np.max(list(self.expt_Pvap.values()))
        bounds = np.asarray([lower_bound, upper_bound], dtype=np.float32)
        return bounds

    @property
    def Hvap_bounds(self):
        """Bounds on enthaply of vaporization in units of kJ/kg"""

        lower_bound = np.min(list(self.expt_Hvap.values()))
        upper_bound = np.max(list(self.expt_Hvap.values()))
        bounds = np.asarray([lower_bound, upper_bound], dtype=np.float32)
        return bounds
    
    @property
    def diff_coeff_bounds(self):
        """Bounds on diffusion coefficient in units of m^2/s"""
        if self.expt_diff_coeff:
            lower_bound = np.min(list(self.expt_diff_coeff.values()))
            upper_bound = np.max(list(self.expt_diff_coeff.values()))
            bounds = np.asarray([lower_bound, upper_bound], dtype=np.float32)
        else:
            bounds = None
        return bounds

# Ethylene glycol (EG)
name = "EG"
mol_wt = 62.07
rho_c = 391.9405  # https://doi.org/10.1016/0009-2509(89)85265-0
Tc = 719.6
n_atoms = 10
smiles_str = "C(CO)O"
param_names = (
    "sigma_C1",
    "sigma_H1",  # Attached to C
    "sigma_H2",  # Attached to O
    "sigma_O1",
    "epsilon_C1",
    "epsilon_H1",
    "epsilon_H2",
    "epsilon_O1",
)

gaff_params = {
    "sigma_C1": 3.400,
    "sigma_H1": 2.471,  # Attached to C
    "sigma_H2": 0.000,  # Attached to O
    "sigma_O1": 3.066,
    "epsilon_C1": 0.457728 * (1 / 0.0083144598),
    "epsilon_H1": 0.065693 * (1 / 0.0083144598),
    "epsilon_H2": 0.0 * (1 / 0.0083144598),
    "epsilon_O1": 0.880310 * (1 / 0.0083144598),
}

bnds_sig = [
    [2.0, 4.0],  # [3.0, 4.0],  # C
    [1.5, 3.0],  # H1
    [0.0, 1e-14],  # H2 True Value is apparently 0. Ask EM)
    [2.0, 4.0],  # O #Check with EM what is reasonable here
]
bnds_eps = [
    [10.0, 75.0],  # [20.0, 75.0],  # C
    [2.0, 10.0],  # H
    [0.0, 1e-14],  # H2 (True Value is apparently 0. Ask EM)
    [75.0, 135.0],  # O #Check with EM what is reasonable here
]

# # Taylor, C. A., & Rinkenbach, Wm. H. (1926), J. Ind. Eng. Chem., 18(7), 676–678.
# # 10.1021/ie50199a004
# expt_liq_density = {
#     378.15: 1048.112,
#     383.15: 1044.162,
#     393.15: 1036.264,
#     403.15: 1028.365,
#     413.15: 1020.466,
# }

# # Correlation from 10.1063/1.3253106
# # gamma = 50.21-0.089*T(C)
# expt_surf_tens = {
#     378.15: 40.865,
#     383.15: 40.42,
#     393.15: 39.53,
#     403.15: 38.64,
#     413.15: 37.75,
# }

# # Taylor, C. A., & Rinkenbach, Wm. H. (1926), J. Ind. Eng. Chem., 18(7), 676–678.
# # 10.1021/ie50199a004
# expt_Pvap = {
#     378.15: (3.600 * u.kPa).to_value(u.bar),
#     383.15: (4.000 * u.kPa).to_value(u.bar),
#     393.15: (5.600 * u.kPa).to_value(u.bar),
#     403.15: (9.066 * u.kPa).to_value(u.bar),
#     413.15: (13.732 * u.kPa).to_value(u.bar),
# }

# # Taylor, C. A., & Rinkenbach, Wm. H. (1926), J. Ind. Eng. Chem., 18(7), 676–678.
# # 10.1021/ie50199a004
# expt_Hvap = {
#     403.75: 985.33,
#     433.95: 954.79,
#     461.55: 879.90,
# }

# uncertainty = {
#     "expt_surf_tens": 0.022,
# }

# From REFPROP
expt_liq_density = {
    378.15: 1051.7,
    383.15: 1047.9,
    393.15: 1040.2,
    403.15: 1032.3,
    413.15: 1024.3,
}

# From REFPROP
expt_surf_tens = {
    378.15: 40.939,
    383.15: 40.472,
    393.15: 39.533,
    403.15: 38.587,
    413.15: 37.634,
}

# From REFPROP
expt_Pvap = {
    378.15: (2.7108 * u.kPa).to_value(u.bar),
    383.15: (3.4853 * u.kPa).to_value(u.bar),
    393.15: (5.6304 * u.kPa).to_value(u.bar),
    403.15: (8.8396 * u.kPa).to_value(u.bar),
    413.15: (13.518 * u.kPa).to_value(u.bar),
}

# From REFPROP
expt_Hvap = {
    378.15: 977.45,
    383.15: 972.15,
    393.15: 961.57,
    403.15: 951.01,
    413.15: 940.46,
}

# From REFPROP
expt_vap_density = {
    378.15: 0.053545,
    383.15: 0.067953,
    393.15: 0.10702,
    403.15: 0.16393,
    413.15: 0.24479,
}

# DOI: doi: 10.1021/je60048a003 (m^2/s)
expt_diff_coeff = {
    298.15: 9.61*10**-11,
    303.15: 1.136*10**-10,
    313.15: 1.509*10**-10,
    323.15: 2.067*10**-10,
}

uncertainty = {
    "expt_surf_tens": 0.01,
    "expt_liq_density": 0.001,
    "expt_vap_density": 0.001,
}

# Create an instance of the EsolvsConstants class
EG = EsolvsConstants(
    name=name,
    mol_wt=mol_wt,
    Tc=Tc,
    rhoc=rho_c,
    n_atoms=n_atoms,
    smiles_str=smiles_str,
    param_names=param_names,
    gaff_params=gaff_params,
    bnds_sig=bnds_sig,
    bnds_eps=bnds_eps,
    expt_liq_density=expt_liq_density,
    expt_surf_tens=expt_surf_tens,
    expt_Pvap=expt_Pvap,
    expt_Hvap=expt_Hvap,
    expt_vap_density=expt_vap_density,
    expt_diff_coeff= expt_diff_coeff,
    uncertainty=uncertainty,
)


# Glycerol (Gly)
name = "Glycerol"
mol_wt = 92.09
rho_c = None
Tc = 850.0
n_atoms = 14
smiles_str = "C(C(CO)O)O"
param_names = (
    "sigma_C1",  # Outside Cs
    "sigma_C2",  # Middle C
    "sigma_H1",  # Attached to C1
    "sigma_H2",  # Attached to C2
    "sigma_H3",  # Attached to O1
    "sigma_H4",  # Attached to O2
    "sigma_O1",  # Outside O
    "sigma_O2",  # Inside O
    "epsilon_C1",
    "epsilon_C2",
    "epsilon_H1",
    "epsilon_H2",
    "epsilon_H3",
    "epsilon_H4",
    "epsilon_O1",
    "epsilon_O2",
)

gaff_params = {
    "sigma_C1": 3.400,  # Outside Cs
    "sigma_C2": 3.400,  # Middle C
    "sigma_H1": 2.471,  # Attached to C1
    "sigma_H2": 2.471,  # Attached to C2
    "sigma_H3": 0.00,  # Attached to O1
    "sigma_H4": 0.00,  # Attached to O2
    "sigma_O1": 3.066,  # Outside O
    "sigma_O2": 3.066,  # Inside O
    "epsilon_C1": 0.457728 * (1 / 0.0083144598),
    "epsilon_C2": 0.457728 * (1 / 0.0083144598),
    "epsilon_H1": 0.065693 * (1 / 0.0083144598),
    "epsilon_H2": 0.065693 * (1 / 0.0083144598),
    "epsilon_H3": 0.0 * (1 / 0.0083144598),
    "epsilon_H4": 0.0 * (1 / 0.0083144598),
    "epsilon_O1": 0.880310 * (1 / 0.0083144598),
    "epsilon_O2": 0.880310 * (1 / 0.0083144598),
}

bnds_sig = [
    [2.0, 4.0],  # C1
    [2.0, 4.0],  # C2
    [1.5, 3.0],  # H1
    [1.5, 3.0],  # H2
    [0.0, 1e-14],  # H3
    [0.0, 1e-14],  # H4 (Check lower bound since GAFF is 0)
    [2.0, 4.0],  # O1 #Check with EM what is reasonable here
    [2.0, 4.0],  # O2 #Check with EM what is reasonable here
]
bnds_eps = [
    [10.0, 75.0],  # C1
    [10.0, 75.0],  # C2
    [2.0, 10.0],  # H1
    [2.0, 10.0],  # H2
    [0.0, 1e-14],  # H3
    [0.0, 1e-14],  # H4 (Check lower bound since GAFF is 0)
    [75.0, 135.0],  # O1 #Check with EM what is reasonable here
    [75.0, 135.0],  # O2 #Check with EM what is reasonable here
]

# https://doi.org/10.1016/j.jct.2017.11.008
expt_liq_density = {
    293.15: 1261.1,
    303.15: 1255.6,
    313.15: 1249.5,
    323.15: 1242.9,
    333.15: 1236.0,
}

# mN/m (equal to dyn/cm)
# https://doi.org/10.1016/j.jct.2019.03.014
expt_surf_tens = {
    293.15: 63.1,
    303.15: 62.4,
    313.15: 62.0,
    323.15: 61.1,
    333.15: 60.5,
}

# https://doi.org/10.1016/j.jct.2017.11.008 (from correlation)
# P (Pa)= 10^5*EXP((-23263/388.7 + 82273*(1/388.7 - 1/T(K)) - 83*((388.7/T(K)) - 1 + LN(T(K)/388.7)))/8.314)

expt_Pvap = {
    293.15: (1.2022 * 10**-2 * u.Pa).to_value(u.bar),
    303.15: (4.053 * 10**-2 * u.Pa).to_value(u.bar),
    313.15: (1.251 * 10**-1 * u.Pa).to_value(u.bar),
    323.15: (3.567 * 10**-1 * u.Pa).to_value(u.bar),
    333.15: (9.459 * 10**-1 * u.Pa).to_value(u.bar),
}

# From https://doi.org/10.1016/j.fluid.2015.03.038
expt_Hvap = {
    298.15: 982.734,
}

#From  10.1103/PhysRevLett.96.145502 (m^2/s)
expt_diff_coeff = {
    293: 8.15e-13,
    299: 1.36e-12,
    306: 2.68e-12,
    313: 4.87e-12,
    321: 9.83e-12,
    329: 1.86e-11,
}
uncertainty = {
    "expt_liq_density": 0.0021,
    "expt_surf_tens": 0.0050,
    "expt_Pvap": 0.05,
    "expt_Hvap": 0.0081,
}

# Create an instance of the EsolvsConstants class
Gly = EsolvsConstants(
    name = name,
    mol_wt=mol_wt,
    Tc=Tc,
    rhoc=rho_c,
    n_atoms=n_atoms,
    smiles_str=smiles_str,
    param_names=param_names,
    gaff_params=gaff_params,
    bnds_sig=bnds_sig,
    bnds_eps=bnds_eps,
    expt_liq_density=expt_liq_density,
    expt_surf_tens=expt_surf_tens,
    expt_Pvap=expt_Pvap,
    expt_Hvap=expt_Hvap,
    expt_vap_density=None,
    expt_diff_coeff= expt_diff_coeff,
    uncertainty=uncertainty,
)


# Acetonitrile  (ACN)
name = "ACN"
mol_wt = 41.05
rho_c = 240.143  # kg/m^3 (DOI: 10.1021/j150462a016)
Tc = 547.9
n_atoms = 6
smiles_str = "CC#N"
param_names = (
    "sigma_C1",  # C attached to N
    "sigma_C2",  # C attached to H
    "sigma_H1",  # Attached to C2
    "sigma_N1",
    "epsilon_C1",
    "epsilon_C2",
    "epsilon_H1",
    "epsilon_N1",
)


gaff_params = {
    "sigma_C1": 3.400,  # C attached to N #Angstroms
    "sigma_C2": 3.400,  # C attached to H
    "sigma_H1": 2.65,  # Attached to C2
    "sigma_N1": 3.25,
    "epsilon_C1": 0.878693 * (1 / 0.0083144598), #kJ/mol
    "epsilon_C2": 0.457728 * (1 / 0.0083144598),
    "epsilon_H1": 0.065693 * (1 / 0.0083144598),
    "epsilon_N1": 0.711277 * (1 / 0.0083144598),
}

bnds_sig = [
    [2.0, 4.0],  # C1
    [2.0, 4.0],  # C2
    [1.5, 3.0],  # H1
    [2.0, 4.0],  # N #Check with EM what is reasonable here
]
bnds_eps = [
    [75.0, 135.0],  # C1 (Check with EM what is reasonable here)
    [10.0, 75.0],  # C2
    [2.0, 10.0],  # H1
    [40.0, 100.0],  # N #Check with EM what is reasonable here
]

# (equal to dyn/cm)
# DOI: 10.1021/ja02201a003
# expt_surf_tens = {
#     286.95: 29.18,
#     304.55: 26.81,
#     315.95: 24.43,
#     328.15: 23.82,
#     340.15: 22.35,
# }
# From correlation (10.1021/ja02201a003)
expt_surf_tens = {
    293.15: 28.23,
    298.15: 27.5961,
    303.15: 26.9675,
    308.15: 26.3441,
    313.15: 25.726,
}

# https://doi.org/10.1016/j.molliq.2014.10.017
expt_liq_density = {
    293.15: 782.04,
    298.15: 776.66,
    303.15: 771.24,
    308.15: 765.78,
    313.15: 760.29,
}

# https://doi.org/10.1063/1.1696002
# expt_Pvap = {
#     280.409: (4.997 * u.kPa).to_value(u.bar),
#     286.941: (6.914 * u.kPa).to_value(u.bar),
#     295.055: (10.244 * u.kPa).to_value(u.bar),
#     298.713: (12.156 * u.kPa).to_value(u.bar),
#     300.530: (13.187 * u.kPa).to_value(u.bar),
# }

# https://doi.org/10.1063/1.1696002 (From Antoine correlation fit to data)
expt_Pvap = {
    293.15: (9.3694 * u.kPa).to_value(u.bar),
    298.15: (11.8343 * u.kPa).to_value(u.bar),
    303.15: (14.8329 * u.kPa).to_value(u.bar),
    308.15: (18.4556 * u.kPa).to_value(u.bar),
    313.15: (22.8033 * u.kPa).to_value(u.bar),
}


# https://doi.org/10.1063/1.1696002
expt_Hvap = {
    298.15: 809.382,
}

uncertainty = {
    "expt_liq_density": 0.002,
    "expt_surf_tens": 0.01,
    "expt_Pvap": 0.00427,
    "expt_Hvap": 0.0051,
}

# Create an instance of the EsolvsConstants class
ACN = EsolvsConstants(
    name=name,
    mol_wt=mol_wt,
    Tc=Tc,
    rhoc=rho_c,
    n_atoms=n_atoms,
    smiles_str=smiles_str,
    param_names=param_names,
    gaff_params=gaff_params,
    bnds_sig=bnds_sig,
    bnds_eps=bnds_eps,
    expt_liq_density=expt_liq_density,
    expt_surf_tens=expt_surf_tens,
    expt_Pvap=expt_Pvap,
    expt_Hvap=expt_Hvap,
    expt_vap_density=None,
    uncertainty=uncertainty,
)


# Methanol (MeOH)
name = "MeOH"
mol_wt = 32.04
rho_c = 273.846  # kg/m^3 (DOI: 10.1021/j150462a016)
Tc = 512.5
n_atoms = 6
smiles_str = "CO"
param_names = (
    "sigma_C1",  # C attached to H
    "sigma_H1",  # Attached to C
    "sigma_H2",  # Attached to O
    "sigma_O1",
    "epsilon_C1",
    "epsilon_H1",
    "epsilon_H2",
    "epsilon_O1",
)

gaff_params = {
    "sigma_C1": 3.400,  # C attached to H
    "sigma_H1": 2.471,  # Attached to C
    "sigma_H2": 0.00,  # Attached to O
    "sigma_O1": 3.066,
    "epsilon_C1": 0.457728 * (1 / 0.0083144598),
    "epsilon_H1": 0.065693 * (1 / 0.0083144598),
    "epsilon_H2": 0.0 * (1 / 0.0083144598),
    "epsilon_O1": 0.880310 * (1 / 0.0083144598),
}

bnds_sig = [
    [2.0, 4.0],  # C1
    [1.5, 3.0],  # H1
    [0.0, 1e-14],  # H2
    [2.0, 4.0],  # O #Check with EM what is reasonable here
]
bnds_eps = [
    [10.0, 75.0],  # C1
    [2.0, 10.0],  # H1
    [0.0, 1e-14],  # H2
    [75.0, 135.0],  # O #Check with EM what is reasonable here
]


# # (equal to dyn/cm)
# # Table 2.7aa Thermophysical Properties of Methanol along the Saturation Line. From Knovel
# expt_surf_tens = {
#     200.0: 31.1,
#     300.0: 22.1,
#     350.0: 17.9,
#     400.0: 13.0,
#     500.0: 1.20,
# }

# # Table 2.7aa Thermophysical Properties of Methanol along the Saturation Line. From Knovel
# expt_liq_density = {
#     200.0: 880.28,
#     300.0: 784.51,
#     350.0: 735.84,
#     400.0: 678.59,
#     500.0: 451.53,
# }

# expt_vap_density = {
#     200.0: 0.0001,
#     300.0: 0.2462,
#     350.0: 1.9053,
#     400.0: 8.7343,
#     500.0: 109.88,
# }

# # Table 2.7aa Thermophysical Properties of Methanol along the Saturation Line. From Knovel
# expt_Pvap = {
#     200.0: (0.061 * u.kPa).to_value(u.bar),
#     300.0: (18.7 * u.kPa).to_value(u.bar),
#     350.0: (161.7 * u.kPa).to_value(u.bar),
#     400.0: (773.3 * u.kPa).to_value(u.bar),
#     500.0: (6525.0 * u.kPa).to_value(u.bar),
# }

# # https://doi.org/10.1063/1.1696002
# expt_Hvap = {
#     200.0: 1289.99,
#     300.0: 1166.17,
#     350.0: 1075.936,
#     400.0: 944.57,
#     500.0: 391.09,
# }


# (equal to dyn/cm)
# REFPROP
expt_surf_tens = {
    200.0: 31.717,
    250.0: 26.347,
    300.0: 21.993,
    350.0: 17.727,
    400.0: 12.928,
}

# REFPROP
expt_liq_density = {
    200.0: 880.28,
    250.0: 831.52,
    300.0: 784.51,
    350.0: 735.84,
    400.0: 678.59,
}

# REFPROP
expt_vap_density = {
    200.0: 1.1754*1e-4,
    250.0: 0.012577,
    300.0: 0.24623,
    350.0: 1.9053,
    400.0: 8.7343,
}

# REFPROP
expt_Pvap = {
    200.0: (6.0958*1e-03 * u.kPa).to_value(u.bar),
    250.0: (0.81030 * u.kPa).to_value(u.bar),
    300.0: (18.682 * u.kPa).to_value(u.bar),
    350.0: (161.72 * u.kPa).to_value(u.bar),
    400.0: (773.74 * u.kPa).to_value(u.bar),
}

# REFPROP
uncertainty = {
    "expt_liq_density": 0.001,
    "expt_surf_tens": 0.0126,
    "expt_vap_density": 0.001,
    "expt_diff_coeff": 0.008,
}

# REFPROP
expt_Hvap = {
    200.0: 1289.99,
    250.0: 1234.79,
    300.0: 1166.17,
    350.0: 1075.936,
    400.0: 944.57,
}

#From DOI: 10.1039/F19858100769 (m^2/s)
expt_diff_coeff = {
    278.2: 1.67*10**-9,
    283.2: 1.82*10**-9,
    298.2: 2.42*10**-9,
    313.2: 3.09*10**-9,
    328.2: 3.80*10**-9,
}
# Create an instance of the EsolvsConstants class
MeOH = EsolvsConstants(
    name=name,
    mol_wt=mol_wt,
    Tc=Tc,
    rhoc=rho_c,
    n_atoms=n_atoms,
    smiles_str=smiles_str,
    param_names=param_names,
    gaff_params=gaff_params,
    bnds_sig=bnds_sig,
    bnds_eps=bnds_eps,
    expt_liq_density=expt_liq_density,
    expt_surf_tens=expt_surf_tens,
    expt_Pvap=expt_Pvap,
    expt_Hvap=expt_Hvap,
    expt_vap_density=expt_vap_density,
    expt_diff_coeff=expt_diff_coeff,
    uncertainty=uncertainty,    
)

# Dimethylformamide (DMF)
name = "DMF"
mol_wt = 73.09
n_atoms = 12
smiles_str = "CN(C)C=O"
param_names = (
    "sigma_C1",  # C attached to N and H
    "sigma_C2",  # C attached to O, and N
    "sigma_H1",  # Attached to C1
    "sigma_H2",  # Attached to C2
    "sigma_O1",
    "sigma_N1",
    "epsilon_C1",
    "epsilon_C2",
    "epsilon_H1",
    "epsilon_H2",
    "epsilon_O1",
    "epsilon_N1",
)

rho_c = (
    279.204  # kg/m^3 (https://webbook.nist.gov/cgi/cbook.cgi?ID=C68122&Mask=4#ref-2)
)
Tc = 649.6  # K (https://webbook.nist.gov/cgi/cbook.cgi?ID=C68122&Mask=4#ref-2)

gaff_params = {
    "sigma_C1": 3.400,  # C attached to N and H
    "sigma_C2": 3.400,  # C attached to O, and N
    "sigma_H1": 2.471,  # Attached to C1
    "sigma_H2": 2.421,  # Attached to C2
    "sigma_O1": 2.96,
    "sigma_N1": 3.25,
    "epsilon_C1": 55.052,
    "epsilon_C2": 43.277,
    "epsilon_H1": 7.90105,
    "epsilon_H2": 7.548,
    "epsilon_O1": 105.676,
    "epsilon_N1": 85.547,
}

bnds_sig = [
    [2.0, 4.0],  # C1
    [2.0, 4.0],  # C2
    [1.5, 3.0],  # H1
    [1.5, 3.0],  # H2
    [2.0, 4.0],  # O1 #Check with EM what is reasonable here
    [2.0, 4.0],  # N1 #Check with EM what is reasonable here
]
bnds_eps = [
    [10.0, 75.0],  # C1
    [10.0, 75.0],  # C2
    [2.0, 10.0],  # H1
    [2.0, 10.0],  # H2
    [75.0, 135.0],  # O1 #Check with EM what is reasonable here
    [40.0, 100.0],  # N1 #Check with EM what is reasonable here
]


# mN/m (equal to dyn/cm)
# https://doi.org/10.1139/v70-464
# expt_surf_tens = {
#     287.81: 36.96,
#     297.82: 35.83,
#     307.86: 34.65,
#     317.86: 33.37,
#     327.89: 32.03,
# }

# https://doi.org/10.1139/v70-464 (From correlation: Note: Extrapolating outside of temp range)
# expt_surf_tens = {
#     308.15: 34.9265,
#     313.15: 34.306,
#     323.15: 33.6855,
#     333.15: 33.065,
#     343.15: 32.4445,
# }

# DOI: 10.1021/je0201323 (From correlation)
# gamma = 71.99-0.12158(T(K))
expt_surf_tens = {
    308.15: 34.525,
    313.15: 33.917,
    318.15: 33.309,
    323.15: 32.701,
    328.15: 32.094,
}

# https://doi.org/10.1016/j.molliq.2019.02.097
expt_liq_density = {
    308.15: 941.56,
    313.15: 936.76,
    318.15: 931.94,
    323.15: 927.10,
    328.15: 922.24,
}

expt_vap_density = None

# DOI: 10.1021/je060224i
# P(Kpa)=5.22*1000*EXP((596.6/K10)*(36.48688*(1-K10/596.6) -68.67427*(1-K10/596.6)^1.25 + 62.2708*(1-K10/596.6)^3 -229.0057*(1-K10/596.6)^7))
expt_Pvap = {
    308.15: (0.9980 * u.kPa).to_value(u.bar),
    313.15: (1.3740 * u.kPa).to_value(u.bar),
    318.15: (1.8500 * u.kPa).to_value(u.bar),
    323.15: (2.4437 * u.kPa).to_value(u.bar),
    328.15: (3.1753 * u.kPa).to_value(u.bar),
}

# https://doi.org/10.3390/molecules29051110
expt_Hvap = {
    298.15: 641.538,
}

# From DOI: 10.1039/C6RA18085J (m^2/s)
expt_diff_coeff = {
    222: 3.01*10**-10,
    243: 5.57*10**-10,
    274: 1.11*10**-9,
    298: 1.64*10**-9,
    334: 2.59*10**-9,
}

uncertainty = {
    "expt_liq_density": 1.43e-6,
    "expt_surf_tens": 0.0065,
    "expt_Pvap": 0.03428,
    "expt_Hvap": 0.0126,
    "expt_diff_coeff": 0.06232,
}

# Create an instance of the EsolvsConstants class
DMF = EsolvsConstants(
    name=name,
    mol_wt=mol_wt,
    Tc=Tc,
    rhoc=rho_c,
    n_atoms=n_atoms,
    smiles_str=smiles_str,
    param_names=param_names,
    gaff_params=gaff_params,
    bnds_sig=bnds_sig,
    bnds_eps=bnds_eps,
    expt_liq_density=expt_liq_density,
    expt_surf_tens=expt_surf_tens,
    expt_Pvap=expt_Pvap,
    expt_Hvap=expt_Hvap,
    expt_vap_density=expt_vap_density,
    expt_diff_coeff=expt_diff_coeff,
    uncertainty=uncertainty,
)


# Dimethylsulfoxide (DMSO)
name = "DMSO"
mol_wt = 78.13
n_atoms = 10
smiles_str = "CS(=O)C"
param_names = (
    "sigma_C1",  # C attached to S
    "sigma_H1",  # Attached to C1
    "sigma_O1",
    "sigma_S1",
    "epsilon_C1",
    "epsilon_H1",
    "epsilon_O1",
    "epsilon_S1",
)

rho_c = 366  # kg/m^3 (https://doi.org/10.1139/v79-114)
Tc = 693  # K (https://doi.org/10.1016/j.fluid.2018.05.029)


gaff_params = {
    "sigma_C1": 3.400,  # C attached to S
    "sigma_H1": 2.471,  # Attached to C1
    "sigma_O1": 2.960,
    "sigma_S1": 3.564,
    "epsilon_C1": 0.457728 * (1 / 0.0083144598),
    "epsilon_H1": 0.065693 * (1 / 0.0083144598),
    "epsilon_O1": 0.878639 * (1 / 0.0083144598),
    "epsilon_S1": 1.046001 * (1 / 0.0083144598),
}

bnds_sig = [
    [2.0, 4.0],  # C1
    [1.5, 3.0],  # H1
    [2.0, 4.0],  # O #Check with EM what is reasonable here
    [2.5, 4.5],  # S #Check with EM what is reasonable here
]
bnds_eps = [
    [10.0, 75.0],  # C1
    [2.0, 10.0],  # H1
    [75.0, 135.0],  # O #Check with EM what is reasonable here
    [90.0, 150.0],  # S #Check with EM what is reasonable here
]

# # https://doi.org/10.1016/j.molliq.2016.10.115
# expt_liq_density = {
#     308.15: 1085.25,
#     318.15: 1075.22,
#     333.15: 1060.17,
#     348.15: 1045.09,
#     363.15: 1029.96,
# }

# https://doi.org/10.1016/j.molliq.2016.10.115
expt_liq_density = {
    313.15: 1080.23,
    318.15: 1075.22,
    323.15: 1070.21,
    328.15: 1065.19,
    333.15: 1060.17,
}

expt_vap_density = None

# Correlation from https://doi.org/10.1016/0021-9614(72)90007-9
# expt_Pvap = {
#     308.15: (0.1228 * u.kPa).to_value(u.bar),
#     318.15: (0.2340 * u.kPa).to_value(u.bar),
#     333.15: (0.5676 * u.kPa).to_value(u.bar),
#     348.15: (1.2629 * u.kPa).to_value(u.bar),
#     363.15: (2.6069 * u.kPa).to_value(u.bar),
# }

# Correlation from https://doi.org/10.1016/0021-9614(72)90007-9
# EXP(-7820.48/T(K)-5.03613*LN(T(K)) + 54.4129)
expt_Pvap = {
    313.15: (0.219431 * u.kPa).to_value(u.bar),
    318.15: (0.299989 * u.kPa).to_value(u.bar),
    323.15: (0.405674 * u.kPa).to_value(u.bar),
    328.15: (0.542924 * u.kPa).to_value(u.bar),
    333.15: (0.719452 * u.kPa).to_value(u.bar),
}

# mN/m (equal to dyn/cm)
# # https://doi.org/10.1016/j.jct.2013.02.021
# expt_surf_tens = {
#     293.15: 36.14,
#     298.15: 35.90,
#     303.15: 35.56,
#     308.15: 35.35,
#     313.15: 35.10,
# }

# DOI: 10.1021/j100798a501 (45.78 -0.1145*T(C))
expt_surf_tens = {
    313.15: 41.2,
    318.15: 40.6275,
    323.15: 40.055,
    328.15: 39.4825,
    333.15: 38.91,
}

# NIST https://webbook.nist.gov/cgi/inchi?ID=C67685&Mask=4#Thermo-Phase
expt_Hvap = {
    308.15: 669.397,
    318.15: 666.837,
    320.15: 661.718,
    340.15: 647.639,
    368.15: 615.641,
}

#From DOI https://doi.org/10.1039/TF9716701302
expt_diff_coeff = {
    288.2: 6.5*10**-10,
    299.2: 8.4*10**-10,
    308.2: 1.1*10**-9,
}
uncertainty = {
    "expt_liq_density": 0.003,
    "expt_surf_tens": 0.0012,
    "expt_Pvap": 0.067,
    "expt_diff_coeff": 0.05,
}

# Create an instance of the EsolvsConstants class
DMSO = EsolvsConstants(
    name=name,
    mol_wt=mol_wt,
    Tc=Tc,
    rhoc=rho_c,
    n_atoms=n_atoms,
    smiles_str=smiles_str,
    param_names=param_names,
    gaff_params=gaff_params,
    bnds_sig=bnds_sig,
    bnds_eps=bnds_eps,
    expt_liq_density=expt_liq_density,
    expt_surf_tens=expt_surf_tens,
    expt_Pvap=expt_Pvap,
    expt_Hvap=expt_Hvap,
    expt_vap_density=expt_vap_density,
    expt_diff_coeff= expt_diff_coeff,
    uncertainty=uncertainty,
)


# Tetrahydrofuran (THF)
name = "THF"
mol_wt = 72.11
n_atoms = 13
smiles_str = "C1CCOC1"
param_names = (
    "sigma_C1",  # C attached to O
    "sigma_C2",  # C not attached to O
    "sigma_H1",  # Attached to C1
    "sigma_H2",  # Attached to C2
    "sigma_O1",
    "epsilon_C1",
    "epsilon_C2",
    "epsilon_H1",
    "epsilon_H2",
    "epsilon_O1",
)

gaff_params = {
    "sigma_C1": 3.400,  # C attached to O
    "sigma_C2": 3.400,  # C not attached to O
    "sigma_H1": 2.471,  # Attached to C1
    "sigma_H2": 2.650,  # Attached to C2
    "sigma_O1": 3.000,
    "epsilon_C1": 0.457728 * (1 / 0.0083144598),
    "epsilon_C2": 0.457728 * (1 / 0.0083144598),
    "epsilon_H1": 0.065693 * (1 / 0.0083144598),
    "epsilon_H2": 0.065693 * (1 / 0.0083144598),
    "epsilon_O1": 0.711277 * (1 / 0.0083144598),
}

bnds_sig = [
    [2.0, 4.0],  # C1
    [2.0, 4.0],  # C2
    [1.5, 3.0],  # H1
    [1.5, 3.0],  # H2
    [2.0, 4.0],  # O1 #Check with EM what is reasonable here
]
bnds_eps = [
    [10.0, 75.0],  # C1
    [10.0, 75.0],  # C2
    [2.0, 10.0],  # H1
    [2.0, 10.0],  # H2
    [40.0, 100.0],  # O1 #Check with EM what is reasonable here
]

rho_c = 320.168  # kg/m^3 (https://doi.org/10.1007/s10765-023-03258-3)
Tc = 540.2  # K (https://doi.org/10.1007/s10765-023-03258-3)

# # https://doi.org/10.1016/j.jct.2005.06.003
# expt_liq_density = {
#     283.15: 384.628,
#     293.15: 380.93,
#     303.15: 377.361,
#     313.15: 373.976,
#     323.15: 370.327,
# }

# https://doi.org/10.1016/j.jct.2005.06.003
expt_liq_density = {
    293.15: 380.930,
    298.15: 379.127,
    303.15: 377.361,
    308.15: 375.651,
    313.15: 373.976,
}

expt_vap_density = None

# # https://doi.org/10.1016/j.jct.2005.06.003
# expt_Pvap = {
#     308.15: (10.73 * u.kPa).to_value(u.bar),
#     318.15: (17.253 * u.kPa).to_value(u.bar),
#     333.15: (26.812 * u.kPa).to_value(u.bar),
#     348.15: (40.207 * u.kPa).to_value(u.bar),
#     363.15: (58.543 * u.kPa).to_value(u.bar),
# }

# https://doi.org/10.1016/j.jct.2005.06.003
expt_Pvap = {
    293.15: (17.253 * u.kPa).to_value(u.bar),
    298.15: (21.596 * u.kPa).to_value(u.bar),
    303.15: (26.812 * u.kPa).to_value(u.bar),
    308.15: (32.935 * u.kPa).to_value(u.bar),
    313.15: (40.207 * u.kPa).to_value(u.bar),
}

# mN/m (equal to dyn/cm)
# https://doi.org/10.1016/j.jct.2013.02.021
expt_surf_tens = {
    293.15: 36.14,
    298.15: 35.90,
    303.15: 35.56,
    308.15: 35.35,
    313.15: 35.10,
}

# https://doi.org/10.1016/0021-9614(81)90046-X
expt_Hvap = {
    301.8: 441.035,
    319.18: 428.346,
    339.10: 413.466,
}

uncertainty = {
    "expt_liq_density": 1.0 * 10**-7,
    "expt_surf_tens": 0.0094,
    "expt_Pvap": 2.5 * 10**-4,
    "expt_Hvap": 3.9 * 10**-4,
}

# Create an instance of the EsolvsConstants class
THF = EsolvsConstants(
    name=name,
    mol_wt=mol_wt,
    Tc=Tc,
    rhoc=rho_c,
    n_atoms=n_atoms,
    smiles_str=smiles_str,
    param_names=param_names,
    gaff_params=gaff_params,
    bnds_sig=bnds_sig,
    bnds_eps=bnds_eps,
    expt_liq_density=expt_liq_density,
    expt_surf_tens=expt_surf_tens,
    expt_Pvap=expt_Pvap,
    expt_Hvap=expt_Hvap,
    expt_vap_density=expt_vap_density,
    uncertainty=uncertainty,
)


# Dichloromethane (DCM)
name = "DCM"
mol_wt = 84.93
n_atoms = 5
smiles_str = "C(Cl)Cl"
param_names = (
    "sigma_C1",  # C
    "sigma_H1",  # H
    "sigma_Cl1",  # Cl
    "epsilon_C1",
    "epsilon_H1",
    "epsilon_Cl1",
)

# Fix me later
gaff_params = {
    "sigma_C1": 3.400,  # C
    "sigma_H1": 2.293,  # H
    "sigma_Cl1": 3.471,  # Cl
    "epsilon_C1": 0.457728 * (1 / 0.0083144598),
    "epsilon_H1": 0.065693 * (1 / 0.0083144598),
    "epsilon_Cl1": 1.108758 * (1 / 0.0083144598),
}

bnds_sig = [
    [2.0, 4.0],  # C1
    [1.5, 3.0],  # H1
    [2.5, 4.5],  # Cl #Check with EM what is reasonable here
]
bnds_eps = [
    [10.0, 75.0],  # C1
    [2.0, 10.0],  # H2
    [90.0, 150.0],  # Cl #Check with EM what is reasonable here
]

rho_c = 444  # kg/m^3 (https://doi.org/10.1002/bbpc.19850890715)
Tc = 510  # K (https://doi.org/10.1002/bbpc.19850890715)

# #DOI: 10.1021/je60045a018
# expt_liq_density = {
#     277.26: 1355.4,
#     288.45: 1334.4,
#     313.21: 1287.0,
#     343.22: 1228.0,
#     373.93: 1165.9,
# }

# # https://doi.org/10.1016/j.jct.2010.08.001 (Extrapolated)
# expt_liq_density = {
#     270.0: 1367.5,
#     281.0: 1348.6,
#     293.15: 1326.3,
#     310.0: 1294.8,
#     330.0: 1256.4,
# }

# expt_vap_density = None

# # From correlation in https://doi.org/10.1016/j.jct.2010.08.001
# expt_Pvap = {
#     270.0: (16.35 * u.kPa).to_value(u.bar),
#     281.0: (27.76 * u.kPa).to_value(u.bar),
#     293.15: (47.21 * u.kPa).to_value(u.bar),
#     310.0: (91.02 * u.kPa).to_value(u.bar),
#     330.0: (197.54 * u.kPa).to_value(u.bar),
# }

# https://doi.org/10.1016/j.jct.2010.08.001 (Extrapolated)
# rho(kg/m^3) = (84.932*6426*10^3/(8314*510))*(1/(0.27036)^(1+(1-T(K)/510)^(2/7)))
expt_liq_density = {
    293.15: 1326.18,
    298.15: 1317.19,
    303.15: 1301.10,
    308.15: 1298.92,
    313.15: 1289.65,
}
expt_vap_density = None

# From correlation in https://doi.org/10.1016/j.jct.2010.08.001
# P(kpa) = 6426*exp((6.8708 + 9.2238*(1-T(K)/510)^1.89 + 21.757*(1-T(K)/510)^5.67)*ln(T(K)/510))
expt_Pvap = {
    293.15: (47.21 * u.kPa).to_value(u.bar),
    298.15: (57.89 * u.kPa).to_value(u.bar),
    303.15: (70.42 * u.kPa).to_value(u.bar),
    308.15: (85.04 * u.kPa).to_value(u.bar),
    313.15: (101.98 * u.kPa).to_value(u.bar),
}
# mN/m (equal to dyn/cm)
# DOI: 10.1021/j100345a065 and https://srd.nist.gov/jpcrdreprint/1.3253106.pdf
expt_surf_tens = {
    293.15: 27.84,
    298.15: 27.20,
    303.15: 26.56,
    308.15: 25.91,
    313.15: 25.27,
}

# https://webbook.nist.gov/cgi/cbook.cgi?ID=C75092&Mask=4#Thermo-Phase (NIST)
# Unreliable
expt_Hvap = {
    248.0: 355.587,
    279.0: 356.764,
    308.0: 343.813,
    313.0: 330.39,
    326.0: 341.458,
}

uncertainty = {
    "expt_liq_density": 0.003,
    "expt_surf_tens": 0.0094,
    "expt_Pvap": 0.017,
}

# Create an instance of the EsolvsConstants class
DCM = EsolvsConstants(
    name = name,
    mol_wt=mol_wt,
    Tc=Tc,
    rhoc=rho_c,
    n_atoms=n_atoms,
    smiles_str=smiles_str,
    param_names=param_names,
    gaff_params=gaff_params,
    bnds_sig=bnds_sig,
    bnds_eps=bnds_eps,
    expt_liq_density=expt_liq_density,
    expt_surf_tens=expt_surf_tens,
    expt_Pvap=expt_Pvap,
    expt_Hvap=expt_Hvap,
    expt_vap_density=expt_vap_density,
    uncertainty=uncertainty,
)

# Diethyl Carbonate (DEC)
name = "DEC"
mol_wt = 118.13
n_atoms = 18
smiles_str = "CCOC(=O)OCC"
param_names = (
    "sigma_C1",  # Outsidemost C
    "sigma_C2",  # C attached to one O
    "sigma_C3",  # C attached to three O
    "sigma_H1",  # H attached to C1
    "sigma_H2",  # H attached to C2
    "sigma_O1",  # O with no double bond
    "sigma_O2",  # O with double bond
    "epsilon_C1",
    "epsilon_C2",
    "epsilon_C3",
    "epsilon_H1",
    "epsilon_H2",
    "epsilon_O1",
    "epsilon_O2",
)

gaff_params = {
    "sigma_C1": 3.400,  # Outsidemost C
    "sigma_C2": 3.400,  # C attached to one O
    "sigma_C3": 3.400,  # C attached to three O
    "sigma_H1": 2.650,  # H attached to C1
    "sigma_H2": 2.471,  # H attached to C2
    "sigma_O1": 3.000,  # O with no double bond
    "sigma_O2": 2.960,  # O with double bond
    "epsilon_C1": 0.457728 * (1 / 0.0083144598),
    "epsilon_C2": 0.457728 * (1 / 0.0083144598),
    "epsilon_C3": 0.359825 * (1 / 0.0083144598),
    "epsilon_H1": 0.065693 * (1 / 0.0083144598),
    "epsilon_H2": 0.065693 * (1 / 0.0083144598),
    "epsilon_O1": 0.711277 * (1 / 0.0083144598),
    "epsilon_O2": 0.878639 * (1 / 0.0083144598),
}

bnds_sig = [
    [2.0, 4.0],  # C1
    [2.0, 4.0],  # C2
    [2.0, 4.0],  # C3
    [1.5, 3.0],  # H1
    [1.5, 3.0],  # H2
    [2.0, 4.0],  # O1 #Check with EM what is reasonable here
    [2.0, 4.0],  # O2 #Check with EM what is reasonable here
]
bnds_eps = [
    [10.0, 75.0],  # C1
    [10.0, 75.0],  # C2
    [10.0, 75.0],  # C3
    [2.0, 10.0],  # H1
    [2.0, 10.0],  # H2
    [40.0, 100.0],  # O1 #Check with EM what is reasonable here
    [75.0, 135.0],  # O2 #Check with EM what is reasonable here
]

rho_c = 245.46  # kg/m^3 (DOI: 10.1021/je100494z)
Tc = 576  # K (DOI: 10.1021/je100494z)

# https://doi.org/10.1016/j.fluid.2010.03.040
expt_liq_density = {
    273.18: 998.2,
    293.16: 975.7,
    313.19: 952.5,
    333.18: 929.1,
    353.19: 906.3,
}

# https://doi.org/10.1016/j.fluid.2010.03.040
expt_vap_density = {
    273.18: 0.036,
    293.16: 0.057,
    313.19: 0.085,
    333.18: 0.121,
    353.19: 0.164,
}

# From correlation in https://doi.org/10.1016/j.jct.2008.02.012
expt_Pvap = {
    273.18: (0.2845 * u.kPa).to_value(u.bar),
    293.16: (1.1101 * u.kPa).to_value(u.bar),
    313.19: (3.5244 * u.kPa).to_value(u.bar),
    333.18: (9.4256 * u.kPa).to_value(u.bar),
    353.19: (21.9621 * u.kPa).to_value(u.bar),
}

# mN/m (equal to dyn/cm)
# https://doi.org/10.1016/j.fluid.2010.03.040
expt_surf_tens = {
    273.18: 29.2,
    293.16: 26.8,
    313.19: 24.5,
    333.18: 22.4,
    353.19: 20.2,
}

# https://doi.org/10.1016/j.jct.2008.02.012
expt_Hvap = {
    294.15: 527.022,
    298.15: 522.195,
    330.95: 494.054,
    338.35: 481.809,
    384.10: 451.584,
}

# DOI: 10.1021/acs.jced.1c00101 (m^2/s) (Estimations from plot)
expt_diff_coeff = {
    263: 2.6*10**-10,
    303: 5.9*10**-10,
    353: 1.8*10**-9,
}

uncertainty = {
    "expt_surf_tens": 0.0064,
    "expt_Hvap": 0.0050,
}

# Create an instance of the EsolvsConstants class
DEC = EsolvsConstants(
    name=name,
    mol_wt=mol_wt,
    Tc=Tc,
    rhoc=rho_c,
    n_atoms=n_atoms,
    smiles_str=smiles_str,
    param_names=param_names,
    gaff_params=gaff_params,
    bnds_sig=bnds_sig,
    bnds_eps=bnds_eps,
    expt_liq_density=expt_liq_density,
    expt_surf_tens=expt_surf_tens,
    expt_Pvap=expt_Pvap,
    expt_Hvap=expt_Hvap,
    expt_vap_density=expt_vap_density,
    expt_diff_coeff=expt_diff_coeff,
    uncertainty=uncertainty,
)


# Dichloromethane (DCM)
name = "DCM"
mol_wt = 84.93
n_atoms = 5
smiles_str = "C(Cl)Cl"
param_names = (
    "sigma_C1",  # C
    "sigma_H1",  # H
    "sigma_Cl1",  # Cl
    "epsilon_C1",
    "epsilon_H1",
    "epsilon_Cl1",
)

# Fix me later
gaff_params = {
    "sigma_C1": 3.400,  # C
    "sigma_H1": 2.293,  # H
    "sigma_Cl1": 3.471,  # Cl
    "epsilon_C1": 0.457728 * (1 / 0.0083144598),
    "epsilon_H1": 0.065693 * (1 / 0.0083144598),
    "epsilon_Cl1": 1.108758 * (1 / 0.0083144598),
}

bnds_sig = [
    [2.0, 4.0],  # C1
    [1.5, 3.0],  # H1
    [2.5, 4.5],  # Cl #Check with EM what is reasonable here
]
bnds_eps = [
    [10.0, 75.0],  # C1
    [2.0, 10.0],  # H2
    [90.0, 150.0],  # Cl #Check with EM what is reasonable here
]

rho_c = 444  # kg/m^3 (https://doi.org/10.1002/bbpc.19850890715)
Tc = 510  # K (https://doi.org/10.1002/bbpc.19850890715)

# #DOI: 10.1021/je60045a018
# expt_liq_density = {
#     277.26: 1355.4,
#     288.45: 1334.4,
#     313.21: 1287.0,
#     343.22: 1228.0,
#     373.93: 1165.9,
# }

# # https://doi.org/10.1016/j.jct.2010.08.001 (Extrapolated)
# expt_liq_density = {
#     270.0: 1367.5,
#     281.0: 1348.6,
#     293.15: 1326.3,
#     310.0: 1294.8,
#     330.0: 1256.4,
# }

# expt_vap_density = None

# # From correlation in https://doi.org/10.1016/j.jct.2010.08.001
# expt_Pvap = {
#     270.0: (16.35 * u.kPa).to_value(u.bar),
#     281.0: (27.76 * u.kPa).to_value(u.bar),
#     293.15: (47.21 * u.kPa).to_value(u.bar),
#     310.0: (91.02 * u.kPa).to_value(u.bar),
#     330.0: (197.54 * u.kPa).to_value(u.bar),
# }

# https://doi.org/10.1016/j.jct.2010.08.001 (Extrapolated)
# rho(kg/m^3) = (84.932*6426*10^3/(8314*510))*(1/(0.27036)^(1+(1-T(K)/510)^(2/7)))
expt_liq_density = {
    293.15: 1326.18,
    298.15: 1317.19,
    303.15: 1308.10,
    308.15: 1298.92,
    313.15: 1289.65,
}
expt_vap_density = None

# From correlation in https://doi.org/10.1016/j.jct.2010.08.001
# P(kpa) = 6426*exp((6.8708 + 9.2238*(1-T(K)/510)^1.89 + 21.757*(1-T(K)/510)^5.67)*ln(T(K)/510))
expt_Pvap = {
    293.15: (47.21 * u.kPa).to_value(u.bar),
    298.15: (57.89 * u.kPa).to_value(u.bar),
    303.15: (70.42 * u.kPa).to_value(u.bar),
    308.15: (85.04 * u.kPa).to_value(u.bar),
    313.15: (101.98 * u.kPa).to_value(u.bar),
}
# mN/m (equal to dyn/cm)
# DOI: 10.1021/j100345a065 and https://srd.nist.gov/jpcrdreprint/1.3253106.pdf
expt_surf_tens = {
    293.15: 27.84,
    298.15: 27.20,
    303.15: 26.56,
    308.15: 25.91,
    313.15: 25.27,
}

# https://webbook.nist.gov/cgi/cbook.cgi?ID=C75092&Mask=4#Thermo-Phase (NIST)
# Unreliable
expt_Hvap = {
    248.0: 355.587,
    279.0: 356.764,
    308.0: 343.813,
    313.0: 330.39,
    326.0: 341.458,
}

uncertainty = {
    "expt_liq_density": 0.003,
    "expt_surf_tens": 0.0094,
    "expt_Pvap": 0.017,
}

# Create an instance of the EsolvsConstants class
DCM = EsolvsConstants(
    name=name,
    mol_wt=mol_wt,
    Tc=Tc,
    rhoc=rho_c,
    n_atoms=n_atoms,
    smiles_str=smiles_str,
    param_names=param_names,
    gaff_params=gaff_params,
    bnds_sig=bnds_sig,
    bnds_eps=bnds_eps,
    expt_liq_density=expt_liq_density,
    expt_surf_tens=expt_surf_tens,
    expt_Pvap=expt_Pvap,
    expt_Hvap=expt_Hvap,
    expt_vap_density=expt_vap_density,
    uncertainty=uncertainty,
)

# R125 (Test) - All data from REFPROP
name = "HFC-125"
mol_wt = 120.02
n_atoms = 8
smiles_str = "C(F)(F)C(F)(F)F"
param_names = (
            "sigma_C1",
            "sigma_C2",
            "sigma_F1",
            "sigma_F2",
            "sigma_H1",
            "epsilon_C1",
            "epsilon_C2",
            "epsilon_F1",
            "epsilon_F2",
            "epsilon_H1",
        )

# gaff_params = {
#             "sigma_C1":3.400, #2Fs
#             "sigma_C2": 3.400, #3Fs
#             "sigma_F1":3.118,  
#             "sigma_F2":3.118,
#             "sigma_H1":2.293,
#             "epsilon_C1":55.052, #2Fs
#             "epsilon_C2":55.052, #3Fs
#             "epsilon_F1":30.696,
#             "epsilon_F2":30.696,
#             "epsilon_H1":7.901,
#         }

gaff_params = {
            "sigma_C1":3.69463709, #2Fs
            "sigma_C2":3.10028756, #3Fs
            "sigma_F1":2.51917801,  #C1 F
            "sigma_F2":3.15864751, #C2 F
            "sigma_H1":2.38891812,
            "epsilon_C1":45.04630943, #2Fs
            "epsilon_C2":50.45904441, #3Fs
            "epsilon_F1":31.68405593,
            "epsilon_F2":28.93799857,
            "epsilon_H1":6.52832078,
        }

bnds_sig = [[2.0,4.0], #[3.0, 4.0],  # C
            [2.0,4.0], #[3.0, 4.0],  # C
            [2.0,4.0], #[2.5, 3.5],  # F
            [2.0,4.0], #[2.5, 3.5],  # F
            [1.5,3.0], #[1.7, 2.7],  # H
]
bnds_eps = [[10.0,75.0], #[20.0, 60.0],  # C
            [10.0,75.0], #[20.0, 60.0],  # C
            [15.0,50.0], #[15.0, 40.0],  # F
            [15.0,50.0], #[15.0, 40.0],  # F
            [2.0, 10.0],  # H
            ]

rho_c = 571.9 # kg/m^3 
Tc =  339.4  # K

expt_liq_density = {
            229.0: 1501.1,
            249.0: 1425.9,
            269.0: 1340.6,
            289.0: 1241.1,
            295.15: 1206.9,
            298.15: 1189.4,
            309.0: 1118.0,
        }

expt_vap_density = {
            229.0: 8.190,
            249.0: 18.524,
            269.0: 37.291,
            289.0: 69.667,
            295.15: 82.784,
            298.15: 90.557,
            309.0: 126.446,
        }

expt_Pvap = {
            229.0: (123.65 * u.kPa).to_value(u.bar),
            249.0: (290.76 * u.kPa).to_value(u.bar),
            269.0: (592.27 * u.kPa).to_value(u.bar),
            289.0: (1082.84 * u.kPa).to_value(u.bar),
            295.15: (1272.2 * u.kPa).to_value(u.bar),
            298.15: (1377.9 * u.kPa).to_value(u.bar),
            309.0: (1824.93 * u.kPa).to_value(u.bar),
        }

# mN/m (equal to dyn/cm)
# https://doi.org/10.1063/1.4768782 From correlation
#gamma = =10^3*(0.05252*(1-(T/Tc))^(1.237))
expt_surf_tens = {
    229.0: 13.069,
    249.0: 10.201,
    269.0: 7.480,
    289.0: 4.939,
    295.15: 4.2017,
    298.15: 3.8504,
    309.0: 2.633,
}

expt_Hvap = {
            229.0: 162.1,
            249.0: 149.8,
            269.0: 135.6,
            289.0: 118.5,
            295.15: 113.54,
            298.15: 110.4,
            309.0: 96.7,
        }

uncertainty = {
            "expt_liq_density": 0.001,
            "expt_vap_density": 0.001,
            "expt_surf_tens": 0.0131,
            "expt_Pvap": 0.0005,
            "expt_Hvap": 0.005
        }

# Create an instance of the EsolvsConstants class
R125 = EsolvsConstants(
    name=name,
    mol_wt=mol_wt,
    Tc=Tc,
    rhoc=rho_c,
    n_atoms=n_atoms,
    smiles_str=smiles_str,
    param_names=param_names,
    gaff_params=gaff_params,
    bnds_sig=bnds_sig,
    bnds_eps=bnds_eps,
    expt_liq_density=expt_liq_density,
    expt_surf_tens=expt_surf_tens,
    expt_Pvap=expt_Pvap,
    expt_Hvap=expt_Hvap,
    expt_vap_density=expt_vap_density,
)


def make_dict(mol_names=None):
    """
    Make a dictionary of all the solvents
    """
    solvents = {}
    if mol_names is None:
        mol_names = ["EG", "Gly", "ACN", "MeOH", "DMSO", "DMF", "THF", "DCM", "DEC", "R125"]

    for name in mol_names:
        if name == "EG":
            solvents[name] = EG
        elif name == "Gly":
            solvents[name] = Gly
        elif name == "ACN":
            solvents[name] = ACN
        elif name == "MeOH":
            solvents[name] = MeOH
        elif name == "DMSO":
            solvents[name] = DMSO
        elif name == "DMF":
            solvents[name] = DMF
        elif name == "THF":
            solvents[name] = THF
        elif name == "DCM":
            solvents[name] = DCM
        elif name == "DEC":
            solvents[name] = DEC
        elif name == "R125":
            solvents[name] = R125
        else:
            raise ValueError(f"Unknown solvent name: {name}")

    return solvents
