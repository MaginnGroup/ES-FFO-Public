import numpy as np
import unyt as u

class EsolvsConstants:
    """Experimental data and other constants"""
    def __init__(self, mol_wt, Tc, rhoc, n_atoms, smiles_str, param_names, gaff_params, bnds_sig, bnds_eps, expt_liq_density, expt_surftens, expt_Pvap, expt_Hvap, uncertainty=None):
        """Initialize the class with experimental data"""
        # assert (
        #     self.expt_liq_density.keys()
        #     == self.expt_surftens.keys()
        #     == self.expt_Pvap.keys()
        #     == self.expt_Hvap.keys()
        # )
        #FF Properties
        self.param_names = param_names
        self.gaff_params = gaff_params
        self.bnds_sig = bnds_sig
        self.bnds_eps = bnds_eps
        # Experimental data
        self.molecular_weight = mol_wt
        self.expt_Tc = Tc
        self.expt_rhoc = rhoc
        self.n_atoms = n_atoms
        self.smiles_str = smiles_str
        self.expt_liq_density = expt_liq_density
        self.expt_surftens = expt_surftens
        self.expt_Pvap = expt_Pvap
        self.expt_Hvap = expt_Hvap
        self.uncertainty = uncertainty
    
    @property
    def param_bounds(self):
        """Bounds on sigma and epsilon in units of nm and kJ/mol"""

        bounds_sigma = (
            (
                np.asarray(
                    self.bnds_sig
                )
                * u.Angstrom
            )
            .in_units(u.nm)
            .value
        )

        bounds_epsilon = (
            (
                np.asarray(
                    self.bnds_eps
                )
                * u.K
                * u.kb
            )
            .in_units("kJ/mol")
            .value
        )

        bounds = np.vstack((bounds_sigma, bounds_epsilon))

        return bounds

    @property
    def uncertainties(self):
        """
        Dictionary with uncertainty for each calculation
        from: https://doi.org/10.1063/1.555898
        """
        if self.uncertainty is not None:
           uncertainty = self.uncertainty
        else:
            uncertainty = {
                "expt_liq_density": 0.05,
                "expt_surftens": 0.05,
                "expt_Pvap": 0.05,
                "expt_Hvap": 0.05
            }
        return uncertainty
    
    def temperature_bounds(self, prop_name="expt_surftens"):
        """Bounds on temperature in units of K"""

        # Get the minimum and maximum temperature from the experimental data
        if prop_name == "expt_liq_density":
            prop = self.expt_liq_density
        elif prop_name == "expt_surftens":
            prop = self.expt_surftens
        elif prop_name == "expt_Pvap":
            prop = self.expt_Pvap
        elif prop_name == "expt_Hvap":
            prop = self.expt_Hvap
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
    def surftens_bounds(self):
        """Bounds on vapor density in units of kg/m^3"""

        lower_bound = np.min(list(self.expt_surftens.values()))
        upper_bound = np.max(list(self.expt_surftens.values()))
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
    

#Ethylene glycol
mol_wt = 62.07
rho_c = None
Tc = 719.6
n_atoms = 10
smiles_str = "C(CO)O"
param_names = (
    "sigma_C1",
    "sigma_H1", #Attached to C
    "sigma_H2", #Attached to O
    "sigma_O1",
    "epsilon_C1",
    "epsilon_H1",
    "epsilon_H2", 
    "epsilon_O1",
)
gaff_params = {
    "sigma_C1":3.400,
    "sigma_H1":2.65,
    "epsilon_C1":55.052,
    "epsilon_H1":7.901,
}

bounds_sig = [
                [2.0, 4.0], #[3.0, 4.0],  # C
                [1.5, 3.0],  # H1
                [1.5, 3.0],  # H2
                [2.0, 4.0],  # O #Check with EM what is reasonable here
            ]
bounds_eps = [
                [10.0,75.0], #[20.0, 75.0],  # C
                [2.0, 10.0],  # H
                [2.0, 10.0],  # H2
                [10.0, 75.0],  # O #Check with EM what is reasonable here
            ]

#Taylor, C. A., & Rinkenbach, Wm. H. (1926), J. Ind. Eng. Chem., 18(7), 676–678.
expt_liq_density = {
    293.15: 1113.02,
    313.15: 1098.65,
    333.15: 1083.46,
    353.15: 1067.86,
    373.15: 1052.06,
}
#Taylor, C. A., & Rinkenbach, Wm. H. (1926), J. Ind. Eng. Chem., 18(7), 676–678.
expt_surftens = {
    293.15: 48.43,
    313.15: 46.65,
    333.15: 44.87,
    353.15: 43.09,
    373.15: 41.31,
}
#Taylor, C. A., & Rinkenbach, Wm. H. (1926), J. Ind. Eng. Chem., 18(7), 676–678.
expt_Pvap = {
    395.65: (5.87 * u.kPa).to_value(u.bar),
    409.85: (11.06 * u.kPa).to_value(u.bar),
    413.95: (13.47 * u.kPa).to_value(u.bar),
    446.35: (47.64 * u.kPa).to_value(u.bar),
    459.65: (72.57 * u.kPa).to_value(u.bar),
}
#Taylor, C. A., & Rinkenbach, Wm. H. (1926), J. Ind. Eng. Chem., 18(7), 676–678.
expt_Hvap = {
    403.75: 985.33,
    433.95: 954.79,
    461.55: 879.90,
}
# Create an instance of the EsolvsConstants class   
EthGly = EsolvsConstants(
    mol_wt=mol_wt,
    Tc=Tc,
    rhoc=rho_c,
    n_atoms=n_atoms,
    smiles_str=smiles_str,
    param_names=param_names,
    gaff_params=gaff_params,
    bounds_sig=bounds_sig,
    bounds_eps=bounds_eps,
    expt_liq_density=expt_liq_density,
    expt_surftens=expt_surftens,
    expt_Pvap=expt_Pvap,
    expt_Hvap=expt_Hvap
)




#Glycerol
mol_wt = 92.09
rho_c = None
Tc = 850.0
n_atoms = 14
smiles_str = "C(C(CO)O)O"
param_names = (
    "sigma_C1", #Outside Cs
    "sigma_C2", #Middle C
    "sigma_H1", #Attached to outside C
    "sigma_H2", #Attached to middle C
    "sigma_H3", #Attached to O
    "sigma_O1",
    "epsilon_C1",
    "epsilon_C2",
    "epsilon_H1",
    "epsilon_H2", 
    "epsilon_H3",
    "epsilon_O1",
)

#Fix me later
gaff_params = {
    "sigma_C1":3.400,
    "sigma_H1":2.65,
    "epsilon_C1":55.052,
    "epsilon_H1":7.901,
}

bounds_sig = [
                [2.0, 4.0],  # C1
                [2.0, 4.0],  # C2
                [1.5, 3.0],  # H1
                [1.5, 3.0],  # H2
                [1.5, 3.0],  # H3
                [2.0, 4.0],  # O #Check with EM what is reasonable here
            ]
bounds_eps = [
                [10.0,75.0],  # C1
                [10.0,75.0],  # C2
                [2.0, 10.0],  # H1
                [2.0, 10.0],  # H2
                [2.0, 10.0],  # H3
                [10.0, 75.0],  # O #Check with EM what is reasonable here
            ]

#https://doi.org/10.1016/j.jct.2017.11.008
expt_liq_density = {
    293.15: 1261.1,
    303.15: 1255.6,
    313.15: 1249.5,
    323.15: 1242.9,
    333.15: 1236.0,
}

#mN/m (equal to dyn/cm)
#https://doi.org/10.1016/j.jct.2019.03.014
expt_surftens = {
    293.15: 63.1,
    303.15: 62.4,
    313.15: 62.0,
    323.15: 61.1,
    333.15: 60.5,
}

#https://doi.org/10.1016/j.jct.2017.11.008 (from correlation)
expt_Pvap = {
    293.15: (1.20222*10**-5 * u.kPa).to_value(u.bar),
    303.15: (4.04997*10**-5 * u.kPa).to_value(u.bar),
    313.15: (1.24883*10**-4 * u.kPa).to_value(u.bar),
    323.15: (3.55515*10**-4 * u.kPa).to_value(u.bar),
    333.15: (9.41385*10**-4 * u.kPa).to_value(u.bar),
}

#From https://doi.org/10.1016/j.fluid.2015.03.038
expt_Hvap = {
    298.15: 982.734,
}
# Create an instance of the EsolvsConstants class   
Glycerol = EsolvsConstants(
    mol_wt=mol_wt,
    Tc=Tc,
    rhoc=rho_c,
    n_atoms=n_atoms,
    smiles_str=smiles_str,
    param_names=param_names,
    gaff_params=gaff_params,
    bounds_sig=bounds_sig,
    bounds_eps=bounds_eps,
    expt_liq_density=expt_liq_density,
    expt_surftens=expt_surftens,
    expt_Pvap=expt_Pvap,
    expt_Hvap=expt_Hvap
)




#Acetonitrile  (ACN)
mol_wt = 41.05
rho_c = 1958.09 #kg/m^3 (DOI: 10.1021/j150462a016)
Tc = 547.9
n_atoms = 6
smiles_str = "CC#N"
param_names = (
    "sigma_C1", #C attached to H
    "sigma_C2", #C attached to N
    "sigma_H1", #Attached to outside C
    "sigma_N1",
    "epsilon_C1",
    "epsilon_C2",
    "epsilon_H1",
    "epsilon_N1",
)

#Fix me later
gaff_params = {
    "sigma_C1":3.400,
    "sigma_H1":2.65,
    "epsilon_C1":55.052,
    "epsilon_H1":7.901,
}

bounds_sig = [
                [2.0, 4.0],  # C1
                [2.0, 4.0],  # C2
                [1.5, 3.0],  # H1
                [2.0, 4.0],  # N #Check with EM what is reasonable here
            ]
bounds_eps = [
                [10.0,75.0],  # C1
                [10.0,75.0],  # C2
                [2.0, 10.0],  # H1
                [10.0, 75.0],  # N #Check with EM what is reasonable here
            ]


#(equal to dyn/cm)
#DOI: 10.1021/ja02201a003
expt_surftens = {
    286.95: 29.18,
    304.55: 26.81,
    315.95: 24.43,
    328.15: 23.82,
    340.15: 22.35,
}

#https://doi.org/10.1016/j.molliq.2014.10.017
expt_liq_density = {
    293.15: 782.04,
    298.15: 776.66,
    303.15: 771.24,
    308.15: 765.78,
    313.15: 760.29,
}

#https://doi.org/10.1063/1.1696002
expt_Pvap = {
    280.409: (4.997 * u.kPa).to_value(u.bar),
    286.941: (6.914 * u.kPa).to_value(u.bar),
    295.055: (10.244 * u.kPa).to_value(u.bar),
    298.713: (12.156 * u.kPa).to_value(u.bar),
    300.530: (13.187 * u.kPa).to_value(u.bar),
}

#https://doi.org/10.1063/1.1696002
expt_Hvap = {
    298.15: 809.382,
}
# Create an instance of the EsolvsConstants class   
ACN = EsolvsConstants(
    mol_wt=mol_wt,
    Tc=Tc,
    rhoc=rho_c,
    n_atoms=n_atoms,
    smiles_str=smiles_str,
    param_names=param_names,
    gaff_params=gaff_params,
    bounds_sig=bounds_sig,
    bounds_eps=bounds_eps,
    expt_liq_density=expt_liq_density,
    expt_surftens=expt_surftens,
    expt_Pvap=expt_Pvap,
    expt_Hvap=expt_Hvap
)


#STOPPED HERE
#Methanol (MeOH)
mol_wt = 41.05
rho_c = 1958.09 #kg/m^3 (DOI: 10.1021/j150462a016)
Tc = 547.9
n_atoms = 6
smiles_str = "CC#N"
param_names = (
    "sigma_C1", #C attached to H
    "sigma_C2", #C attached to N
    "sigma_H1", #Attached to outside C
    "sigma_N1",
    "epsilon_C1",
    "epsilon_C2",
    "epsilon_H1",
    "epsilon_N1",
)

#Fix me later
gaff_params = {
    "sigma_C1":3.400,
    "sigma_H1":2.65,
    "epsilon_C1":55.052,
    "epsilon_H1":7.901,
}

bounds_sig = [
                [2.0, 4.0],  # C1
                [2.0, 4.0],  # C2
                [1.5, 3.0],  # H1
                [2.0, 4.0],  # N #Check with EM what is reasonable here
            ]
bounds_eps = [
                [10.0,75.0],  # C1
                [10.0,75.0],  # C2
                [2.0, 10.0],  # H1
                [10.0, 75.0],  # N #Check with EM what is reasonable here
            ]


#(equal to dyn/cm)
#DOI: 10.1021/ja02201a003
expt_surftens = {
    286.95: 29.18,
    304.55: 26.81,
    315.95: 24.43,
    328.15: 23.82,
    340.15: 22.35,
}

#https://doi.org/10.1016/j.molliq.2014.10.017
expt_liq_density = {
    293.15: 782.04,
    298.15: 776.66,
    303.15: 771.24,
    308.15: 765.78,
    313.15: 760.29,
}

#https://doi.org/10.1063/1.1696002
expt_Pvap = {
    280.409: (4.997 * u.kPa).to_value(u.bar),
    286.941: (6.914 * u.kPa).to_value(u.bar),
    295.055: (10.244 * u.kPa).to_value(u.bar),
    298.713: (12.156 * u.kPa).to_value(u.bar),
    300.530: (13.187 * u.kPa).to_value(u.bar),
}

#https://doi.org/10.1063/1.1696002
expt_Hvap = {
    298.15: 809.382,
}
# Create an instance of the EsolvsConstants class   
ACN = EsolvsConstants(
    mol_wt=mol_wt,
    Tc=Tc,
    rhoc=rho_c,
    n_atoms=n_atoms,
    smiles_str=smiles_str,
    param_names=param_names,
    gaff_params=gaff_params,
    bounds_sig=bounds_sig,
    bounds_eps=bounds_eps,
    expt_liq_density=expt_liq_density,
    expt_surftens=expt_surftens,
    expt_Pvap=expt_Pvap,
    expt_Hvap=expt_Hvap
)