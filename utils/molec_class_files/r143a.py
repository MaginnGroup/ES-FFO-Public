import numpy as np
import unyt as u

class R143aConstants:
    """Experimental data and other constants for R143a"""
    def __init__(self):
        assert (
            self.expt_liq_density.keys()
            == self.expt_vap_density.keys()
            == self.expt_Pvap.keys()
            == self.expt_Hvap.keys()
        )

    @property
    def molecular_weight(self):
        """Molecular weight of the molecule in g/mol"""
        return 84.04

    @property
    def expt_Tc(self):
        """Critical temperature in K"""
        return 345.88

    @property
    def expt_rhoc(self):
        """Critical density in kg/m^3"""
        return 431

    @property
    def n_atoms(self):
        """Number of atoms in molecule"""
        return 8
        
    @property
    def smiles_str(self):
        """Smiles string representation"""
        return "C(F)(F)(F)C"
    
    @property
    def n_params(self):
        """Number of adjustable parameters"""
        return len(self.param_names)

    @property
    def param_names(self):
        """Adjustable parameter names"""

        param_names = (
            "sigma_C1",
            "sigma_C2",
            "sigma_F1",
            "sigma_H1",
            "epsilon_C1",
            "epsilon_C2",
            "epsilon_F1",
            "epsilon_H1",
        )

        return param_names
    
    @property
    def lit_param_set(self):
        """Adjustable parameter names"""

        lit_param_set = {
            "sigma_C1":3.8090599247748402,
            "sigma_C2": 3.037119312093674,
            "sigma_F1":3.424309963103194,
            "sigma_H1":1.711397937309018,
            "epsilon_C1":69.97073896121991,
            "epsilon_C2":55.40701504905475,
            "epsilon_F1":17.051878638985517,
            "epsilon_H1":3.5509367860039087,
        }

        return lit_param_set
    
    @property
    def gaff_param_set(self):
        """Adjustable parameter names"""

        gaff_param_set = {
            "sigma_C1":3.400,
            "sigma_C2": 3.400,
            "sigma_F1":3.118,
            "sigma_H1":2.65,
            "epsilon_C1":55.052,
            "epsilon_C2":55.052,
            "epsilon_F1":30.696,
            "epsilon_H1":7.901,
        }

        return gaff_param_set

    @property
    def param_bounds(self):
        """Bounds on sigma and epsilon in units of nm and kJ/mol"""

        bounds_sigma = (
            (
                np.asarray(
                    [
                        [2.0,4.0], #[3.0, 4.0],  # C1
                        [2.0,4.0], #[3.0, 4.0],  # C2
                        [2.0, 4.0],  # F1
                        [1.5, 3.0],  # H1
                    ]
                )
                * u.Angstrom
            )
            .in_units(u.nm)
            .value
        )

        bounds_epsilon = (
            (
                np.asarray(
                    [
                        [10.0,75.0], #[20.0, 70.0],  # C1
                        [10.0,75.0], #[20.0, 70.0],  # C2
                        [15.0,50.0], #[15.0, 40.0],  # F1
                        [2.0, 10.0],  # H1
                    ]
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
    def expt_liq_density(self):
        """Dictionary with experimental liquid density

        Temperature units K
        Density units kg/m**3
        """

        expt_liq_density = {
            240: 1127.0,
            260: 1067.2,
            280: 1000.4,
            300: 922.32,
            320: 822.31,
        }

        return expt_liq_density

    @property
    def expt_vap_density(self):
        """Dictionary with experimental vapor density

        Temperature units K
        Density units kg/m**3
        """

        expt_vap_density = {
            240: 8.5946,
            260: 17.796,
            280: 33.738,
            300: 60.818,
            320: 109.18,
        }

        return expt_vap_density

    @property
    def expt_Pvap(self):
        """Dictionary with experimental vapor pressure

        Temperature units K
        Vapor pressure units bar
        """

        expt_Pvap = {
            240: (189.02 * u.kPa).to_value(u.bar),
            260: (402.51 * u.kPa).to_value(u.bar),
            280: (762.76 * u.kPa).to_value(u.bar),
            300: (1323.4 * u.kPa).to_value(u.bar),
            320: (2148.3 * u.kPa).to_value(u.bar),
        }

        return expt_Pvap

    @property
    def expt_Hvap(self):
        """Dictionary with experimental enthalpy of vaporization

        Temperature units K
        Enthalpy of vaporization units kJ/kg
        """

        expt_Hvap = {
            240: 216.4,
            260: 200.09,
            280: 180.78,
            300: 156.88,
            320: 124.85,
        }

        return expt_Hvap

    @property
    def uncertainties(self):
        """
        Dictionary with uncertainty for each calculation
        from: 
        Lemmon, E. and Jacobsen, R. (2000), An International Standard Formulation for 
        the Thermodynamic Properties of 1,1,1-Trifluoroethane (HFC-143a) for Temperatures from 161 to 450 K 
        and Pressures to 50 Mpa, J. Phys. & Chem. Ref. Data (JPCRD), National Institute of Standards and Technology, 
        Gaithersburg, MD (Accessed February 22, 2024)
        """
        uncertainty = {
            "expt_liq_density": 0.001,
            "expt_vap_density": 0.001,
            "expt_Pvap": 0.001,
            "expt_Hvap": 0.005
        }
        return uncertainty
    
    @property
    def temperature_bounds(self):
        """Bounds on temperature in units of K"""

        lower_bound = np.min(list(self.expt_Pvap.keys()))
        upper_bound = np.max(list(self.expt_Pvap.keys()))
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
