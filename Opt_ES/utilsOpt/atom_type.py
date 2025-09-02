import numpy as np
import unyt as u


def make_atom_type_class(at_number):
    """
    Creates an atom type class based on the atom type number

    Parameters
    ----------
    at_number: int, The atom type number

    Returns
    -------
    class, The atom type class
    """
    if at_number == 1: #C, H, O, N, S, Cl
        return AT_Scheme_01()
    elif at_number == 2: #GAFF
        return AT_Scheme_02()
    elif at_number == 0: #Distinct Atom Types
        return AT_Scheme_Dist()
    else:
        raise ValueError("Invalid atom type number")


class Atom_Types:
    """
    Base class for atom typing schemes

    Methods
    -------
    __init__(at_bounds, at_names, molec_map_dicts)
    get_transformation_matrix(self, molec_key)
    check_for_duplicates(self)
    """

    def __init__(self, at_bounds, at_names, molec_map_dicts):
        """
        Initialization Method:

        Parameters
        ----------
        at_bounds: array, The bounds of the new atom type scheme (sigma in A, epsilon in K)
        at_names: list, The names of the new atom types. Should correspond to each element in at_bounds.shape[1]
        molec_map_dicts: dict, The dictionary of molecule property class for the old atom types
        """
        assert isinstance(at_bounds, np.ndarray), "at_bounds must be an np.ndarray"
        assert isinstance(at_names, list), "at_names must be a list"
        assert isinstance(molec_map_dicts, dict), "molec_map_dicts must be a dictionary"
        assert (
            all(isinstance(name, str) for name in at_names) == True
        ), "all at_names must be string"
        assert (
            len(at_names) == at_bounds.shape[0]
        ), "at_bounds must have one column for each name in at_names"

        self.at_bounds = at_bounds
        self.at_names = at_names
        self.molec_map_dicts = molec_map_dicts
        self.at_matrices = {}
        self.at_bounds_nm_kjmol = self.scale_bounds()

    def scale_bounds(self):
        """
        Scales bounds to units of nm and kj/mol and creates self.at_bounds_nm_kjmol
        """
        # Get upper and lower bounds seperately
        bounds_list = [self.at_bounds[:, x] for x in range(self.at_bounds.shape[1])]
        at_bounds_nm_kjmol = np.zeros(self.at_bounds.shape)
        # Get Midpoint of bounds
        midpoint = len(bounds_list[0]) // 2
        # Loop over upper and lower bounds
        for i in range(len(bounds_list)):
            # Create scaled list of upper and lower bounds for sigmas and epsilons
            sigmas = [
                float((x * u.Angstrom).in_units(u.nm).value)
                for x in bounds_list[i][:midpoint]
            ]
            epsilons = [
                float((x * u.K * u.kb).in_units("kJ/mol").value)
                for x in bounds_list[i][midpoint:]
            ]
            # Combine the results and add to array
            new_bound = np.array(sigmas + epsilons)
            at_bounds_nm_kjmol[:, i] = new_bound

        # self.at_bounds_nm_kjmol = at_bounds_nm_kjmol
        return at_bounds_nm_kjmol

    def get_transformation_matrix(self, molec_map_dict):
        """
        Creates transformation matrix between new and old atom types

        Parameters:
        -----------
        molec_map_dict: dict, The dictionary of molecule property class for the old atom types

        Returns:
        --------
        at_matrix: array, The transformation matrix from new to old atom types
        """
        molec_key = list(molec_map_dict.keys())[0]
        molec_data = molec_map_dict[molec_key]
        assert molec_key in self.molec_map_dicts
        # If you already have this matrix, use it. Otherwise generate it
        if not molec_key in self.at_matrices:
            # Get mapping for specific molecule
            map_dict = self.molec_map_dicts[molec_key]
            # Create a matrix based on the keys and map dict length
            at_matrix = np.zeros((len(self.at_names), len(map_dict)))
            # Fill at_matrix with ones or zeros based on the presence of keys
            for i, value in enumerate(self.at_names):
                if value in map_dict.values():
                    indices = [i for i, v in enumerate(map_dict.values()) if v == value]
                    # at_matrix[i, list(map_dict.values()).index(value)] = 1
                    at_matrix[i, indices] = 1
            # Add matrix to self dictionary
            self.at_matrices[molec_key] = at_matrix
        else:
            at_matrix = self.at_matrices[molec_key]

        # Ensure correct order of matrix
        order = molec_data.param_names
        index_mapping = {
            elem: idx
            for idx, elem in enumerate(list(self.molec_map_dicts[molec_key].keys()))
        }
        mapped_indices = [index_mapping[elem] for elem in order]
        at_matrix = at_matrix[:, mapped_indices]
        return at_matrix

    def check_for_duplicates(self):
        """
        Checks for duplicate matricies in at_matrix

        Returns:
        --------
        bool, True if duplicates are present, False otherwise
        """
        arr_list = list(self.at_matrices.values())
        tuple_list = [tuple(map(tuple, arr)) for arr in arr_list]
        len(tuple_list) != len(set(tuple_list))

        return len(tuple_list) != len(set(tuple_list))

class AT_Scheme_01(Atom_Types):
    """
    Class for Atom Type Scheme 1 (C, H, O, N, S, Cl)

    Methods
    -------
    __init__(self)
    """

    def __init__(self):
        # Get Bounds
        at_param_bounds_l = [2.0, 0.0, 2.0,
                             10.0, 0.0, 75.0]  # Units of Angstroms and Kelvin for Sigmas and Epsilons
        # at_param_bounds_l = [2.0, 0.0, 2.0, 2.0, 2.5, 2.5,
        #                      10.0, 0.0, 40.0, 40.0, 90.0, 90.0]  # Units of Angstroms and Kelvin for Sigmas and Epsilons
        at_param_bounds_u = [4.0, 3.0, 4.0,
                             75.0, 10.0, 135.0]  # Units of Angstroms and Kelvin for Sigmas and Epsilons
        # at_param_bounds_u = [4.0, 3.0, 4.0, 4.0, 4.5, 4.5,
        #                      135.0, 10.0, 135.0, 100.0, 150.0, 150.0]  # Units of Angstroms and Kelvin for Sigmas and Epsilons
        at_bounds = np.array([at_param_bounds_l, at_param_bounds_u]).T
        self.scheme_name = "at_01"
        self.scheme_plot_name = "AT-3" #C, H, O
        # self.scheme_plot_name = "AT-4" #C, H, O, N (S, Cl)
        # Get Names
        at_keys = [
            "sigma_C",
            "sigma_H",
            "sigma_O",
            "epsilon_C",
            "epsilon_H",
            "epsilon_O",
        ]
        # at_keys = [
        #     "sigma_C",
        #     "sigma_H",
        #     "sigma_O",
        #     "sigma_N",
        #     "sigma_S",
        #     "sigma_Cl",
        #     "epsilon_C",
        #     "epsilon_H",
        #     "epsilon_O",
        #     "epsilon_N",
        #     "epsilon_S",
        #     "epsilon_Cl",
        # ]
        assert (
            len(at_keys) == len(at_param_bounds_l) == len(at_param_bounds_u)
        ), "Length of at_keys, at_param_bounds_l, and at_param_bounds_u must be the same"
        # Get weight information
        # self.no_opt = ["sigma_Cl", "sigma_S", "epsilon_Cl", "epsilon_S"]

        # for key in self.no_opt:
        #     idx = at_keys.index(key)
        #     at_bounds[idx, 1] = at_bounds[idx, 0]
            
        # Create a file that maps param names (keys) to at_param names for atom type 11 (values) for each molecule
        EG_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_H2": "sigma_H",# Attached to O
            "sigma_H1": "sigma_H",  # Attached to C
            "sigma_O1": "sigma_O",
            "epsilon_C1": "epsilon_C",
            "epsilon_H2": "epsilon_H",
            "epsilon_H1": "epsilon_H",  # Attached to C
            "epsilon_O1": "epsilon_O",
        }

        Gly_map_dict = {
            "sigma_C1": "sigma_C",  # Outside Cs
            "sigma_C2": "sigma_C",  # Middle C
            "sigma_H3": "sigma_H",  # Attached to O1
            "sigma_H4": "sigma_H",  # Attached to O2
            "sigma_H1": "sigma_H",  # Attached to C1
            "sigma_H2": "sigma_H",  # Attached to C2
            "sigma_O1": "sigma_O",  # Outside O
            "sigma_O2": "sigma_O",  # Inside O
            "epsilon_C1": "epsilon_C",  # Outside Cs
            "epsilon_C2": "epsilon_C",  # Middle C
            "epsilon_H3": "epsilon_H",  # Attached to O1
            "epsilon_H4": "epsilon_H",  # Attached to O2
            "epsilon_H1": "epsilon_H",  # Attached to C1
            "epsilon_H2": "epsilon_H",  # Attached to C2
            "epsilon_O1": "epsilon_O",  # Outside O
            "epsilon_O2": "epsilon_O",  # Inside O
        }

        # ACN_map_dict = {
        #     "sigma_C1": "sigma_C",  # C attached to N
        #     "sigma_C2": "sigma_C",  # C attached to H
        #     "sigma_H1": "sigma_H",  # Attached to C2
        #     "sigma_N1": "sigma_N",
        #     "epsilon_C1": "epsilon_C",  # C attached to N
        #     "epsilon_C2": "epsilon_C",  # C attached to H
        #     "epsilon_H1": "epsilon_H",  # Attached to C2
        #     "epsilon_N1": "epsilon_N",
        # }

        MeOH_map_dict = {
            "sigma_C1": "sigma_C",  # C attached to H
            "sigma_H1": "sigma_H",  # Attached to C
            "sigma_H2": "sigma_H",  # Attached to O
            "sigma_O1": "sigma_O",
            "epsilon_C1": "epsilon_C",  # C attached to H
            "epsilon_H1": "epsilon_H",  # Attached to C
            "epsilon_H2": "epsilon_H",  # Attached to O
            "epsilon_O1": "epsilon_O",
        }

        # DMF_map_dict = {
        #     "sigma_C1": "sigma_C",  # C attached to N and H
        #     "sigma_C2": "sigma_C",  # C attached to O, and N
        #     "sigma_H2": "sigma_H",  # Attached to C2
        #     "sigma_H1": "sigma_H",  # Attached to C1
        #     "sigma_O1": "sigma_O",
        #     "sigma_N1": "sigma_N",
        #     "epsilon_C1": "epsilon_C",  # C attached to N and H
        #     "epsilon_C2": "epsilon_C",  # C attached to O, and N
        #     "epsilon_H2": "epsilon_H",  # Attached to C2
        #     "epsilon_H1": "epsilon_H",  # Attached to C1
        #     "epsilon_O1": "epsilon_O",
        #     "epsilon_N1": "epsilon_N",
        # }

        # DMSO_map_dict = {
        #     "sigma_C1": "sigma_C",  # C attached to S
        #     "sigma_H1": "sigma_H",  # Attached to C1
        #     "sigma_O1": "sigma_O",  # Attached to S
        #     "sigma_S1": "sigma_S",
        #     "epsilon_C1": "epsilon_C",  # C attached to S
        #     "epsilon_H1": "epsilon_H",  # Attached to C1
        #     "epsilon_O1": "epsilon_O",  # Attached to S
        #     "epsilon_S1": "epsilon_S",
        # }

        # THF_map_dict = {
        #     "sigma_C1": "sigma_C",  # C attached to O
        #     "sigma_C2": "sigma_C",  # C not attached to O
        #     "sigma_H2": "sigma_H",  # Attached to C2
        #     "sigma_H1": "sigma_H",  # Attached to C1
        #     "sigma_O1": "sigma_O",
        #     "epsilon_C1": "epsilon_C",  # C attached to O
        #     "epsilon_C2": "epsilon_C",  # C not attached to O
        #     "epsilon_H2": "epsilon_H",  # Attached to C2
        #     "epsilon_H1": "epsilon_H",  # Attached to C1
        #     "epsilon_O1": "epsilon_O",
        # }

        # DEC_map_dict = {
        #     "sigma_C3": "sigma_C",  # C attached to three O
        #     "sigma_C1": "sigma_C",  # Outsidemost C
        #     "sigma_C2": "sigma_C",  # C attached to one O
        #     "sigma_H1": "sigma_H",  # H attached to C1
        #     "sigma_H2": "sigma_H",  # H attached to C2
        #     "sigma_O2": "sigma_O",  # O with double bond
        #     "sigma_O1": "sigma_O",  # O with no double bond
        #     "epsilon_C3": "epsilon_C",  # C attached to three O
        #     "epsilon_C1": "epsilon_C",  # Outsidemost C
        #     "epsilon_C2": "epsilon_C",  # C attached to one O
        #     "epsilon_H1": "epsilon_H",  # H attached to C1
        #     "epsilon_H2": "epsilon_H",  # H attached to C2
        #     "epsilon_O2": "epsilon_O",  # O with double bond
        #     "epsilon_O1": "epsilon_O",  # O with no double bond
        # }

        # # Test molecules
        # DCM_map_dict = {
        #     "sigma_C1": "sigma_C",
        #     "sigma_H1": "sigma_H",
        #     "sigma_Cl1": "sigma_Cl",
        #     "epsilon_C1": "epsilon_C",
        #     "epsilon_H1": "epsilon_H",
        #     "epsilon_Cl1": "epsilon_Cl",
        # }


        at_names = at_keys.copy()

        # molec_map_dicts = {
        #     "EG": EG_map_dict,
        #     "Gly": Gly_map_dict,
        #     "ACN": ACN_map_dict,
        #     "MeOH": MeOH_map_dict,
        #     "DMF": DMF_map_dict,
        #     "DMSO": DMSO_map_dict,
        #     "THF": THF_map_dict,
        #     "DEC": DEC_map_dict,
        #     "DCM": DCM_map_dict,
        # }

        molec_map_dicts = {
            "EG": EG_map_dict,
            "Gly": Gly_map_dict,
            "MeOH": MeOH_map_dict,
        }

        super().__init__(at_bounds, at_names, molec_map_dicts)
        # Get scaled bounds
        self.scale_bounds()

class AT_Scheme_02(Atom_Types):
    """
    Class for Atom Type Scheme 2 (GAFF)

    Methods
    -------
    __init__(self)
    """

    def __init__(self):
        # Get Bounds
        at_param_bounds_l = [2.0, 0.0, 1.5, 2.0,
                             10.0, 0.0, 2.0, 75.0,]  # Units of Angstroms and Kelvin for Sigmas and Epsilons
        at_param_bounds_u = [4.0, 0.0, 3.0, 4.0,
                             75.0, 0.0, 10.0, 135.0,]  # Units of Angstroms and Kelvin for Sigmas and Epsilons
        # at_param_bounds_l = [2.0, 2.0, 2.0, 1.5, 0.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5,
        #                      10.0, 75.0, 10.0, 2.0, 0.0, 2.0, 2.0, 2.0, 75.0, 75.0, 40.0, 40.0, 40.0, 90.0, 90.0]  # Units of Angstroms and Kelvin for Sigmas and Epsilons
        # at_param_bounds_u = [4.0, 4.0, 4.0, 3.0, 0.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.5, 4.5,
        #                      75.0, 135.0, 75.0, 10.0, 0.0, 10.0, 10.0, 10.0, 135.0, 135.0, 100.00, 100.00, 100.0, 150.0, 150.0]  # Units of Angstroms and Kelvin for Sigmas and Epsilons
        at_bounds = np.array([at_param_bounds_l, at_param_bounds_u]).T
        self.scheme_name = "at_02"
        # self.scheme_plot_name = "AT-7" #C3, O, Os, Oh, Hc, H1, Ho
        self.scheme_plot_name = "AT-4" #C3, Oh, H1, Ho
        # Get Names
        at_keys = [
            "sigma_C3",
            "sigma_Ho",
            "sigma_H1",
            "sigma_Oh",
            "epsilon_C3",
            "epsilon_Ho",
            "epsilon_H1",
            "epsilon_Oh",
        ]

        # at_keys = [
        #     "sigma_C",
        #     "sigma_C1",
        #     "sigma_C3",
        #     "sigma_Hc",
        #     "sigma_Ho",
        #     "sigma_H1",
        #     "sigma_H2",
        #     "sigma_H5",
        #     "sigma_O",
        #     "sigma_Oh",
        #     "sigma_Os",
        #     "sigma_N",
        #     "sigma_N1",
        #     "sigma_S4",
        #     "sigma_Cl",
        #     "epsilon_C",
        #     "epsilon_C1",
        #     "epsilon_C3",
        #     "epsilon_Hc",
        #     "epsilon_Ho",
        #     "epsilon_H1",
        #     "epsilon_H2",
        #     "epsilon_H5",
        #     "epsilon_O",
        #     "epsilon_Oh",
        #     "epsilon_Os",
        #     "epsilon_N",
        #     "epsilon_N1",
        #     "epsilon_S4",
        #     "epsilon_Cl",
        # ]
        assert (
            len(at_keys) == len(at_param_bounds_l) == len(at_param_bounds_u)
        ), "Length of at_keys, at_param_bounds_l, and at_param_bounds_u must be the same"
        # Get weight information
        # self.no_opt = ["sigma_C", 
        #           "sigma_C1", 
        #           "sigma_H2", 
        #           "sigma_H5", 
        #           "sigma_N", 
        #           "sigma_N1", 
        #           "sigma_S4", 
        #           "sigma_Cl", 
        #           "epsilon_C", 
        #           "epsilon_C1", 
        #           "epsilon_H2", 
        #           "epsilon_H5", 
        #           "epsilon_N", 
        #           "epsilon_N1", 
        #           "epsilon_S4", 
        #           "epsilon_Cl"]

        # for key in self.no_opt:
        #     idx = at_keys.index(key)
        #     at_bounds[idx, 1] = at_bounds[idx, 0]
            
        # Create a file that maps param names (keys) to at_param names for atom type 11 (values) for each molecule
        EG_map_dict = {"sigma_C1":"sigma_C3",
                       "sigma_H2": "sigma_Ho",# Attached to O
                        "sigma_H1": "sigma_H1",  # Attached to C
                        "sigma_O1": "sigma_Oh",
                        "epsilon_C1": "epsilon_C3",
                        "epsilon_H2" : "epsilon_Ho",
                        "epsilon_H1": "epsilon_H1",  # Attached to C
                        "epsilon_O1": "epsilon_Oh",
        }

        Gly_map_dict = {
            "sigma_C1": "sigma_C3",  # Outside Cs
            "sigma_C2": "sigma_C3",  # Middle C
            "sigma_H3": "sigma_Ho",  # Attached to O1
            "sigma_H4": "sigma_Ho",  # Attached to O2
            "sigma_H1": "sigma_H1",  # Attached to C1
            "sigma_H2": "sigma_H1",  # Attached to C2
            "sigma_O1": "sigma_Oh",  # Outside O
            "sigma_O2": "sigma_Oh",  # Inside O
            "epsilon_C1": "epsilon_C3",  # Outside Cs
            "epsilon_C2": "epsilon_C3",  # Middle C
            "epsilon_H3": "epsilon_Ho",  # Attached to O1
            "epsilon_H4": "epsilon_Ho",  # Attached to O2
            "epsilon_H1": "epsilon_H1",  # Attached to C1
            "epsilon_H2": "epsilon_H1",  # Attached to C2
            "epsilon_O1": "epsilon_Oh",  # Outside O
            "epsilon_O2": "epsilon_Oh",  # Inside O
        }

        # ACN_map_dict = {
        #     "sigma_C1": "sigma_C1",  # C attached to N
        #     "sigma_C2": "sigma_C3",  # C attached to H
        #     "sigma_H1": "sigma_Hc",  # Attached to C2
        #     "sigma_N1": "sigma_N1",
        #     "epsilon_C1": "epsilon_C1",  # C attached to N
        #     "epsilon_C2": "epsilon_C3",  # C attached to H
        #     "epsilon_H1": "epsilon_Hc",  # Attached to C2
        #     "epsilon_N1": "epsilon_N1",
        # }

        MeOH_map_dict = {
            "sigma_C1": "sigma_C3",  # C attached to H
            "sigma_H1": "sigma_Ho",  # Attached to C
            "sigma_H2": "sigma_H1",  # Attached to O
            "sigma_O1": "sigma_Oh",
            "epsilon_C1": "epsilon_C3",  # C attached to H
            "epsilon_H1": "epsilon_Ho",  # Attached to C
            "epsilon_H2": "epsilon_H1",  # Attached to O
            "epsilon_O1": "epsilon_Oh",
        }

        # DMF_map_dict = {
        #     "sigma_C1": "sigma_C",  # C attached to N and H
        #     "sigma_C2": "sigma_C3",  # C attached to O, and N
        #     "sigma_H2": "sigma_H1",  # Attached to C2
        #     "sigma_H1": "sigma_H5",  # Attached to C1
        #     "sigma_O1": "sigma_O",
        #     "sigma_N1": "sigma_N",
        #     "epsilon_C1": "epsilon_C",  # C attached to N and H
        #     "epsilon_C2": "epsilon_C3",  # C attached to O, and N
        #     "epsilon_H2": "epsilon_H1",  # Attached to C2
        #     "epsilon_H1": "epsilon_H5",  # Attached to C1
        #     "epsilon_O1": "epsilon_O",
        #     "epsilon_N1": "epsilon_N",
        # }

        # DMSO_map_dict = {
        #     "sigma_C1": "sigma_C3",  # C attached to S
        #     "sigma_H1": "sigma_H1",  # Attached to C1
        #     "sigma_O1": "sigma_O",  # Attached to S
        #     "sigma_S1": "sigma_S4",
        #     "epsilon_C1": "epsilon_C3",  # C attached to S
        #     "epsilon_H1": "epsilon_H1",  # Attached to C1
        #     "epsilon_O1": "epsilon_O",  # Attached to S
        #     "epsilon_S1": "epsilon_S4",
        # }

        # THF_map_dict = {
        #     "sigma_C1": "sigma_C3",  # C attached to O
        #     "sigma_C2": "sigma_C3",  # C not attached to O
        #     "sigma_H2": "sigma_Hc",  # Attached to C2
        #     "sigma_H1": "sigma_H1",  # Attached to C1
        #     "sigma_O1": "sigma_Os",
        #     "epsilon_C1": "epsilon_C3",  # C attached to O
        #     "epsilon_C2": "epsilon_C3",  # C not attached to O
        #     "epsilon_H2": "epsilon_Hc",  # Attached to C2
        #     "epsilon_H1": "epsilon_H1",  # Attached to C1
        #     "epsilon_O1": "epsilon_Os",
        # }

        # DEC_map_dict = {
        #     "sigma_C3": "sigma_C",  # C attached to three O
        #     "sigma_C1": "sigma_C3",  # Outsidemost C
        #     "sigma_C2": "sigma_C3",  # C attached to one O
        #     "sigma_H1": "sigma_Hc",  # H attached to C1
        #     "sigma_H2": "sigma_H1",  # H attached to C2
        #     "sigma_O2": "sigma_O",  # O with double bond
        #     "sigma_O1": "sigma_Os",  # O with no double bond
        #     "epsilon_C3": "epsilon_C",  # C attached to three O
        #     "epsilon_C1": "epsilon_C3",  # Outsidemost C
        #     "epsilon_C2": "epsilon_C3",  # C attached to one O
        #     "epsilon_H1": "epsilon_Hc",  # H attached to C1
        #     "epsilon_H2": "epsilon_H1",  # H attached to C2
        #     "epsilon_O2": "epsilon_O",  # O with double bond
        #     "epsilon_O1": "epsilon_Os",  # O with no double bond
        # }

        # # Test molecules
        # DCM_map_dict = {
        #     "sigma_C1": "sigma_C3",
        #     "sigma_H1": "sigma_H2",
        #     "sigma_Cl1": "sigma_Cl",
        #     "epsilon_C1": "epsilon_C3",
        #     "epsilon_H1": "epsilon_H2",
        #     "epsilon_Cl1": "epsilon_Cl",
        # }


        at_names = at_keys.copy()

        # molec_map_dicts = {
        #     "EG": EG_map_dict,
        #     "Gly": Gly_map_dict,
        #     "ACN": ACN_map_dict,
        #     "MeOH": MeOH_map_dict,
        #     "DMF": DMF_map_dict,
        #     "DMSO": DMSO_map_dict,
        #     "THF": THF_map_dict,
        #     "DEC": DEC_map_dict,
        #     "DCM": DCM_map_dict,
        # }

        molec_map_dicts = {
            "EG": EG_map_dict,
            "Gly": Gly_map_dict,
            "MeOH": MeOH_map_dict,
        }

        super().__init__(at_bounds, at_names, molec_map_dicts)
        # Get scaled bounds
        self.scale_bounds()

class AT_Scheme_Dist(Atom_Types):
    """
    Class for Atom Type Scheme 0 (Distinct Atom Types)

    Methods
    -------
    __init__(self)
    """

    def __init__(self):

        self.scheme_name = "at_00"
        self.scheme_plot_name = "AT-Dis"
        #Note: Everything below this point is irrelevant as with distinct ATs, only the scheme name and plot name are used
        
        # Get Bounds
        at_param_bounds_l = [2.0, 2.0, 2.0, 1.5, 0.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5,
                             10.0, 75.0, 10.0, 2.0, 0.0, 2.0, 2.0, 2.0, 75.0, 75.0, 40.0, 40.0, 40.0, 90.0, 90.0]  # Units of Angstroms and Kelvin for Sigmas and Epsilons
        at_param_bounds_u = [4.0, 4.0, 4.0, 3.0, 0.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.5, 4.5,
                             75.0, 135.0, 75.0, 10.0, 0.0, 10.0, 10.0, 10.0, 135.0, 135.0, 100.00, 100.00, 100.0, 150.0, 150.0]  # Units of Angstroms and Kelvin for Sigmas and Epsilons
        at_bounds = np.array([at_param_bounds_l, at_param_bounds_u]).T
        
        # Get Names
        at_keys = [
            "sigma_C",
            "sigma_C1",
            "sigma_C3",
            "sigma_Hc",
            "sigma_Ho",
            "sigma_H1",
            "sigma_H2",
            "sigma_H5",
            "sigma_O",
            "sigma_Oh",
            "sigma_Os",
            "sigma_N",
            "sigma_N1",
            "sigma_S4",
            "sigma_Cl",
            "epsilon_C",
            "epsilon_C1",
            "epsilon_C3",
            "epsilon_Hc",
            "epsilon_Ho",
            "epsilon_H1",
            "epsilon_H2",
            "epsilon_H5",
            "epsilon_O",
            "epsilon_Oh",
            "epsilon_Os",
            "epsilon_N",
            "epsilon_N1",
            "epsilon_S4",
            "epsilon_Cl",
        ]
        assert (
            len(at_keys) == len(at_param_bounds_l) == len(at_param_bounds_u)
        ), "Length of at_keys, at_param_bounds_l, and at_param_bounds_u must be the same"
        # Get weight information
        self.no_opt = ["sigma_C", 
                  "sigma_C1", 
                  "sigma_H2", 
                  "sigma_H5", 
                  "sigma_N", 
                  "sigma_N1", 
                  "sigma_S4", 
                  "sigma_Cl", 
                  "epsilon_C", 
                  "epsilon_C1", 
                  "epsilon_H2", 
                  "epsilon_H5", 
                  "epsilon_N", 
                  "epsilon_N1", 
                  "epsilon_S4", 
                  "epsilon_Cl"]

        for key in self.no_opt:
            idx = at_keys.index(key)
            at_bounds[idx, 1] = at_bounds[idx, 0]
            
        # Create a file that maps param names (keys) to at_param names for atom type 11 (values) for each molecule
        EG_map_dict = {"sigma_C1":"sigma_C3",
                       "sigma_H2": "sigma_Ho",# Attached to O
                        "sigma_H1": "sigma_H1",  # Attached to C
                        "sigma_O1": "sigma_Oh",
                        "epsilon_C1": "epsilon_C3",
                        "epsilon_H2" : "epsilon_Ho",
                        "epsilon_H1": "epsilon_H1",  # Attached to C
                        "epsilon_O1": "epsilon_Oh",
        }

        Gly_map_dict = {
            "sigma_C1": "sigma_C3",  # Outside Cs
            "sigma_C2": "sigma_C3",  # Middle C
            "sigma_H3": "sigma_Ho",  # Attached to O1
            "sigma_H4": "sigma_Ho",  # Attached to O2
            "sigma_H1": "sigma_H1",  # Attached to C1
            "sigma_H2": "sigma_H1",  # Attached to C2
            "sigma_O1": "sigma_Oh",  # Outside O
            "sigma_O2": "sigma_Oh",  # Inside O
            "epsilon_C1": "epsilon_C3",  # Outside Cs
            "epsilon_C2": "epsilon_C3",  # Middle C
            "epsilon_H3": "epsilon_Ho",  # Attached to O1
            "epsilon_H4": "epsilon_Ho",  # Attached to O2
            "epsilon_H1": "epsilon_H1",  # Attached to C1
            "epsilon_H2": "epsilon_H1",  # Attached to C2
            "epsilon_O1": "epsilon_Oh",  # Outside O
            "epsilon_O2": "epsilon_Oh",  # Inside O
        }

        ACN_map_dict = {
            "sigma_C1": "sigma_C1",  # C attached to N
            "sigma_C2": "sigma_C3",  # C attached to H
            "sigma_H1": "sigma_Hc",  # Attached to C2
            "sigma_N1": "sigma_N1",
            "epsilon_C1": "epsilon_C1",  # C attached to N
            "epsilon_C2": "epsilon_C3",  # C attached to H
            "epsilon_H1": "epsilon_Hc",  # Attached to C2
            "epsilon_N1": "epsilon_N1",
        }

        MeOH_map_dict = {
            "sigma_C1": "sigma_C3",  # C attached to H
            "sigma_H1": "sigma_Ho",  # Attached to C
            "sigma_H2": "sigma_H1",  # Attached to O
            "sigma_O1": "sigma_Oh",
            "epsilon_C1": "epsilon_C3",  # C attached to H
            "epsilon_H1": "epsilon_Ho",  # Attached to C
            "epsilon_H2": "epsilon_H1",  # Attached to O
            "epsilon_O1": "epsilon_Oh",
        }

        DMF_map_dict = {
            "sigma_C1": "sigma_C",  # C attached to N and H
            "sigma_C2": "sigma_C3",  # C attached to O, and N
            "sigma_H2": "sigma_H1",  # Attached to C2
            "sigma_H1": "sigma_H5",  # Attached to C1
            "sigma_O1": "sigma_O",
            "sigma_N1": "sigma_N",
            "epsilon_C1": "epsilon_C",  # C attached to N and H
            "epsilon_C2": "epsilon_C3",  # C attached to O, and N
            "epsilon_H2": "epsilon_H1",  # Attached to C2
            "epsilon_H1": "epsilon_H5",  # Attached to C1
            "epsilon_O1": "epsilon_O",
            "epsilon_N1": "epsilon_N",
        }

        DMSO_map_dict = {
            "sigma_C1": "sigma_C3",  # C attached to S
            "sigma_H1": "sigma_H1",  # Attached to C1
            "sigma_O1": "sigma_O",  # Attached to S
            "sigma_S1": "sigma_S4",
            "epsilon_C1": "epsilon_C3",  # C attached to S
            "epsilon_H1": "epsilon_H1",  # Attached to C1
            "epsilon_O1": "epsilon_O",  # Attached to S
            "epsilon_S1": "epsilon_S4",
        }

        THF_map_dict = {
            "sigma_C1": "sigma_C3",  # C attached to O
            "sigma_C2": "sigma_C3",  # C not attached to O
            "sigma_H2": "sigma_Hc",  # Attached to C2
            "sigma_H1": "sigma_H1",  # Attached to C1
            "sigma_O1": "sigma_Os",
            "epsilon_C1": "epsilon_C3",  # C attached to O
            "epsilon_C2": "epsilon_C3",  # C not attached to O
            "epsilon_H2": "epsilon_Hc",  # Attached to C2
            "epsilon_H1": "epsilon_H1",  # Attached to C1
            "epsilon_O1": "epsilon_Os",
        }

        DEC_map_dict = {
            "sigma_C3": "sigma_C",  # C attached to three O
            "sigma_C1": "sigma_C3",  # Outsidemost C
            "sigma_C2": "sigma_C3",  # C attached to one O
            "sigma_H1": "sigma_Hc",  # H attached to C1
            "sigma_H2": "sigma_H1",  # H attached to C2
            "sigma_O2": "sigma_O",  # O with double bond
            "sigma_O1": "sigma_Os",  # O with no double bond
            "epsilon_C3": "epsilon_C",  # C attached to three O
            "epsilon_C1": "epsilon_C3",  # Outsidemost C
            "epsilon_C2": "epsilon_C3",  # C attached to one O
            "epsilon_H1": "epsilon_Hc",  # H attached to C1
            "epsilon_H2": "epsilon_H1",  # H attached to C2
            "epsilon_O2": "epsilon_O",  # O with double bond
            "epsilon_O1": "epsilon_Os",  # O with no double bond
        }

        # Test molecules
        DCM_map_dict = {
            "sigma_C1": "sigma_C3",
            "sigma_H1": "sigma_H2",
            "sigma_Cl1": "sigma_Cl",
            "epsilon_C1": "epsilon_C3",
            "epsilon_H1": "epsilon_H2",
            "epsilon_Cl1": "epsilon_Cl",
        }


        at_names = at_keys.copy()

        molec_map_dicts = {
            "EG": EG_map_dict,
            "Gly": Gly_map_dict,
            "ACN": ACN_map_dict,
            "MeOH": MeOH_map_dict,
            "DMF": DMF_map_dict,
            "DMSO": DMSO_map_dict,
            "THF": THF_map_dict,
            "DEC": DEC_map_dict,
            "DCM": DCM_map_dict,
        }

        super().__init__(at_bounds, at_names, molec_map_dicts)
        # Get scaled bounds
        self.scale_bounds()