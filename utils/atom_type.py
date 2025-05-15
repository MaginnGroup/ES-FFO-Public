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
    if at_number == 1:
        return AT_Scheme_01()
    elif at_number == 0:
        return AT_Scheme_00()
    elif at_number == 2:
        return AT_Scheme_02()
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

class AT_Scheme_00(Atom_Types):
    """
    Class for Atom Type Scheme 1 (C, F, H)

    Methods
    -------
    __init__(self)
    """

    def __init__(self):
        # Get Bounds
        at_param_bounds_l = [
            2,
            1.5,
            2,
            10,
            2,
            15,
        ]  # Units of Angstroms and Kelvin for Sigmas and Epsilons
        at_param_bounds_u = [4, 3, 4, 75, 15, 50]
        at_bounds = np.array([at_param_bounds_l, at_param_bounds_u]).T
        self.scheme_name = "at_00"
        self.scheme_plot_name = "AT-3"
        # Get Names
        at_keys = [
            "sigma_C",
            "sigma_H",
            "sigma_F",
            "epsilon_C",
            "epsilon_H",
            "epsilon_F",
        ]
        assert (
            len(at_keys) == len(at_param_bounds_l) == len(at_param_bounds_u)
        ), "Length of at_keys, at_param_bounds_l, and at_param_bounds_u must be the same"

        # Create a file that maps param names (keys) to at_param names for atom type 0 (values) for each molecule
        r14_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_F1": "epsilon_F",
        }

        r32_map_dict = {
            "sigma_C": "sigma_C",
            "sigma_H": "sigma_H",
            "sigma_F": "sigma_F",
            "epsilon_C": "epsilon_C",
            "epsilon_H": "epsilon_H",
            "epsilon_F": "epsilon_F",
        }

        r50_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_H1": "sigma_H",
            "epsilon_C1": "epsilon_C",
            "epsilon_H1": "epsilon_H",
        }

        r125_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_C2": "sigma_C",
            "sigma_H1": "sigma_H",
            "sigma_F1": "sigma_F",
            "sigma_F2": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_C2": "epsilon_C",
            "epsilon_H1": "epsilon_H",
            "epsilon_F1": "epsilon_F",
            "epsilon_F2": "epsilon_F",
        }

        r134a_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_C2": "sigma_C",
            "sigma_H1": "sigma_H",
            "sigma_F1": "sigma_F",
            "sigma_F2": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_C2": "epsilon_C",
            "epsilon_H1": "epsilon_H",
            "epsilon_F1": "epsilon_F",
            "epsilon_F2": "epsilon_F",
        }

        r143a_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_C2": "sigma_C",
            "sigma_H1": "sigma_H",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_C2": "epsilon_C",
            "epsilon_H1": "epsilon_H",
            "epsilon_F1": "epsilon_F",
        }

        r170_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_H1": "sigma_H",
            "epsilon_C1": "epsilon_C",
            "epsilon_H1": "epsilon_H",
        }

        # Test molecules
        r41_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_H1": "sigma_H",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_H1": "epsilon_H",
            "epsilon_F1": "epsilon_F",
        }

        r23_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_H1": "sigma_H",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_H1": "epsilon_H",
            "epsilon_F1": "epsilon_F",
        }

        r161_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_C2": "sigma_C",
            "sigma_H1": "sigma_H",
            "sigma_H2": "sigma_H",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_C2": "epsilon_C",
            "epsilon_H1": "epsilon_H",
            "epsilon_H2": "epsilon_H",
            "epsilon_F1": "epsilon_F",
        }

        r152a_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_C2": "sigma_C",
            "sigma_H1": "sigma_H",
            "sigma_H2": "sigma_H",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_C2": "epsilon_C",
            "epsilon_H1": "epsilon_H",
            "epsilon_H2": "epsilon_H",
            "epsilon_F1": "epsilon_F",
        }

        r152_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_H1": "sigma_H",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_H1": "epsilon_H",
            "epsilon_F1": "epsilon_F",
        }

        r143_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_C2": "sigma_C",
            "sigma_H1": "sigma_H",
            "sigma_H2": "sigma_H",
            "sigma_F1": "sigma_F",
            "sigma_F2": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_C2": "epsilon_C",
            "epsilon_H1": "epsilon_H",
            "epsilon_H2": "epsilon_H",
            "epsilon_F1": "epsilon_F",
            "epsilon_F2": "epsilon_F",
        }

        r134_map_dict = {
            "sigma_C": "sigma_C",
            "sigma_H": "sigma_H",
            "sigma_F": "sigma_F",
            "epsilon_C": "epsilon_C",
            "epsilon_H": "epsilon_H",
            "epsilon_F": "epsilon_F",
        }

        r116_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_F1": "epsilon_F",
        }

        at_names = at_keys.copy()

        molec_map_dicts = {
            "R14": r14_map_dict,
            "R32": r32_map_dict,
            "R50": r50_map_dict,
            "R125": r125_map_dict,
            "R134a": r134a_map_dict,
            "R143a": r143a_map_dict,
            "R170": r170_map_dict,
            "R41": r41_map_dict,
            "R23": r23_map_dict,
            "R161": r161_map_dict,
            "R152a": r152a_map_dict,
            "R152": r152_map_dict,
            "R143": r143_map_dict,
            "R134": r134_map_dict,
            "R116": r116_map_dict,
        }

        super().__init__(at_bounds, at_names, molec_map_dicts)
        # Get scaled bounds
        self.scale_bounds()
class AT_Scheme_01(Atom_Types):
    """
    Class for Atom Type Scheme 1 (C1, C2, F, H)

    Methods
    -------
    __init__(self)
    """

    def __init__(self):
        # Get Bounds
        at_param_bounds_l = [
            2,
            2,
            1.5,
            2,
            10,
            10,
            2,
            15,
        ]  # Units of Angstroms and Kelvin for Sigmas and Epsilons
        at_param_bounds_u = [4, 4, 3, 4, 75, 75, 15, 50]
        at_bounds = np.array([at_param_bounds_l, at_param_bounds_u]).T
        self.scheme_name = "at_01"
        self.scheme_plot_name = "AT-4"
        # Get Names
        at_keys = [
            "sigma_C1",
            "sigma_C2",
            "sigma_H",
            "sigma_F",
            "epsilon_C1",
            "epsilon_C2",
            "epsilon_H",
            "epsilon_F",
        ]
        assert (
            len(at_keys) == len(at_param_bounds_l) == len(at_param_bounds_u)
        ), "Length of at_keys, at_param_bounds_l, and at_param_bounds_u must be the same"
        # Get weight information
        # at_weights = np.zeros(len(at_keys))
        # gaff_params = np.array(at_param_bounds_l) #set initial gaff parameters as lower bounds
        # mask_names = list(["sigma_C2_2", "sigma_F_1", "epsilon_C1", "epsilon_C2_1", "epsilon_F_1"])
        # mask = np.isin(at_keys, mask_names)
        # GAFF_params = np.array([3.400, 3.118, 55.052, 55.052, 30.696])
        # #For all of these, a value of 0.15 difference from GAFF is weighted as 5% of the average best objective for ExpVal
        # g_weights = np.array([0.05/0.15**2, 0.05/0.15**2, 0.2/0.15**2, 0.05/0.15**2, 0.05/0.15**2])
        # at_weights[mask] = g_weights
        # gaff_params[mask] = GAFF_params
        # self.weighted_params = mask_names
        # self.at_weights = at_weights
        # self.gaff_params = gaff_params

        # Create a file that maps param names (keys) to at_param names for atom type 11 (values) for each molecule
        r14_map_dict = {
            "sigma_C1": "sigma_C1",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C1",
            "epsilon_F1": "epsilon_F",
        }

        r32_map_dict = {
            "sigma_C": "sigma_C1",
            "sigma_H": "sigma_H",
            "sigma_F": "sigma_F",
            "epsilon_C": "epsilon_C1",
            "epsilon_H": "epsilon_H",
            "epsilon_F": "epsilon_F",
        }

        r50_map_dict = {
            "sigma_C1": "sigma_C1",
            "sigma_H1": "sigma_H",
            "epsilon_C1": "epsilon_C1",
            "epsilon_H1": "epsilon_H",
        }

        r125_map_dict = {
            "sigma_C1": "sigma_C2",
            "sigma_C2": "sigma_C2",
            "sigma_H1": "sigma_H",
            "sigma_F1": "sigma_F",
            "sigma_F2": "sigma_F",
            "epsilon_C1": "epsilon_C2",
            "epsilon_C2": "epsilon_C2",
            "epsilon_H1": "epsilon_H",
            "epsilon_F1": "epsilon_F",
            "epsilon_F2": "epsilon_F",
        }

        r134a_map_dict = {
            "sigma_C1": "sigma_C2",
            "sigma_C2": "sigma_C2",
            "sigma_H1": "sigma_H",
            "sigma_F1": "sigma_F",
            "sigma_F2": "sigma_F",
            "epsilon_C1": "epsilon_C2",
            "epsilon_C2": "epsilon_C2",
            "epsilon_H1": "epsilon_H",
            "epsilon_F1": "epsilon_F",
            "epsilon_F2": "epsilon_F",
        }

        r143a_map_dict = {
            "sigma_C1": "sigma_C2",
            "sigma_C2": "sigma_C2",
            "sigma_H1": "sigma_H",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C2",
            "epsilon_C2": "epsilon_C2",
            "epsilon_H1": "epsilon_H",
            "epsilon_F1": "epsilon_F",
        }

        r170_map_dict = {
            "sigma_C1": "sigma_C2",
            "sigma_H1": "sigma_H",
            "epsilon_C1": "epsilon_C2",
            "epsilon_H1": "epsilon_H",
        }

        # Test molecules
        r41_map_dict = {
            "sigma_C1": "sigma_C1",
            "sigma_H1": "sigma_H",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C1",
            "epsilon_H1": "epsilon_H",
            "epsilon_F1": "epsilon_F",
        }

        r23_map_dict = {
            "sigma_C1": "sigma_C1",
            "sigma_H1": "sigma_H",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C1",
            "epsilon_H1": "epsilon_H",
            "epsilon_F1": "epsilon_F",
        }

        r161_map_dict = {
            "sigma_C1": "sigma_C2",
            "sigma_C2": "sigma_C2",
            "sigma_H1": "sigma_H",
            "sigma_H2": "sigma_H",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C2",
            "epsilon_C2": "epsilon_C2",
            "epsilon_H1": "epsilon_H",
            "epsilon_H2": "epsilon_H",
            "epsilon_F1": "epsilon_F",
        }

        r152a_map_dict = {
            "sigma_C1": "sigma_C2",
            "sigma_C2": "sigma_C2",
            "sigma_H1": "sigma_H",
            "sigma_H2": "sigma_H",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C2",
            "epsilon_C2": "epsilon_C2",
            "epsilon_H1": "epsilon_H",
            "epsilon_H2": "epsilon_H",
            "epsilon_F1": "epsilon_F",
        }

        r152_map_dict = {
            "sigma_C1": "sigma_C2",
            "sigma_H1": "sigma_H",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C2",
            "epsilon_H1": "epsilon_H",
            "epsilon_F1": "epsilon_F",
        }

        r143_map_dict = {
            "sigma_C1": "sigma_C2",
            "sigma_C2": "sigma_C2",
            "sigma_H1": "sigma_H",
            "sigma_H2": "sigma_H",
            "sigma_F1": "sigma_F",
            "sigma_F2": "sigma_F",
            "epsilon_C1": "epsilon_C2",
            "epsilon_C2": "epsilon_C2",
            "epsilon_H1": "epsilon_H",
            "epsilon_H2": "epsilon_H",
            "epsilon_F1": "epsilon_F",
            "epsilon_F2": "epsilon_F",
        }

        r134_map_dict = {
            "sigma_C": "sigma_C2",
            "sigma_H": "sigma_H",
            "sigma_F": "sigma_F",
            "epsilon_C": "epsilon_C2",
            "epsilon_H": "epsilon_H",
            "epsilon_F": "epsilon_F",
        }

        r116_map_dict = {
            "sigma_C1": "sigma_C2",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C2",
            "epsilon_F1": "epsilon_F",
        }

        at_names = at_keys.copy()

        molec_map_dicts = {
            "R14": r14_map_dict,
            "R32": r32_map_dict,
            "R50": r50_map_dict,
            "R125": r125_map_dict,
            "R134a": r134a_map_dict,
            "R143a": r143a_map_dict,
            "R170": r170_map_dict,
            "R41": r41_map_dict,
            "R23": r23_map_dict,
            "R161": r161_map_dict,
            "R152a": r152a_map_dict,
            "R152": r152_map_dict,
            "R143": r143_map_dict,
            "R134": r134_map_dict,
            "R116": r116_map_dict,
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
        at_param_bounds_l = [
            2,
            1.5,
            1.5,
            1.5,
            1.5,
            2,
            10,
            2,
            2,
            2,
            2,
            15,
        ]  # Units of Angstroms and Kelvin for Sigmas and Epsilons
        at_param_bounds_u = [4, 3, 3, 3, 3, 4, 75, 15, 15, 15, 15, 50]
        at_bounds = np.array([at_param_bounds_l, at_param_bounds_u]).T
        self.scheme_name = "at_02"
        self.scheme_plot_name = "AT-6a"
        # Get Names
        at_keys = [
            "sigma_C",
            "sigma_Hc",
            "sigma_H1",
            "sigma_H2",
            "sigma_H3",
            "sigma_F",
            "epsilon_C",
            "epsilon_Hc",
            "epsilon_H1",
            "epsilon_H2",
            "epsilon_H3",
            "epsilon_F",
        ]
        assert (
            len(at_keys) == len(at_param_bounds_l) == len(at_param_bounds_u)
        ), "Length of at_keys, at_param_bounds_l, and at_param_bounds_u must be the same"
        # Get weight information
        # at_weights = np.zeros(len(at_keys))
        # gaff_params = np.array(at_param_bounds_l) #set initial gaff parameters as lower bounds
        # mask_names = list(["sigma_C2_2", "sigma_F_1", "epsilon_C1", "epsilon_C2_1", "epsilon_F_1"])
        # mask = np.isin(at_keys, mask_names)
        # GAFF_params = np.array([3.400, 3.118, 55.052, 55.052, 30.696])
        # #For all of these, a value of 0.15 difference from GAFF is weighted as 5% of the average best objective for ExpVal
        # g_weights = np.array([0.05/0.15**2, 0.05/0.15**2, 0.2/0.15**2, 0.05/0.15**2, 0.05/0.15**2])
        # at_weights[mask] = g_weights
        # gaff_params[mask] = GAFF_params
        # self.weighted_params = mask_names
        # self.at_weights = at_weights
        # self.gaff_params = gaff_params

        # Create a file that maps param names (keys) to at_param names for atom type 11 (values) for each molecule
        r14_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_F1": "epsilon_F",
        }

        r32_map_dict = {
            "sigma_C": "sigma_C",
            "sigma_H": "sigma_H2",
            "sigma_F": "sigma_F",
            "epsilon_C": "epsilon_C",
            "epsilon_H": "epsilon_H2",
            "epsilon_F": "epsilon_F",
        }

        r50_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_H1": "sigma_Hc",
            "epsilon_C1": "epsilon_C",
            "epsilon_H1": "epsilon_Hc",
        }

        r125_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_C2": "sigma_C",
            "sigma_H1": "sigma_H2",
            "sigma_F1": "sigma_F",
            "sigma_F2": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_C2": "epsilon_C",
            "epsilon_H1": "epsilon_H2",
            "epsilon_F1": "epsilon_F",
            "epsilon_F2": "epsilon_F",
        }

        r134a_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_C2": "sigma_C",
            "sigma_H1": "sigma_H1",
            "sigma_F1": "sigma_F",
            "sigma_F2": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_C2": "epsilon_C",
            "epsilon_H1": "epsilon_H1",
            "epsilon_F1": "epsilon_F",
            "epsilon_F2": "epsilon_F",
        }

        r143a_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_C2": "sigma_C",
            "sigma_H1": "sigma_Hc",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_C2": "epsilon_C",
            "epsilon_H1": "epsilon_Hc",
            "epsilon_F1": "epsilon_F",
        }

        r170_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_H1": "sigma_Hc",
            "epsilon_C1": "epsilon_C",
            "epsilon_H1": "epsilon_Hc",
        }

        # Test molecules
        r41_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_H1": "sigma_H1",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_H1": "epsilon_H1",
            "epsilon_F1": "epsilon_F",
        }

        r23_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_H1": "sigma_H3",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_H1": "epsilon_H3",
            "epsilon_F1": "epsilon_F",
        }

        r161_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_C2": "sigma_C",
            "sigma_H2": "sigma_Hc",
            "sigma_H1": "sigma_H1",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_C2": "epsilon_C",
            "epsilon_H2": "epsilon_Hc",
            "epsilon_H1": "epsilon_H1",
            "epsilon_F1": "epsilon_F",
        }

        r152a_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_C2": "sigma_C",
            "sigma_H2": "sigma_Hc",
            "sigma_H1": "sigma_H2",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_C2": "epsilon_C",
            "epsilon_H2": "epsilon_Hc",
            "epsilon_H1": "epsilon_H2",
            "epsilon_F1": "epsilon_F",
        }

        r152_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_H1": "sigma_H1",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_H1": "epsilon_H1",
            "epsilon_F1": "epsilon_F",
        }

        r143_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_C2": "sigma_C",
            "sigma_H2": "sigma_H1",
            "sigma_H1": "sigma_H2",
            "sigma_F1": "sigma_F",
            "sigma_F2": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_C2": "epsilon_C",
            "epsilon_H2": "epsilon_H1",
            "epsilon_H1": "epsilon_H2",
            "epsilon_F1": "epsilon_F",
            "epsilon_F2": "epsilon_F",
        }

        r134_map_dict = {
            "sigma_C": "sigma_C",
            "sigma_H": "sigma_H2",
            "sigma_F": "sigma_F",
            "epsilon_C": "epsilon_C",
            "epsilon_H": "epsilon_H2",
            "epsilon_F": "epsilon_F",
        }

        r116_map_dict = {
            "sigma_C1": "sigma_C",
            "sigma_F1": "sigma_F",
            "epsilon_C1": "epsilon_C",
            "epsilon_F1": "epsilon_F",
        }

        at_names = at_keys.copy()

        molec_map_dicts = {
            "R14": r14_map_dict,
            "R32": r32_map_dict,
            "R50": r50_map_dict,
            "R125": r125_map_dict,
            "R134a": r134a_map_dict,
            "R143a": r143a_map_dict,
            "R170": r170_map_dict,
            "R41": r41_map_dict,
            "R23": r23_map_dict,
            "R161": r161_map_dict,
            "R152a": r152a_map_dict,
            "R152": r152_map_dict,
            "R143": r143_map_dict,
            "R134": r134_map_dict,
            "R116": r116_map_dict,
        }

        super().__init__(at_bounds, at_names, molec_map_dicts)
        # Get scaled bounds
        self.scale_bounds()
