import numpy as np
from scipy.stats import qmc
import pandas as pd

def values_real_to_scaled(values, bounds):
    """Convert values in physical units to values scaled by bounds

    Parameters
    ----------
    values : array_like, shape=(n,m)
        Input values (unscaled)
    bounds : array_like, shape=(m,2)
        Bounds to scale `values`. Lower bound is 0 and upper bound
        is 1 in `scaled_values`.

    Returns
    -------
    scaled_values : np.ndarray, shape=(n,m)
        The values scaled by `bounds`

    Notes
    -----
    The `bounds` define the 0 and 1 limits of the `scaled_values`.
    The `values` may exceed the bounds; in this case the
    `scaled_values` will have values < 0 or > 1.
    """
    values, bounds = _clean_bounds_values(values, bounds)

    # Check where bounds are equal
    denom = bounds[:, 1] - bounds[:, 0]
    equal_mask = denom == 0

    #Normalize data in between bounds based on this info
    normalized = np.empty_like(values, dtype=float)
    normalized[:, equal_mask] = 0 #Fix all values where bounds are equal as zero (the lower bound)
    normalized[:, ~equal_mask] = (values[:,~equal_mask] - bounds[~equal_mask, 0]) / denom[~equal_mask]

    # (values - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

    return normalized


def values_scaled_to_real(scaled_values, bounds):
    """Convert scaled values to values in physical units

    Parameters
    ----------
    scaled_values : array_like, shape=(n,m)
        Input values (scaled)
    bounds : array_like, shape=(m,2)
        Bounds to scale `values`. Lower bound is 0 and upper bound
        is 1 in `scaled_values`.

    Returns
    -------
    values : np.ndarray, shape=(n,m)
        The values in unscaled units

    Notes
    -----
    The `bounds` define the 0 and 1 limits of the `scaled_values`.
    The `scaled_values` may exceed the 0 and 1; in this case the
    `values` will have values < lower bound or > upper bound.
    """
    scaled_values, bounds = _clean_bounds_values(scaled_values, bounds)
    return scaled_values * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


def variances_scaled_to_real(scaled_variances, bounds):
    """Convert variance in scaled dimensionless values to physical units

    Parameters
    ----------
    scaled_variances : array_like, shape=(n,m)
        Input variances (scaled)
    bounds : array_like, shape=(m,2)
        Bounds to scale `scaled_variances`. Lower bound is 0 and upper bound
        is 1 in `scaled_values`.

    Returns
    -------
    real_vars : np.ndarray, shape=(n,m)
        The variance values in unscaled units
    """
    scaled_variances, bounds = _clean_bounds_values(scaled_variances, bounds)

    if (scaled_variances < 0.0).any():
        raise ValueError("Variance cannot be less than zero")

    return scaled_variances * (bounds[:, 1] - bounds[:, 0]) ** 2

def variances_real_to_scaled(real_variances, bounds):
    """Convert variance in scaled dimensionless values to physical units

    Parameters
    ----------
    real_variances : array_like, shape=(n,m)
        Input variances (unscaled)
    bounds : array_like, shape=(m,2)
        Bounds to scale `real_variances`. Lower bound is 0 and upper bound
        is 1 in `real_values`.

    Returns
    -------
    real_vars : np.ndarray, shape=(n,m)
        The variance values in unscaled units
    """
    real_variances, bounds = _clean_bounds_values(real_variances, bounds)

    if (real_variances < 0.0).any():
        raise ValueError("Variance cannot be less than zero")

    return real_variances / (bounds[:, 1] - bounds[:, 0]) ** 2


def _clean_bounds_values(values, bounds):
    values = np.asarray(values)
    bounds = np.asarray(bounds)
    bounds = bounds.reshape(-1, 2)

    if not (bounds[:, 0] <= bounds[:, 1]).all():
        raise ValueError(
            "Lower bound must always be less than or equal to the upper bound."
        )

    if bounds.shape[0] == 1:
        values = values.reshape(-1, 1)
    else:
        if len(values.shape) != 2 or values.shape[1] != bounds.shape[0]:
            raise ValueError(
                "Shapes of `values` and `bounds` must be consistent. "
                "Please see the doc strings for more information."
            )

    return values, bounds


def shuffle_and_split(df, param_names, property_name, fraction_train=0.8, shuffle_seed=None):
    """Randomly shuffle the DataFrame and extracts the train and test sets

    Parameters
    ----------
    df : pandas.DataFrame
        The pandas dataframe with the samples
    param_names : list-like
        names of the parameters to extract from the dataframe (x_data)
    property_name : string
        Name of the property to extract from the dataframe (y_data)
    fraction_train : float, optional, default = 0.8
        Fraction of sample to use as training data. Remainder is test data.
    shuffle_seed : int, optional, default = None
        seed for random number generator for shuffle

    Returns
    -------
    x_train : np.ndarray
        Training inputs
    y_train : np.ndarray
        Training results
    x_test : np.ndarray
        Testing inputs
    y_test : np.ndarray
        Testing results
    """
    if fraction_train < 0.0 or fraction_train > 1.0:
        raise ValueError("`fraction_train` must be between 0 and 1.")
    else:
        fraction_test = 1.0 - fraction_train

    try:
        prp_idx = df.columns.get_loc(property_name)
    except KeyError:
        raise ValueError(
            "`property_name` does not match any headers of `df`"
        )
    if type(param_names) not in (list, tuple):
        raise TypeError("`param_names` must be a list or tuple")
    else:
        param_names = list(param_names)

    data = df[param_names + [property_name]].values
    # if property_name == "sim_surf_tens":
    #     # print(np.max(data[:, -1]), np.min(data[:, -1]))
    #     #Plot histogram of data
    #     import matplotlib.pyplot as plt

    #     #Remove rows where the property value is > 20 or < -10
    #     # data = data[data[:, -1] <= 40]
    #     data = data[(data[:, -1] <= 10) & (data[:, -1] >= -10)]
    #     plt.hist(data[:, -1], bins=30)
    #     plt.xlabel("Surface Tension [mN/m]")
    #     plt.ylabel("Frequency")
    #     plt.title("Histogram of Surface Tension Data")
    #     plt.savefig("surface_tension_histogram.png")

    total_entries = data.shape[0]

    # Ensure at least one training entry
    if total_entries == 1:
        x_train = np.atleast_2d(data[0, :-1].astype(np.float64))
        y_train = np.atleast_1d(data[0, -1].astype(np.float64))
        x_test = np.empty((0, x_train.shape[0]), dtype=np.float64)
        y_test = np.empty((0,), dtype=np.float64)
    else:
        train_entries = max(int(total_entries * fraction_train), 1)

        # Shuffle the data before splitting train/test sets
        if shuffle_seed is not None:
            np.random.seed(shuffle_seed)
        np.random.shuffle(data)

        x_train = data[:train_entries, :-1].astype(np.float64)
        y_train = data[:train_entries, -1].astype(np.float64)
        x_test = data[train_entries:, :-1].astype(np.float64)
        y_test = data[train_entries:, -1].astype(np.float64)

    return x_train, y_train, x_test, y_test

def generate_lhs(samples, bounds, seed, labels = None):
    assert bounds.shape[1] == 2, "Bounds must be a 2D array"
    assert isinstance(samples, int), "Number of samples must be an integer"
    #Define number of dimensions
    dimensions = bounds.shape[0]
    # #Define sampler
    sampler = qmc.LatinHypercube(d=dimensions, seed = seed)
    lhs_data = sampler.random(n=samples)

    # #Generate LHS data given bounds
    # lhs_data = qmc.scale(lhs_data, bounds[:,0], bounds[:,1])
    # sample = pd.DataFrame(lhs_data)

    # Scale each dimension manually
    scaled = np.zeros_like(lhs_data)
    for i in range(dimensions):
        low, high = bounds[i]
        if np.isclose(low, high):  # fixed dimension
            scaled[:, i] = low     # constant value
        else:
            scaled[:, i] = qmc.scale(np.array([lhs_data[:, i]]), low, high)

    sample = pd.DataFrame(scaled)

    if labels is not None:
        assert len(labels) == bounds.shape[0], "Number of labels must match number of bounds"
        sample.columns = labels

    return lhs_data