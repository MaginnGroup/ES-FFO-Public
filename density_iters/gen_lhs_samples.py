from scipy.stats import qmc
import pandas as pd
import csv

d = 4  # Number of dimensions
n = 200
sampler = qmc.LatinHypercube(d)
sample = sampler.random(n)
sample = pd.DataFrame(sample)
sample.columns = [
    "sigma_C1",
    "sigma_F1",
    "epsilon_C1",
    "epsilon_F1",
]  # change to sigma and epsilon name of different atom types
# sample.set_index('sigma_C1')

filename = "LHS_" + str(n) + "_x_" + str(d) + ".csv"
sample.to_csv(filename, index=True)

# class_dict = _get_class_from_molecule(job.sp.mol_name)
#         class_data = class_dict[job.sp.mol_name]
#         d = class_data.num_params  # Number of dimensions
#         seed = 7
#         n = 200
#         sampler = qmc.LatinHypercube(d, seed=seed)
#         lh_samples = sampler.random(n)
#         bounds = class_data.param_bounds

#         # Save the samples to a csv file
#         sample = pd.DataFrame(lh_samples)
#         sample.columns = class_data.param_names
#         filename = "LHS_" + str(n) + "_x_" + str(d) + ".csv"
#         sample.to_csv(filename, index=True)
