import signac
import json

project = signac.init_project("opt_at_params")

# Set Initial Parameters
repeats = 25  # Repeats for full optimization
repeats_ind = 25  # Repeats for individual molecule optimization
lhs_pts = int(1e5)  # Number of LHS points to generate
seed = 1
save_data = True
training_molecules = list(
    ["EG", "Gly", "MeOH", "DMSO", "DEC", "DMF"]
)

Objective = "ExpVal"

#For Distinct Atom types, optimize each molecule individually
if len(training_molecules) > 1:
    for molec in training_molecules:
        #Check if vle iters are finished yet
        # Make a dumped list of the molecule to pass to the job
        molec_dump = json.dumps(list([molec]))
        for j in range(0, repeats_ind):
            sp = {
                "total_repeats": repeats_ind,
                "repeat_number": j + 1,
                "training_molecules": molec_dump,
                "num_train_molec": 1,
                "obj_choice": Objective,
                "save_data": save_data,
                "seed": seed,
            }
            if j == 0:
                sp["lhs_pts"] = lhs_pts
            job = project.open_job(sp).init()
