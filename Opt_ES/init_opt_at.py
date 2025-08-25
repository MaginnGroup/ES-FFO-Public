import signac
import json

project = signac.init_project("opt_at_params")

# Set Initial Parameters
Atom_Types = [1,2]
repeats = 25  # Repeats for full optimization
repeats_ind = 25  # Repeats for individual molecule optimization
lhs_pts = int(1e5)  # Number of LHS points to generate
seed = 1
save_data = True
training_molecules = list(
    ["EG", "Gly", "MeOH", "DMSO", "DEC", "DMF"]
)
train_mol_genFF = list(["EG", "Gly", "MeOH"])

if isinstance(train_mol_genFF, list):
    gen_ff_train_mol = json.dumps(train_mol_genFF)

Objective = "ExpVal"

for Atom_Type in Atom_Types:
    #Optimize full generalized FF for EG, Gly, and MeOH
    for i in range(0, repeats):
        # Create job parameter dict
        sp = {
            "atom_type": Atom_Type,
            "total_repeats": repeats,
            "repeat_number": i + 1,
            "training_molecules": gen_ff_train_mol,
            "num_train_molec": len(train_mol_genFF),
            "obj_choice": Objective,
            "save_data": save_data,
            "new_weight": True,
            "seed": seed,
        }
        if i == 0:
            sp["lhs_pts"] = lhs_pts
        # Create jobs for exploration bias study
        job = project.open_job(sp).init()

    #For GAFF Atom types, optimize each molecule individually
    # if Atom_Type == 2:
    #     if len(training_molecules) > 1:
    #         for molec in training_molecules:
    #             #Check if vle iters are finished yet
    #             # Make a dumped list of the molecule to pass to the job
    #             molec_dump = json.dumps(list([molec]))
    #             for j in range(0, repeats_ind):
    #                 sp = {
    #                     "atom_type": Atom_Type,
    #                     "total_repeats": repeats_ind,
    #                     "repeat_number": j + 1,
    #                     "training_molecules": molec_dump,
    #                     "num_train_molec": 1,
    #                     "obj_choice": Objective,
    #                     "save_data": save_data,
    #                     "seed": seed,
    #                 }
    #                 if j == 0:
    #                     sp["lhs_pts"] = lhs_pts
    #                 job = project.open_job(sp).init()
