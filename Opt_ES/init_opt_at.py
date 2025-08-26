import signac
import json
import os

project = signac.init_project("opt_at_params")

# Set Initial Parameters
Atom_Types = [0,1,2]
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
    #Only make generalized FFs for ATs 1 and 2
    if Atom_Type > 0:
        #Check if ift iters are finished for all molecs in generalized FF
        all_finished = []
        for molecs in train_mol_genFF:
            all_finished.append(os.path.exists(f"../Build_GPs/analysis/{molecs}/vle_iters/iter-1/final-params.csv"))
        #If so, create jobs for generalized FF for EG, Gly, and MeOH
        if all(all_finished):
            #Number of restarts
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
                    "seed": seed,
                }
                if i == 0:
                    sp["lhs_pts"] = lhs_pts
                # Create jobs for exploration bias study
                job = project.open_job(sp).init()

    #For AT Scheme 0 (Distinct atom types, optimize each molecule individually)
    elif Atom_Type == 0:
        if len(training_molecules) > 1:
            for molec in training_molecules:
                #Check if vle iters are finished yet
                vle_iters_path = f"../Build_GPs/analysis/{molec}/vle_iters/iter-1/final-params.csv"
                if os.path.exists(vle_iters_path):
                    # Make a dumped list of the molecule to pass to the job
                    molec_dump = json.dumps(list([molec]))
                    for j in range(0, repeats_ind):
                        sp = {
                            "atom_type": Atom_Type,
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
