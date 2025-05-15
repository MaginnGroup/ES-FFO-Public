# Import dependencies
from flow import FlowProject, directives
import signac
import templates.ndcrc
import warnings
from pathlib import Path
import os
import json
import sys

sys.path.append("..")
from utils.molec_class_files import (
    r14,
    r32,
    r50,
    r125,
    r134a,
    r143a,
    r170,
    r41,
    r23,
    r161,
    r152a,
    r152,
    r134,
    r143,
    r116,
)
from utils import atom_type, opt_atom_types

sys.path.remove("..")

# Ignore warnings caused by "nan" values
import warnings
from warnings import simplefilter

warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
# Ignore warning from scikit learn hp tuning
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)


class ProjectOPT(FlowProject):
    def __init__(self):
        current_path = Path(os.getcwd()).absolute()
        # Set Project Path to be that of the current working directory
        super().__init__(path=current_path)


@ProjectOPT.label
def pareto_set_exists(job):
    # # Define driver class
    # training_molecules = job.sp.training_molecules
    # training_molecules = list(json.loads(training_molecules))
    # driver = opt_atom_types.Problem_Setup(
    #     training_molecules, job.sp.atom_type, job.sp.obj_choice
    # )
    # pareto_save = driver.use_dir_name / "pareto_info.csv"
    # # Check if pareto info exists
    # return pareto_save.exists()
    #For jobs where the repeat number is not 1, completion of repeat 1 will set this value to True
    if "pareto_info" not in job.doc:
        return False
    else:
        return job.doc["pareto_info"]

# Only run this operation for the first repeat job
@ProjectOPT.pre(
    lambda job: job.sp.repeat_number == 1 and job.sp.atom_type in [0, 1, 2, 3, 4, 5, 6, 8, 11]
)
@ProjectOPT.post(pareto_set_exists)
@ProjectOPT.operation()
def gen_pareto_sets(job):
    # Define method, ep_enum classes, indecies to consider, and kernel
    training_molecules = job.sp.training_molecules
    training_molecules = list(json.loads(training_molecules))
    # Get all gp data and make driver class
    driver = opt_atom_types.Problem_Setup(
        training_molecules, job.sp.atom_type, job.sp.obj_choice
    )
    # For the 1st repeat for any job, we will generate the pareto sets
    # Genrate and pareto set info
    pareto_info = driver.gen_pareto_sets(
        job.sp.lhs_pts, driver.at_class.at_bounds_nm_kjmol, save_data=True
    )

    #Load signac project that the job belongs to
    project = signac.get_project()

    #Find all jobs with the same atom type, obj_choice, and training molecules
    #dict(atom_type= job.sp.atom_type, obj_choice= job.sp.obj_choice, training_molecules=job.sp.training_molecules)
    for other_job in project.find_jobs({"atom_type" : job.sp.atom_type, "obj_choice" : job.sp.obj_choice, "training_molecules":job.sp.training_molecules}):
        # and set their pareto_info to True
        other_job.doc["pareto_info"] = True

@ProjectOPT.label
def results_computed(job):
    # Write script that checks whether the intermediate job files are there
    return job.isfile("opt_at_results.csv") and job.isfile("sorted_at_res.csv")


@ProjectOPT.pre(pareto_set_exists)
@ProjectOPT.post(results_computed)
@ProjectOPT.operation()
def run_obj_alg(job):
    # Define method, ep_enum classes, indecies to consider, and kernel
    training_molecules = job.sp.training_molecules
    training_molecules = list(json.loads(training_molecules))

    # Set params for saving results, # of repeats, and the seed
    obj_choice = job.sp.obj_choice
    total_repeats = job.sp.total_repeats
    repeat_num = job.sp.repeat_number
    seed = job.sp.seed
    save_data = job.sp.save_data

    # Get all gp data and make driver class
    driver = opt_atom_types.Opt_ATs(
        training_molecules, job.sp.atom_type, total_repeats, seed, obj_choice
    )

    if job.sp.obj_choice == "ExpValPrior":
        # Save weight scaler to job document
        job.doc["weight_sclr"] = driver.weight_sclr
        # Add weights to job document
        job.doc.weights = driver.at_class.at_weights
        job.doc.wt_params = driver.at_class.weighted_params

    # Create param sets for the AT optimization based on seed.
    # Save these to the Results folder in the directory above for reuse
    param_inits = driver.get_param_inits()

    # Get data if the repeat number is less than or equal to the number of param_inits
    if repeat_num <= len(param_inits):
        # Get the correct parameter set from the param_inits based on which repeat we are evaluating
        param_guess = param_inits[repeat_num - 1].reshape(1, -1)
        # Optimize atom types
        ls_results, sort_ls_res, best_runs = driver.optimize_ats(
            param_guess, repeat_num - 1
        )

        dir_name = driver.use_dir_name
        job.document.dir_name = str(dir_name)

        # Store intermediate results in job directory
        # Save original results
        save_path3 = job.fn("best_run.csv")
        best_runs.to_csv(save_path3, index=False)
        save_path1 = job.fn("opt_at_results.csv")
        ls_results.to_csv(save_path1, index=False)
        # Save sorted results
        save_path2 = job.fn("sorted_at_res.csv")
        sort_ls_res.to_csv(save_path2, index=False)
    # If the repeat number is greater than the number of param_inits, then we have already done all the repeats. Delete the job
    else:
        print("Repeat number is greater than the number of param_inits. Deleting job.")
        job.remove()


if __name__ == "__main__":
    ProjectOPT().main()
