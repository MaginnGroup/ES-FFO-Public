import signac
import glob
import os
import shutil

# Load the project
project = signac.get_project()

#Loop over all jobs in the project
for job in project.find_jobs({"mol_name":"MeOH"}):
    # Check if job document exists
    if os.path.exists(job.fn("signac_job_document.json")):
        if "eq_liq_dens" in job.document and job.doc["eq_liq_dens"] < 1.0:
            print(f"Job {job.id} has eq_liq_dens < 1.0: {job.doc['eq_liq_dens']}")
        # elif "npt_prod_fin" not in job.document and "ld_fail" not in job.document:
        #     print(f"Job {job.id}")
        #     #Print the last line of the run_npt_prod.out log file
        #     log_file = job.fn("run_npt_prod.out")
        #     if os.path.exists(log_file):
        #         with open(log_file, "r") as f:
        #             lines = f.readlines()
        #             if lines:
        #                 print(lines[-1].strip())
        #     else:
        #         print("Log file does not exist.")