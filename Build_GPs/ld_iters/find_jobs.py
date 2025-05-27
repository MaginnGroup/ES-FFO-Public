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
        if "nvt_eq_fin" not in job.document:
            print(f"Job {job.id} has nvt_eq_fin set to False in document.")