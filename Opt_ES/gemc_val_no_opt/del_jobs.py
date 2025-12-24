import signac
import glob
import os
import shutil

# Load the project
project = signac.get_project()
count = 0
#Loop over all jobs in the project
for job in project.find_jobs({"mol_name": "DEC"}):
    # Move old results to a subfolder
    # count += 1
    # Check if job document exists
    # print(f"Checking job {job.id}")
    if os.path.exists(job.fn("signac_job_document.json")):
        # count_running += 1
        if "nvt_fin" not in job.doc.keys():
            count += 1
            print(f"Job {job.id} mol_name {job.sp.mol_name} T {job.sp.T} restart {job.sp.restart} has gemc_failed in doc.")
            job.remove()
    # with job:
    #     if job.sp.atom_type > 0:
    #         job.remove()
print(count)
