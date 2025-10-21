import signac
import glob
import os
import shutil

# Load the project
project = signac.get_project()

count = 0
#Loop over all jobs in the project
for job in project.find_jobs({"mol_name":"DEC", "iter": 2}):
    # Check if job document exists
    # print(f"Checking job {job.id}")
    if os.path.exists(job.fn("signac_job_document.json")):
        # if "eq_liq_dens" in job.document and job.doc["eq_liq_dens"] < 1.0:
        #     count +=1
        #     print(f"Job {job.id} has eq_liq_dens < 1.0: {job.doc['eq_liq_dens']}")
        if "npt_prod_fin" not in job.document and "ld_fail" not in job.document:
            count += 1
            # print(f"Job {job.id}")
            #Print the last line of the run_npt_prod.out log file
            log_file = job.fn("run_npt_prod.out")
            if os.path.exists(log_file):
                # print(f"Job {job.id}")
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    if lines:
                        if "Mon" in lines[-1].strip():
                            print(f"Job {job.id}")
                            print(lines[-1].strip())
                            # count += 1
            else:
                print("Log file does not exist.")
print(f"Total unfinished jobs: {count}")