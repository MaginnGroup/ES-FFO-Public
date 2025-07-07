import signac
import glob
import os
import shutil

# Load the project
project = signac.get_project()

#Loop over all jobs in the project
for job in project.find_jobs():
    # Move old results to a subfolder
    with job:
        if os.path.exists(job.fn("signac_job_document.json")):
            subfolder = "old4"
            os.makedirs(subfolder, exist_ok=True)

            # Files to preserve (base names or extensions)
            preserve_files = {
                "signac_statepoint.json",
                "signac_job_document.json",
                "ff.xml",
            }
            preserve_extensions = {".gro", ".top"}

            for item in os.listdir("."):
                # Skip preserved files
                if item in preserve_files:
                    continue
                # Skip .gro or .top files
                if os.path.splitext(item)[1] in preserve_extensions:
                    continue
                # Skip anything starting with "old"
                if item.startswith("old"):
                    continue

                dest = os.path.join(subfolder, item)
                if os.path.isdir(item):
                    shutil.move(item, dest)
                elif os.path.isfile(item):
                    shutil.move(item, dest)

            # Ensure signac_job_document.json is preserved in job root
            if os.path.exists("signac_job_document.json"):
                shutil.copy("signac_job_document.json", os.path.join(subfolder, "signac_job_document.json"))

            # Keep only 'system' key in job.doc
            keys_to_keep = {"system"}
            for key in list(job.doc.keys()):
                if key not in keys_to_keep:
                    del job.doc[key]

            #Remove subfolder
            # if os.path.exists(subfolder):
            #     shutil.rmtree(subfolder)
