#Get all jobs in a project
import signac
import glob
import os
import shutil
import tarfile
import subprocess
from pathlib import Path
import shlex
import fnmatch

mode = "check" #compress or decompress or check

# Load the project
project = signac.get_project("ift_val_no_opt")
#Loop over all jobs in the project
count = 0
for job in project.find_jobs():
# for job in project.find_jobs({"mol_name": "Gly", "T":313.15, "param_set": 1, "restart": 1}):
    if count == 0:
        print(job.id)
        count += 1
    #If gemc failed or vap_density and liq_density in job.doc
    job_init_doc = job.fn("signac_job_document.json")
    # Check if the file exists
    if os.path.exists(job_init_doc):
        cond1 =  job.doc.get("ld_fail", False) == True
        cond2 = "surf_tens" in job.doc and "surf_tens_unc" in job.doc and "inter_prod_fin" in job.doc
        cond3 = "eq_liq_dens" in job.doc and "liq_density" in job.doc and "liq_density_unc" in job.doc

        #If the job failed or finished, archive it
        if cond1 or cond2 or cond3:
            #Make a new directory in job.fn() 
            subfolder= "archive_res"
            with job:
                source_dir = job.fn("")
                
                tar_name = f"{subfolder}.tar.gz"
                tar_filename = os.path.basename(tar_name)
                safe_tar_name = shlex.quote(tar_name)

                keep_files = ["em/em.log", "nvt_eq/nvt_eq.log", "init_inter_eq/init_inter_eq.gro", "ff.xml", "system.gro", 
                              "unedited.top", "system.top", "*.json", "*.mdp",
                              "calc_props", "*.png", tar_name]

                #For check, decompress and then compress since all files have since been compressed
                if mode == "decompress" or mode == "check":
                    #To uncompress and delete tar file
                    cmd2 = f"pigz -dc {tar_name} | tar -xf - -C {source_dir}"
                    if os.path.exists(tar_name):
                        subprocess.run(cmd2, shell=True, check=True)
                        os.remove(tar_name)
                        # print(f"Decompressed and removed {tar_name} for {job.id}")

                if mode == "compress" or mode == "check":
                    #Compress in place
                    #tar streams data → pigz compresses using multiple cores
                    tar_cmd = ["tar", "-cf", "-"]
                    #append files to keep
                    for item in keep_files:
                        tar_cmd.append(f"--exclude={item}")
                    #Use source dir to save archive
                    tar_cmd.extend(["-C", source_dir, "."])
                    #Pipe
                    pigz_cmd = f"pigz > {safe_tar_name}"
                    #Full command
                    full_command = f"{' '.join([shlex.quote(arg) for arg in tar_cmd])} | {pigz_cmd}"
                    #Make tar file if it doesn't exist
                    if not os.path.exists(tar_name):
                        subprocess.run(full_command, shell=True, check=True)
                        #Then delete the files that were tarred
                        for root, dirs, files in os.walk(source_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                rel_path = os.path.relpath(file_path, source_dir)
                                # --- Logic to keep files ---
                                is_archive = (file == tar_filename)
                                # Check matches against keep_files patterns
                                is_matched = any(fnmatch.fnmatch(file, p) or fnmatch.fnmatch(rel_path, p) for p in keep_files)
                                # Specifically protect anything inside calc_props/
                                is_in_calc_props = rel_path.startswith("calc_props" + os.sep) or rel_path == "calc_props"
                                if not (is_archive or is_matched or is_in_calc_props):
                                    try:
                                        os.remove(file_path)
                                    except:
                                        pass
                    # print(f"Archived {job.id}")

                
