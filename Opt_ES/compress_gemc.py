#Get all jobs in a project
import signac
import glob
import os
import shutil
import tarfile
import subprocess

mode = "decompress" #compress or decompress or check

# Load the project
project = signac.get_project("gemc_val_opt")
count = 0
#Loop over all jobs in the project
for job in project.find_jobs():
# for job in project.find_jobs({"mol_name": "EG", "T":378.15, "restart": 2}):
    if count == 0:
        print(job.id)
        count += 1
    #If gemc failed or vap_density and liq_density in job.doc
    job_init_doc = job.fn("signac_job_document.json")
    # Check if the file exists
    if os.path.exists(job_init_doc):
        cond1 =  job.doc.get("gemc_failed", False) == True
        cond2 = "vap_density" in job.doc and "liq_density" in job.doc

        #If the job failed or finished, archive it
        if cond1 or cond2:
            #Make a new directory in job.fn() 
            subfolder= "archive_res"
            with job:
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder)
                
                glob_args = ["*.H", "*.log", "*.inp", "*.out.xyz", "*.out.box*.xyz", "*.chk", "*.prp"]
                #Get list of files not to move
                keep_files = ["gemc.eq.out.box1.prp", "npt.eq.out.prp", "nvt.eq.out.prp"]

                try:
                    last_prod = sorted(glob.glob("prod.*out.box1.prp"))[-1]
                    keep_files.append(last_prod)
                except:
                    pass
                try:
                    last_eq = sorted(glob.glob("gemc.eq.*.out.box1.prp"))[-1]
                    keep_files.append(last_eq)
                except:
                    pass
                try:
                    last_eq_chk = sorted(glob.glob("gemc.eq*out.chk"))[-1]
                    keep_files.append(last_eq_chk)
                except:
                    pass
                

                tar_name = f"{subfolder}.tar.gz"


                if mode == "compress" or mode == "check":
                    for glob_arg in glob_args:
                        for file_path in glob.glob(glob_arg):
                            if file_path not in keep_files and "results_" not in os.path.dirname(file_path):
                                shutil.move(
                                file_path, os.path.join(subfolder, os.path.basename(file_path)))

                    #Move old results folders
                    for folder_path in glob.glob("results_*"):
                        if os.path.isdir(folder_path):
                            shutil.move(folder_path, os.path.join(subfolder, os.path.basename(folder_path)))

                    #Compress in place
                    #tar streams data → pigz compresses using multiple cores
                    cmd = f"tar -cf - {subfolder} | pigz > {tar_name}"
                    if not os.path.exists(tar_name):
                        subprocess.run(cmd, shell=True, check=True)
                    shutil.rmtree("archive_res")

                if mode == "decompress" or mode == "check":
                    #To uncompress and delete tar file
                    cmd2 = f"pigz -dc {tar_name} | tar -xf - --strip-components=1"
                    if os.path.exists(tar_name):
                        subprocess.run(cmd2, shell=True, check=True)
                        os.remove(tar_name)
                    try:
                        shutil.rmtree("archive_res")
                    except:
                        pass
        
