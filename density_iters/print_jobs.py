import signac

project = signac.get_project("runs")
for job in project:
    if "aspect_ratio" not in job.sp:
        print(job)
        job.sp["aspect_ratio"] = 3.0
