import signac

project = signac.get_project("runs")
for job in project:
    print(job.id)
