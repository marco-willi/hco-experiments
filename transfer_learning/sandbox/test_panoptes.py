# display all projects
for project in Project.where():
    print(project.title)


# get all projects
projects = list()
for project in Project.where():
    projects.append(project)

# extract all project titles
project_titles = list()
for project in projects:
    project_titles.append(project.title)