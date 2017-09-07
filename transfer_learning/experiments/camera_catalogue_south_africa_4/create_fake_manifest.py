""" Code to import CamCat Manifest & create fake manifest for testing """
import csv
import random

# Parameters
path = 'D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\project_data\\camera_catalogue\\'
file_name_in = 'manifest_set_28_2017.09.07.csv'
file_name_out = 'manifest_set_28_2017.09.07_FAKE.csv'

# randomly sample data from this
classes_parent = ["blank", "vehicle", "species"]
classes = [
    "HUMAN", "aardvark", "aardwolf", "africancivet",
    "bat", "batEaredFox", "bird", "buffalo", "bushbaby", "bushbuck", "bushpig",
    "caracal", "cheetah", "domesticanimal", "duikersteenbok", "eland",
    "elephant",
    "gemsbock", "genet", "giraffe", "hartebeest", "hippopotamus",
    "honeyBadger", "hyaenabrown", "hyaenaspotted", "impala", "insect",
    "jackal", "klipspringer", "kudu", "leopard", "lion", "mongoose",
    "monkeybaboon", "nyala", "porcupine", "rabbithare", "reedbuck", "rhino",
    "roansable", "rodent", "serval", "warthog",  "waterbuck", "wildcat",
    "wilddog", "wildebeest", "zebra"]

non_experiment_classes = ["insect", "bushbaby", "bat", "serval", "wildcat"]
experiment_flag = [0, 1]

random.seed(12)
random.choice(experiment_flag)
# open file
file_in = open(path + file_name_in, "r")
file_out = open(path + file_name_out, "w", newline='')
file_in_reader = csv.reader(file_in)
file_out_writer = csv.writer(file_out)
counter = 0
write_lines = 10000
for row in file_in_reader:
    # modify header
    if counter == 0:
        row = row + ['#experiment_flag', '#machine_probability',
                     '#machine_prediction']
    else:
        # choose class
        random.seed(counter)
        cl_p = random.choice(classes_parent)
        if cl_p == "species":
            random.seed(counter)
            cl = random.choice(classes)
        else:
            cl = cl_p
        # choose experiment
        if cl in non_experiment_classes:
            exp = 0
        else:
            random.seed(counter)
            exp = random.choice(experiment_flag)
        # probability
        prop = random.uniform(0, 1)
        row = row + [exp, prop, cl]

    file_out_writer.writerow(row)
    counter += 1
    if counter > write_lines:
        break
file_in.close()
file_out.close()
