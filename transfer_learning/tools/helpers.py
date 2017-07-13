# helper functions
import os

def second_to_str(seconds):
    day = seconds // (24 * 3600)
    time = seconds % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time
    return ("Days:%d Hours:%d Mins:%d Secs:%d" % (day, hour, minutes, seconds))


def get_most_rescent_file_with_string(dirpath, in_str='', excl_str='!'):
    """ get most recent file from directory, that includes string """
    a = [s for s in os.listdir(dirpath)
         if os.path.isfile(os.path.join(dirpath, s))]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
    b = [x for x in a if (in_str in x) and not (excl_str in x)]
    latest = b[-1]
    return dirpath + '/' + latest


# get_most_rescent_file_with_string('D:\\Studium_GD\\Zooniverse\\Data\\' +
#                                   'transfer_learning_project\\models\\3663',
#                                   in_str='mnist_testing')
