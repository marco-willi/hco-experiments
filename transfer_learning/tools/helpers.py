# helper functions

def second_to_str(seconds):
    day = seconds // (24 * 3600)
    time = seconds % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time
    return ("Days:%d Hours:%d Mins:%d Secs:%d" % (day, hour, minutes, seconds))
