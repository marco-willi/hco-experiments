# helper functions
import os
from datetime import datetime


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


def createSplitIDs(ids_orig, labels_orig, meta_data=None, split_mode="1_on_1"):
    """ Creates splitting ids to be used for test/train splitting
        Uses meta_data dictionary with ids_orig as key
    """

    # Return 1 to 1 mapping
    if split_mode == "1_on_1":
        split_ids = ids_orig
        split_labels = labels_orig

    # return a mapping based on splitting along locations
    # and interval between subsequent images
    elif split_mode == "location_date_time":
        # loop through all subjects and get relevant attributes
        locations = list()
        dates = list()
        times = list()
        # create some dummy data for fields that are missing / have Default
        # values
        dummy_secs = list()
        dummy_loc_id = 0
        dummy_sec_id = 0
        # loop over all ids
        for ii in ids_orig:
            meta = meta_data[ii]
            # get attrs and store in list
            for tag, ll in zip(['location', 'date', 'time'],
                               [locations, dates, times]):
                # handle different types of meta data
                if tag in meta:
                    # if location is unkown create a dummy location id
                    if (tag == 'location') & (meta[tag] == 'unknown'):
                        ll.append('unknown' + str(dummy_loc_id))
                        dummy_loc_id += 1
                    # if date is unkown add a dummy second
                    elif (tag == 'date') & (meta[tag] in
                                            ["20010101", "unknown"]):
                        ll.append(meta[tag])
                        dummy_secs.append(dummy_sec_id)
                        dummy_sec_id += 1
                    # append time as is
                    else:
                        ll.append(meta[tag])
                        dummy_secs.append(0)
                else:
                    ll.append('unknown')

        # create date time seconds
        seconds = list()
        for dat, tm, ds in zip(dates, times, dummy_secs):
            try:
                dtm = datetime.strptime(dat + tm, '%Y%m%d%H%M%S')
                secs = dtm.timestamp()
            except:
                secs = 0
            # add dummy seconds to distinguish unknown times / dates to ensure
            # they don't all end up with the same splitting id
            seconds.append(secs+ds)

        # divide data into different locations
        loc_dat = dict()
        for loc, dd, tt, ss, ii, lab in zip(locations, dates, times,
                                            seconds,
                                            ids_orig, labels_orig):
            # prepare location dictionary
            if loc not in loc_dat:
                current_loc = {'ids': list(),
                               'labels': list(),
                               'dates': list(),
                               'times': list(),
                               'seconds': list()}
                loc_dat[loc] = current_loc

            # add all information
            for tag, ll in zip(['ids', 'labels', 'dates',
                                'times', 'seconds'],
                               [ii, lab, dd, tt, ss]):
                loc_dat[loc][tag].append(ll)

        # now we have a dictionary entry for each location with all its
        # ids, labels, dates and times, now create splitting
        # ids for each location
        for k, v in loc_dat.items():
            loc_labels = v['labels']
            loc_seconds = v['seconds']

            # define temporal ordering of capture events
            time_order_ids = sorted(range(len(loc_seconds)),
                                    key=lambda x: loc_seconds[x])

            # reorder all attributes according to time ordering
            loc_labels_order = [loc_labels[i] for i in time_order_ids]
            loc_seconds_order = [loc_seconds[i] for i in time_order_ids]

            # calculate time diffs and label diffs among subsequent
            # capture events
            time_diffs = [b - a for (a, b) in zip(loc_seconds_order[0:-1],
                                                  loc_seconds_order[1:])]
            label_diffs = [a != b for (a, b) in zip(loc_labels_order[0:-1],
                                                    loc_labels_order[1:])]

            # insert dummy data for first observation to ensure correct length
            time_diffs.insert(0, 0.0)
            label_diffs.insert(0, True)

            # assign ids
            split_ids_loc = list()
            run_id = -1
            # min. minutes difference for capture events of the same class
            # at the same location getting different splitting ids
            minutes_diff = 30
            new_id = k + '_' + str(run_id)
            # loop over all label and time diffs
            for lab_diff, time_diff in zip(label_diffs, time_diffs):
                if (lab_diff or (time_diff > (60*minutes_diff))):
                    run_id += 1
                    new_id = k + '_' + str(run_id)
                split_ids_loc.append(new_id)
            # save new split ids in dictionary
            jj = [time_order_ids.index(j) for j in
                  range(0, len(time_order_ids))]
            split_ids_loc_ord = [split_ids_loc[i] for i in jj]
            loc_dat[k]['split_ids'] = split_ids_loc_ord

        # map old ids on new split ids
        ids_map_old_new = dict()
        for k, v in loc_dat.items():
            for ii in range(0, len(v['ids'])):
                ids_map_old_new[v['ids'][ii]] = \
                 {'id': v['split_ids'][ii],
                  'lab': v['labels'][ii]}

        # retrieve new split ids in correct order
        split_ids = list()
        split_labels = list()
        for ii in ids_orig:
            split_ids.append(ids_map_old_new[ii]['id'])
            split_labels.append(ids_map_old_new[ii]['lab'])

    return ids_orig, split_ids, split_labels


if __name__ == '__main__':
    # Create dummy dataset
    import random
    random.seed(22)
    labels_all = ["elephant", "zebra", "monkey", "elephant_zebra"]
    ids_orig = [str(i) for i in range(0, 40)]
    labels_orig = [random.choice(labels_all) for i in range(0, len(ids_orig))]
    meta_data = dict()
    for ii in ids_orig:
        cc = {'location': random.choice(["T1", "T2"]),
              'date': "20170820",
              'time': random.choice(["1503" + str(int(random.uniform(10, 59))),
                                     "1502" + str(int(random.uniform(10, 59))),
                                     '1600' + str(int(random.uniform(10,
                                                                     59)))])}
        meta_data[ii] = cc

    meta_data['0']
    ids, split_ids, split_labels = createSplitIDs(ids_orig, labels_orig,
                                                  meta_data=meta_data,
                                                  split_mode="none")
    ids, split_ids, split_labels = createSplitIDs(
        ids_orig, labels_orig,
        meta_data=meta_data,
        split_mode="location_date_time")

    for i, ll, loc, tm in zip(ids_orig, labels_orig,
                              [meta_data[i]['location'] for i in ids_orig],
                              [meta_data[i]['time'] for i in ids_orig]):
        print("ID: %s - Label: %s - Loc: %s - Time: %s" % (i, ll, loc, tm))

    for i, lo, s, l in zip(ids, labels_orig, split_ids, split_labels):
        print("ID: %s, Split-ID: %s, Label-Orig: %s, Label-Split: %s" %
              (i, s, lo, l))

    from sklearn.model_selection import train_test_split
    from tools.subjects import SubjectSet
    labels = labels_orig
    # create id to label mapper
    class_mapper_id = dict()
    for i, l in zip(ids, labels):
        class_mapper_id[i] = l

    # create split id to label mapper
    class_mapper_split_id = dict()
    for i, l in zip(split_ids, split_labels):
        class_mapper_split_id[i] = l

    # mapper split ids to orig ids
    split_id_mapper = dict()
    for jj in range(0, len(split_ids)):
        if split_ids[jj] not in split_id_mapper:
            split_id_mapper[split_ids[jj]] = [ids_orig[jj]]
        else:
            split_id_mapper[split_ids[jj]].append(ids_orig[jj])

    # mapper orig id to split id
    id_to_split_id_mapper = dict()
    for k, v in split_id_mapper.items():
        for i in v:
            id_to_split_id_mapper[i] = k

    # get rid of all split ids of ids which have been removed by
    # class mapper and balanced sampling
    split_ids = [id_to_split_id_mapper[i] for i in ids]
    split_labels = [class_mapper_split_id[i] for i in split_ids]

    # deduplicate splitting ids to be used in creating test / train splits
    split_ids_dedup, split_labels_dedup = list(), list()
    for k, v in class_mapper_split_id.items():
        split_ids_dedup.append(k)
        split_labels_dedup.append(v)

    # training and test split
    id_train_s, id_test_s = train_test_split(split_ids_dedup,
                                             train_size=0.7,
                                             test_size=0.3,
                                             stratify=split_labels_dedup,
                                             random_state=1)

    # validation split
    labels_s_val = [class_mapper_split_id[x] for x in id_test_s]
    id_test_s, id_val_s = train_test_split(id_test_s,
                                           train_size=0.5,
                                           stratify=labels_s_val,
                                           random_state=1)

    # map split ids to original ids
    id_train = [[x for x in split_id_mapper[i]] for i in id_train_s]
    id_test = [[x for x in split_id_mapper[i]] for i in id_test_s]
    id_val = [[x for x in split_id_mapper[i]] for i in id_val_s]

    # get rid of sublists
    id_train = [item for sublist in id_train for item in sublist]
    id_test = [item for sublist in id_test for item in sublist]
    id_val = [item for sublist in id_val for item in sublist]

    set(id_train) & set(id_test)
    set(id_train) & set(id_val)
    set(id_val) & set(id_test)



    set_ids = [id_train, id_test, id_val]

    for si in set_ids:
        for i in si:
            # change label
            new_label = class_mapper_id[i]



    # get_most_rescent_file_with_string('D:\\Studium_GD\\Zooniverse\\Data\\' +
    #                                   'transfer_learning_project\\models\\3663',
    #                                   in_str='mnist_testing')
