
class Labels(object):
     """ class to generate labels from data """
    def __init__(self):
        self.labels = None

    def getAnnotations():
        raise NotImplementedError

    def extractAnnotations():
        raise NotImplementedError

    def _annotationStatus():
        raise NotImplementedError

class PanoptesLabels(Labels):
    """ Class to handle annotations from Panoptes """
    def __init__(self, subs, le):
        super(PanoptesLabels, self).__init__()
        self.subs = subs
        self.le = le

    def extractAnnotations():


class panoptesAnnotation(annotation):
    """ Class to handle annotations from Panoptes """
    def __init__(self, subs, le)


    def

def generate_annotations_from_panoptes(subs, le):
    # generate labels from annotations (or somewhere else)
        labels = dict()
        for key, val in subs.items():
            if '#label' not in val['metadata']:
                next
            else:
                labels[key] = int(le.transform([val['metadata']['#label']]))

        # get subjects with labels
        subs_remove = subs.keys() - labels.keys()

        # remove subjects without label
        for rem in subs_remove:
            subs.pop(rem, None)

        ########################
        # Data Directory
        ########################

        # create generic dictionary to be used for the modelling part
        # contains generic id, y_label, url, subject_id
        data_dict = dict()
        i = 0
        for key, val in subs.items():
            data_dict[i] = {'y_data': int(le.transform([val['metadata']
                                                           ['#label']])),
                            'class': val['metadata']['#label'],
                            'url': val['url'],
                            'subject_id': key}
            i += 1

        return data_dict