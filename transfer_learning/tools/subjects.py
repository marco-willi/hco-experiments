from sklearn.preprocessing import LabelEncoder

class SubjectSet(object):
    def __init__(self, labels):
        self.labels = labels
        self.subjects = dict()
        self.le = LabelEncoder()
        self.le.fit(self.labels)
        self.labels_num = self.le.transform(self.labels)

    def addSubject(self, subject_id, subject):
        self.subjects[subject_id] = subject

    def getSubject(self, subject_id):
        return self.subjects[subject_id]

    def getLabelEncoder(self):
        return self.le

    def getAllIDs(self):
        return self.subjects.keys()

    def getAllIDsLabels(self):
        ids = list()
        labels = list()
        for k, value in self.subjects:
            ids.append(k)
            labels.append(value.getLabel())
        return ids, labels

    def getAllURLsLabels(self):
        """ return all URLS and corresponding labels """
        urls = list()
        labels = list()
        for k, value in self.subjects.items():
            for url in value.getURLs():
                labels.append(value.getLabel())
                urls.append(url)
        return urls, labels


    def getAllURLsLabelsIDs(self):
        """ return all URLS and corresponding labels """
        urls = list()
        labels = list()
        ids = list()
        for k, value in self.subjects.items():
            for url in value.getURLs():
                labels.append(value.getLabel())
                urls.append(url)
                ids.append(k)
        return urls, labels, ids


class Subject(object):
    """ Subject definition """
    def __init__(self, identifier, label, meta_data=None, urls=None,
                 label_num=None):
        self.identifier = identifier
        self.label = label
        self.meta_data = meta_data
        if isinstance(urls, list):
            self.urls = urls
        else:
            self.urls = [urls]
        self.label_num = label_num

    def getLabel(self):
        return self.label

    def getURLs(self):
        return self.urls

    def setURLs(self, urls):
        self.urls = urls

    def setMetaData(self, meta_data):
        self.meta_data = meta_data


