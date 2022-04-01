class LabelEncoder():
    def __init__(self):
        self.unique_labels = {}


    def save_unique_labels_to_indices_dictionary(self, unique_labels):
        self.unique_labels = {unique_label: index for index, unique_label in enumerate(unique_labels)}


    def fit(self, labels):
        old_unique_labeles = set(self.unique_labels.keys())
        all_unique_labels = old_unique_labeles.union(set(labels))
        new_unique_labels = list(all_unique_labels.difference(old_unique_labeles))
        self.save_unique_labels_to_indices_dictionary(list(self.unique_labels.keys()) + new_unique_labels)


    def transform(self, labels):
        return [self.unique_labels[label] for label in labels]
