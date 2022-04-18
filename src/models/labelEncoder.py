class LabelEncoder:
    def __init__(self):
        self.labels_to_indices = {}

    @property
    def n_label(self):
        return len(self.labels_to_indices)

    def get_state(self):
        return self.labels_to_indices

    def set_state(self, state):
        self.labels_to_indices = state

    def update_new_labels(self, new_labels):
        new_labels_to_indices = {
            label: index + self.n_label for index, label in enumerate(new_labels)
        }
        self.labels_to_indices.update(new_labels_to_indices)

    def fit(self, labels):
        old_unique_labels = set(self.labels_to_indices.keys())
        all_unique_labels = old_unique_labels.union(set(labels))
        new_unique_labels = sorted(
            list(all_unique_labels.difference(old_unique_labels))
        )
        self.update_new_labels(new_unique_labels)

    def transform(self, labels):
        return [self.labels_to_indices[label] for label in labels]

    def fit_transform(self, labels):
        self.fit(labels)
        return self.transform(labels)

    def inverse_transform(self, indices):
        indices_to_labels = {}
        for label in self.labels_to_indices:
            indices_to_labels[self.labels_to_indices[label]] = label
        return [indices_to_labels[index] for index in indices]
