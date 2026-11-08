from collections import defaultdict
import copy
import random


def balance_classes(pair_index, labels, seed=42):
    """
    Oversamples less occurring classes in labels until classes are balanced
    :param pair_index: list of tuples of indices representing drugs/proteins
    :param labels: list of ints representing classes
    :return: updated pair_index and labels
    """
    random.seed(seed)
    classes = defaultdict(list)
    for i, label in enumerate(labels):
        classes[label].append(i)

    max_class_count = max([len(classes[cls]) for cls in classes])
    for cls in classes:
        to_sample = max_class_count - len(classes[cls])
        assert to_sample >= 0
        if to_sample > 0:
            choices = classes[cls]
            result = copy.deepcopy(choices)
            for _ in range(to_sample // len(choices)):
                result += choices
            result += random.sample(choices, to_sample % len(choices))
            classes[cls] = result

    all_idx = []
    for cls in classes:
        all_idx += classes[cls]
    new_pair_index = [pair_index[i] for i in all_idx]
    new_labels = [labels[i] for i in all_idx]
    return new_pair_index, new_labels