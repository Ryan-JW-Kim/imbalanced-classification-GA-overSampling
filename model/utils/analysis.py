from collections import Counter

def get_minority_label(y):
    minority_label = 0
    sample_counts = Counter(y)
    for label in sample_counts:
        if sample_counts[label] < sample_counts[minority_label]:
            minority_label = label
    return minority_label