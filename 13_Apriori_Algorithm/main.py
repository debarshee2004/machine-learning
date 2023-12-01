import numpy as np


def load_data():
    # Replace this with your dataset or generate sample data
    data = np.array([[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]])
    return data


def create_candidates(itemsets, k):
    candidates = set()
    for i in range(len(itemsets)):
        for j in range(i + 1, len(itemsets)):
            union_set = itemsets[i].union(itemsets[j])
            if len(union_set) == k:
                candidates.add(tuple(sorted(union_set)))
    return list(candidates)


def prune_candidates(candidates, prev_frequent, k):
    pruned_candidates = []
    for candidate in candidates:
        subsets = [
            set(combo)
            for combo in np.array(np.meshgrid(*candidate)).T.reshape(-1, len(candidate))
        ]
        if all(subset in prev_frequent for subset in subsets):
            pruned_candidates.append(candidate)
    return pruned_candidates


def find_frequent_itemsets(data, min_support):
    itemsets = [set([item]) for transaction in data for item in transaction]
    frequent_itemsets = []

    k = 2
    while itemsets:
        candidates = create_candidates(itemsets, k)
        candidate_counts = np.zeros(len(candidates))

        for transaction in data:
            for i, candidate in enumerate(candidates):
                if set(candidate).issubset(set(transaction)):
                    candidate_counts[i] += 1

        frequent_candidates = [
            candidate
            for i, candidate in enumerate(candidates)
            if candidate_counts[i] >= min_support
        ]
        frequent_itemsets.extend(frequent_candidates)

        itemsets = prune_candidates(candidates, frequent_candidates, k)
        k += 1

    return frequent_itemsets


if __name__ == "__main__":
    data = load_data()
    min_support = 2
    frequent_itemsets = find_frequent_itemsets(data, min_support)

    print("Frequent Itemsets:")
    for itemset in frequent_itemsets:
        print(set(itemset))
