#Evoltionary algorithm for feature seleciton, using Tournament selection to generate the optimal feature subset for an ensamble 
#of three SKLearn models, KNN, SVM, Random Forest. 

import random
import numpy as np
from functools import lru_cache
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#print('imports done')

# Load breast cancer dataset and precompute split indices as global variable to save time
X_global, y_global = load_breast_cancer(return_X_y=True)
X_global = np.array(X_global)
y_global = np.array(y_global)
train_idx_global, test_idx_global = train_test_split(
    np.arange(len(y_global)), test_size=0.2, random_state=42, shuffle=True
)

# Random individual (feature subset) generator (max depth = max subset size)
def subset_generator(max_depth, X):
    """Randomly generate a subset of features up to max_depth."""
    selected, remaining = [], list(range(X.shape[1]))
    depth = max_depth
    while depth > 0 and remaining and random.random() >= 0.5:
        feat = random.choice(remaining)
        selected.append(feat)
        remaining.remove(feat)
        depth -= 1
    return selected

# Make dataset of only features in selected subset
def selected_subset_features(X, subset):
    if not subset:
        return X
    filtered = [i for i in subset if 0 <= i < X.shape[1]]
    return X[:, filtered] if filtered else X

# Cache the error_selection to avoid redundant recomputation
# get individual model errors, soft vote for total
@lru_cache(maxsize=None)
def error_selection_cached(individual_tuple):
    X_sel = selected_subset_features(X_global, list(individual_tuple))
    X_train, X_test = X_sel[train_idx_global], X_sel[test_idx_global]
    y_train, y_test = y_global[train_idx_global], y_global[test_idx_global]

    rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    knn = KNeighborsClassifier(n_jobs=-1)
    svm = SVC(probability=True, random_state=42)

    rf.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    avg_proba = (
        rf.predict_proba(X_test) +
        knn.predict_proba(X_test) +
        svm.predict_proba(X_test)
    ) / 3.0
    y_pred   = np.argmax(avg_proba, axis=1)

    f1 = f1_score(y_test, y_pred, average='macro')
    return 1.0 - f1

def sort_by_f1score_selection(population):
    scored = [(error_selection_cached(tuple(ind)), ind) for ind in population]
    scored.sort(key=lambda x: x[0])
    return [ind for _, ind in scored]

#helper
def mutate_feature_selection(parent, X, max_depth):
    child = subset_generator(max_depth, X)
    return child

# Crossover function: one-point split and swap for two subsets

def crossover_feature_selection(a, b):
    if not a or not b:
        return a
    pt = random.randint(1, min(len(a), len(b)))
    merged = a[:pt] + b[pt:]
    # remove duplicates preserving order
    seen, out = set(), []
    for f in merged:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out

# Tournament selection, default size 6
def select_in_population(population, tournament_size=6):
    competitors = random.sample(population, tournament_size)
    return min(competitors, key=lambda ind: error_selection_cached(tuple(ind)))

# Main evolutionary loop for feature selection, using tournament selection

def evolve_feature_selection(popsize, max_depth, dataset_name, target_error):
    print(f"Starting tournament ensemble selection on '{dataset_name}'...")
    population = [subset_generator(max_depth, X_global) for _ in range(popsize)]
    population = sort_by_f1score_selection(population)

    generation = 0
    while True:
        best = population[0]
        best_err = error_selection_cached(tuple(best))
        print(f"Gen {generation}: Best error={best_err:.4f}, Subset={best}")
        if best_err < target_error:
            print(f"Success! Best subset: {best}")
            break

        next_pop = []
        # mutations (20%)
        for _ in range(int(0.2 * popsize)):
            parent = select_in_population(population)
            next_pop.append(mutate_feature_selection(parent, X_global, max_depth))
        # crossovers (30%)
        for _ in range(int(0.3 * popsize)):
            p1 = select_in_population(population)
            p2 = select_in_population(population)
            next_pop.append(crossover_feature_selection(p1, p2))
        # tournament (70%)
        for _ in range(int(0.7 * popsize)):
            next_pop.append(select_in_population(population))

        population = sort_by_f1score_selection(next_pop)
        generation += 1


if __name__ == "__main__":
    print("dataset loaded")
    depth = X_global.shape[1]
    print("depth set to", depth)
    evolve_feature_selection(
        X_global, y_global,
        popsize=50,
        max_depth=depth,
        dataset_name='breast_cancer',
        target_error=0.01
    )
    print("program end")
