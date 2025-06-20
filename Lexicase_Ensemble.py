#Evoltionary algorithm for feature seleciton, using Lexicase selection to generate the optimal feature subset for an ensamble of three 
#SKLearn models, KNN, SVM, Random Forest. 

#LEXICASE ENSAMBEL
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from functools import lru_cache

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

# Cache per-classifier error computations to avoid recomputing repeats
# Calculate error (1-f1 score) for each of three classifers (random forest: rf, K-nearest-neighbors: knn, support vector machine: svm) 
@lru_cache(maxsize=None)
def measure_classifier_errors(individual_tuple):
    # use globals for data and split
    features = selected_subset_features(X_global, list(individual_tuple))
    X_train, X_test = features[train_idx_global], features[test_idx_global]
    y_train, y_test = y_global[train_idx_global], y_global[test_idx_global]

    rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    knn = KNeighborsClassifier(n_jobs=-1)
    svm = SVC(probability=True, random_state=42)

    rf.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    rf_err  = 1.0 - f1_score(y_test, rf.predict(X_test),  average='macro')
    knn_err = 1.0 - f1_score(y_test, knn.predict(X_test), average='macro')
    svm_err = 1.0 - f1_score(y_test, svm.predict(X_test), average='macro')
    return {'rf': rf_err, 'knn': knn_err, 'svm': svm_err}

# Compute inverted F1 error (averaged over the three models) for a feature subset using train_test_split of 80/20
@lru_cache(maxsize=None)
def error_selection(individual_tuple):
    features = selected_subset_features(X_global, list(individual_tuple))
    X_train, X_test = features[train_idx_global], features[test_idx_global]
    y_train, y_test = y_global[train_idx_global], y_global[test_idx_global]

    rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    knn = KNeighborsClassifier(n_jobs=-1)
    svm = SVC(probability=True, random_state=42)

    rf.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    avg_prob = (rf.predict_proba(X_test) + knn.predict_proba(X_test) + svm.predict_proba(X_test)) / 3.0
    y_pred   = np.argmax(avg_prob, axis=1)
    return 1.0 - f1_score(y_test, y_pred, average='macro')

# Lexicase selection: select individual based on performance on each classifier
# This method is based off pseudocode presented by Thomas Helmuth and William La Cava in their
# GECCO 2021 Lexicase Selection Tutorial
def lexicase_selection(population):
    performance = {tuple(ind): measure_classifier_errors(tuple(ind)) for ind in population}

    objectives = list(next(iter(performance.values())).keys())
    random.shuffle(objectives)
    survivors = population.copy()
    for obj in objectives:
        min_err = min(performance[tuple(ind)][obj] for ind in survivors)
        survivors = [ind for ind in survivors if performance[tuple(ind)][obj] == min_err]
        if len(survivors) == 1:
            return survivors[0]
    return random.choice(survivors)

# Override previous selection method to use Lexicase, allows for options to use other seleciton methods if want to combine this later
def select_in_population(population):
    return lexicase_selection(population)

# Crossover function: one-point split and swap for two subsets
def crossover_feature_selection(a, b):
    if not a or not b:
        return a
    pt = random.randint(1, len(a))
    merged = a[:pt] + b[pt:]
    seen, out = set(), []
    for f in merged:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out

# Sort population, ranked by soft-voting error on the three models
def sort_by_f1score_selection(population):
    scored = [(error_selection(tuple(ind)), ind) for ind in population]
    scored.sort(key=lambda x: x[0])
    return [ind for _, ind in scored]

# Main evolutionary loop for feature selection, using lexicase selection, same mutation, crossover and selection rates as tournament selection
def evolve_feature_selection(popsize, max_depth, dataset_name, target_error):
    print(f"Starting optimized Lexicase‐based feature selection on '{dataset_name}'...")
    population = [subset_generator(max_depth, X_global) for _ in range(popsize)]
    population = sort_by_f1score_selection(population)

    generation = 0
    while True:
        best = population[0]
        best_err = error_selection(tuple(best))
        print(f"Gen {generation}: Best error={best_err:.4f}, Subset={best}")
        if best_err < target_error:
            print(f"Success! Best subset found for '{dataset_name}': {best}")
            break

        next_pop = []
        # Mutation
        for _ in range(int(0.2 * popsize)):
            parent = select_in_population(population)
            next_pop.append(mutate_feature_selection(parent, X_global, max_depth))
        # Crossover
        for _ in range(int(0.3 * popsize)):
            p1 = select_in_population(population)
            p2 = select_in_population(population)
            next_pop.append(crossover_feature_selection(p1, p2))
        # Selection
        for _ in range(int(0.7 * popsize)):
            next_pop.append(select_in_population(population))

        population = sort_by_f1score_selection(next_pop)
        generation += 1

# Mutation helper function (quick fix, maybe remove later)

def mutate_feature_selection(parent, X, max_depth):
    child = subset_generator(max_depth, X)
    return child 

if __name__ == "__main__":
    print("dataset loaded")
    depth = X_global.shape[1]
    print("depth set to", depth)
    popsize = 50 #set popsize at will
    evolve_feature_selection(popsize=popsize, max_depth=depth, dataset_name='breast_cancer', target_error=0.01)
    print("program end")
