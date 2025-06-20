#Single model tournament selection 
import random
import sklearn.neighbors as neighbors
import sklearn.metrics as metrics
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# Loads the breast cancer dataset from ucimlrepo. Also works for spam dataset but I never ended up using it, default is BC
def load_dataset(dataset_name):
    if dataset_name == 'breast_cancer':
        X, y = datasets.load_breast_cancer(return_X_y=True)
    else:  # spam not used
        spambase = fetch_ucirepo(id=94)
        X = spambase.data.features.values           
        y = spambase.data.targets.values.ravel()    
    return X, y
    

# Generate numbers to represent features for selection (number 1 = feature 1, ect...)
def generate_feature_numbers(features_x):
    return list(range(len(features_x[0])))

# Random individual (feature subset) generator (max depth = max subset size)
def subset_generator(max_depth, features_x):
    selected = []
    remaining = generate_feature_numbers(features_x).copy()
    depth = max_depth
    while depth > 0 and remaining and random.random() >= 0.5:
        feature = random.choice(remaining)
        selected.append(feature)
        remaining.remove(feature)
        depth -= 1
    return selected

# Make dataset of only features in selected subset
def selected_subset_features(features, subset):
    if not subset:
        return features
    filtered = [i for i in subset if 0 <= i < len(features[0])]
    if not filtered:
        return features
    return [[row[i] for i in filtered] for row in features]

# Compute inverted F1 error for a feature subset using train_test_split of 80/20
def error_selection(individual, X, y):
    data = selected_subset_features(X, individual)
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.2, shuffle=True, random_state=42
    )

    knn = neighbors.KNeighborsClassifier()
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    f1 = metrics.f1_score(y_test, preds, average='macro')
    return 1 - f1

# Sort a population of individuals (feature sets) by their F1 score
def sort_by_f1score_selection(population, X, y):
    scored = [(error_selection(ind, X, y), ind) for ind in population]
    scored.sort(key=lambda tup: tup[0])
    return [ind for _, ind in scored]

# Mutation function: generate a new random individual (feature subset)
def mutate_feature_selection(ind, X, max_depth):
    new = subset_generator(max_depth, X)
    return new if new != ind else ind

# Crossover function: one-point split and swap for two subsets
def crossover_feature_selection(i, j):
    if not i or not j:
        return i
    pt = random.randint(1, len(i))
    child = i[:pt] + j[pt:]
    seen = set()
    result = []
    for f in child:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result

# Tournament selection, default size 6
def select_in_population(population, X, y, tournament_size=6):
    competitors = random.sample(population, tournament_size)
    return min(competitors, key=lambda ind: error_selection(ind, X, y))

# Determine optimal depth for subset size (in this case full subset)
def set_optimal_depth_feature_selection(features_x):
    # return max(len(features_x[0]) // 2)
    return len(features_x[0])

# Main evolutionary loop for tournament based feature selection
def evolve_feature_selection(X, y, popsize, max_depth, dataset_name, target_error):
    print(f"Starting feature selection evolution on '{dataset_name}' dataset...")
    population = [subset_generator(max_depth, X) for _ in range(popsize)]
    population = sort_by_f1score_selection(population, X, y)
    generation = 0
    while True:
        best = population[0]
        best_err = error_selection(best, X, y)
        print(f"Gen {generation}: Best error={best_err:.4f}, Subset={best}")
        if best_err < target_error:
            print(f"Success! Best subset found for '{dataset_name}': {best}")
            break
        next_pop = []
        # Mutation
        for _ in range(int(0.2 * popsize)):
            parent = select_in_population(population, X, y)
            next_pop.append(mutate_feature_selection(parent, X, max_depth))
        # Crossover
        for _ in range(int(0.3 * popsize)):
            p1 = select_in_population(population, X, y)
            p2 = select_in_population(population, X, y)
            next_pop.append(crossover_feature_selection(p1, p2))
        # Selection
        for _ in range(int(0.7 * popsize)):
            next_pop.append(select_in_population(population, X, y))
        population = sort_by_f1score_selection(next_pop, X, y)
        generation += 1

if __name__ == '__main__':
    print("program start")
    X, y = load_dataset('breast_cancer')
    print("dataset loaded")
    print("program start")
    X, y = load_dataset('breast_cancer')
    print("dataset loaded")
    depth = 30
    popsize = 50 # change popsize to desired amount, determined experimentally
    print('popsize = ', popsize) 
    print("depth set to ", depth)
    target = 0.01
    evolve_feature_selection(X, y, popsize, max_depth=depth,
                             dataset_name='breast_cancer', target_error=target)
    print("program end")

