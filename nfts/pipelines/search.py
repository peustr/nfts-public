from timeit import default_timer as timer

import numpy as np
from scipy.spatial.distance import hamming

from nfts.pipelines.core import validate


def topk_diverse(population, fitness, diversity_threshold, k):
    topk = [population[fitness.argmax()]]
    topk_fitness = [fitness.max()]
    for i in range(k - 1):
        candidate = topk[-1]
        distances = np.asarray([hamming(candidate, population[j]) for j in range(population.shape[0])])
        population = population[distances >= diversity_threshold]
        fitness = fitness[distances >= diversity_threshold]
        assert population.shape[0] == fitness.shape[0]
        if population.shape[0] == 0:
            break
        topk.append(population[fitness.argmax()])
        topk_fitness.append(fitness.max())
    return topk, topk_fitness


def random_search(model, data_loader, args):
    paths = np.unique(
        np.random.rand(args.max_evaluations, model.num_decisions).round().astype(np.uint8),
        axis=0,
    )
    fitness_scores = _evaluate_population(model, data_loader, paths, args)
    print(f"Best candidate: {paths[fitness_scores.argmax()]}, Fitness: {fitness_scores.max()}\n")
    # topk_population, topk_fitness = topk_diverse(population, fitness, topk_return_diversity_threshold, 3)
    return paths[fitness_scores.argmax()]


def evolutionary_search(model, data_loader, args):
    def recombine(g1, g2):
        assert g1.shape == g2.shape
        g3 = np.zeros_like(g1)
        for i in range(g1.shape[0]):
            if np.random.rand() < 0.5:
                g3[i] = g1[i]
            else:
                g3[i] = g2[i]
        return g3

    def mutate(g):
        for i in range(g.shape[0]):
            if np.random.rand() < 0.05:
                g[i] = not g[i]
        return g

    population_seed = None
    if population_seed is None:
        paths = np.random.rand(args.population_size, model.num_decisions).round().astype(np.uint8)
    else:
        seed_size = population_seed.shape[0]
        paths = np.random.rand(args.population_size - seed_size, model.num_decisions).round().astype(np.uint8)
        paths = np.concatenate([population_seed, paths], axis=0)

    evaluation_budget = args.max_evaluations
    fitness_scores = _evaluate_population(model, data_loader, paths, args)
    evaluation_budget -= paths.shape[0]
    generation = 1
    while True:
        start_time = timer()
        topk_idc = fitness_scores.argpartition(-args.topk_crossover)[-args.topk_crossover :]
        topk = paths[topk_idc]
        offspring = np.asarray(
            [
                mutate(recombine(topk[i], topk[j]))
                for i in range(args.topk_crossover)
                for j in range(args.topk_crossover)
                if i != j
            ]
        )
        offspring_fitness_scores = _evaluate_population(model, data_loader, offspring, args)
        evaluation_budget -= offspring.shape[0]
        paths = np.concatenate([paths, offspring], axis=0)
        fitness_scores = np.concatenate([fitness_scores, offspring_fitness_scores], axis=0)
        topk_idc = fitness_scores.argpartition(-args.population_size)[-args.population_size :]
        paths = paths[topk_idc]
        fitness_scores = fitness_scores[topk_idc]
        print(_generation_summary(generation, paths, fitness_scores, evaluation_budget))
        end_time = timer()
        interval = (end_time - start_time) / 60.0
        print(f"Computed in {interval:.1f} minutes.")
        generation += 1
        if evaluation_budget <= 0:
            break
    print(f"Best candidate: {paths[fitness_scores.argmax()]}, Fitness: {fitness_scores.max()}\n")
    # topk_population, topk_fitness = topk_diverse(population, fitness, topk_return_diversity_threshold, 3)
    return paths[fitness_scores.argmax()]


def _evaluate_population(model, data_loader, population, args):
    num_evaluations = population.shape[0]
    fitness_scores = np.zeros(num_evaluations)
    for i_conf in range(num_evaluations):
        _, fitness_scores[i_conf] = validate(
            model,
            data_loader,
            args,
            path=population[i_conf],
        )
        print(f"[{i_conf + 1}/{num_evaluations}]", population[i_conf], ":", fitness_scores[i_conf])
    return fitness_scores


def _generation_summary(generation, population, fitness, evaluation_budget):
    s = f"Generation {generation:03} summary\n----------\n"
    s += f"Population fitness: {fitness.round(3)}\n"
    s += f"Best candidate in this generation: {population[fitness.argmax()]}, Fitness: {fitness.max()}\n"
    s += f"Remaining budget: {evaluation_budget} evaluations.\n"
    return s
