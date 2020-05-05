import operator

from deap import creator
from numpy import random

from common import get_common_parser, display_results, set_creator, get_toolbox, evaluate_particles, get_pso_parameters, \
    save_fitness_history, add_pso_args_to_parser, save_best_fitness_history


def parse_args():
    parser = get_common_parser()
    add_pso_args_to_parser(parser)
    args = parser.parse_args()
    return args


def update_swarm(args, pop, toolbox, epoch):
    weight, phi1, phi2 = get_pso_parameters(args)
    history = []
    best = evaluate_particles(pop, toolbox)
    for part in pop:
        if part != best:
            history.append([part.fitness.values[0], epoch])
            random_chance = random.rand()
            if random_chance < 0.6:
                toolbox.update(part, best, weight, phi1, phi2)
            else:
                part[:] = list(map(operator.mul, part, random.normal(0, 1, len(part))))

    best_value = toolbox.evaluate(best)[0]
    return best, best_value, history


def update_elite_particles(current_swarm, swarms, epoch):
    new_position = []
    history = []
    for i in range(0, len(current_swarm.best)):
        dimension_sum = 0
        for swarm in swarms:
            if swarm != current_swarm:
                dimension_sum += swarm.best[i]
        value = (dimension_sum / (len(swarms) - 1)) * (1 + random.normal(0, 1))
        new_position.append(value)
    best_index = current_swarm.index(current_swarm.best)
    for i in range(0, len(current_swarm.best)):
        current_swarm[best_index][i] = new_position[i]
    history.append([current_swarm[best_index].fitness.values[0], epoch])
    return history


def run_mspso(args):
    toolbox = get_toolbox(args.size, args.pminimum, args.pmaximum, args.function)
    pop = toolbox.population(n=args.population)
    history = []
    best_history = []
    swarm_size = int(args.population/args.subswarms)
    divided_pop = [pop[x:x+swarm_size] for x in range(0, len(pop), swarm_size)]
    swarms = []
    for pop_fragment in divided_pop:
        swarms.append(creator.Swarm(pop_fragment))
    epoch = 0
    best_accuracy = float("inf")
    while (args.epoch is None or epoch < args.epoch) and (args.accuracy is None or best_accuracy > args.accuracy):
        best_fitness = None
        for swarm in swarms:
            swarm.best, swarm.best_value, partial_history = update_swarm(args, swarm, toolbox, epoch)
            history += partial_history
            if best_fitness is None or best_fitness < swarm.best.fitness.values[0]:
                best_fitness = swarm.best.fitness.values[0]
                accuracy = abs(args.solution - swarm.best_value)
                if best_accuracy > accuracy:
                    best_accuracy = accuracy
        best_history.append((best_fitness, epoch))
        for swarm in swarms:
            elite_particles_history = update_elite_particles(swarm, swarms, epoch)
            history += elite_particles_history
        epoch += 1
    return epoch, best_accuracy, history, best_history


def main():
    args = parse_args()
    set_creator(args.minimum)
    creator.create("Swarm", list, best=list, best_value=None)
    best_histories = []
    epochs = []
    accuracies = []
    for i in range(args.iteration):
        result = run_mspso(args)
        save_fitness_history("../results/mspso/", result[2])
        best_histories.append(result[3])
        accuracies.append(result[1])
        if accuracies[i] <= args.accuracy:
            epochs.append(result[0])

    save_best_fitness_history("../results/mspso/", best_histories)
    display_results(epochs, accuracies, args.accuracy)


if __name__ == "__main__":
    main()
