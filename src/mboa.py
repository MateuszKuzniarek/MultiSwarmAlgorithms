import operator

from deap import base, creator, tools
from numpy import random

from common import get_common_parser, display_and_save_results, generate_particle, get_function, save_fitness_history, \
    merge_best_histories, save_best_fitness_history


def parse_args():
    parser = get_common_parser()
    parser.add_argument("-c", "--sensorModality", type=float, default=0.3, help='sensor modality')
    parser.add_argument("-sp", "--switchProbability", type=float, default=0.6, help='switch probability')
    parser.add_argument("-ae", "--aExponent", type=float, default=0.3, help='a exponent')
    parser.add_argument("-se", "--swarmEpochs", type=int, default=3, help='number of epochs per swarm')
    args = parser.parse_args()
    return args


def evaluate_butterflies(pop, toolbox):
    best = None
    for part in pop:
        part.fitness.values = toolbox.evaluate(part)
        if not best or best.fitness < part.fitness:
            best = creator.Particle(part)
            best.fitness.values = part.fitness.values

    return best


def set_fragrance(part, sensor_modality, a_exponent):
    part.fragrance = sensor_modality * (part.fitness.values[0] ** a_exponent)


def move_butterfly(part, vector1, vector2):
    random_number = random.rand()
    random_squared = random_number * random_number
    modification = [i * random_squared for i in vector1]
    modification = [modification[i] - vector2[i] for i in range(0, len(modification))]
    modification = [i * part.fragrance for i in modification]
    part[:] = list(map(operator.add, part, modification))


def move_towards_best(part, best):
    move_butterfly(part, best, part)


def move_randomly(part, random_part_1, random_part_2):
    move_butterfly(part, random_part_1, random_part_2)


def run_boa(args, pop, toolbox, mboa_epoch):
    epoch = 0
    history = []
    best_history = []
    best = evaluate_butterflies(pop, toolbox)

    while epoch < args.swarmEpochs:
        best_history.append([best.fitness.values[0], mboa_epoch * args.swarmEpochs + epoch])
        for part in pop:
            set_fragrance(part, args.sensorModality, args.aExponent)
        for part in pop:
            history.append([part.fitness.values[0], mboa_epoch * args.swarmEpochs + epoch])
            random_number = random.rand()
            if random_number < args.switchProbability:
                move_towards_best(part, best)
            else:
                random_index_1 = random.choice(len(pop))
                random_index_2 = random.choice(len(pop))
                move_randomly(part, pop[random_index_1], pop[random_index_2])

        best = evaluate_butterflies(pop, toolbox)
        best_value = toolbox.evaluate(best)[0]
        epoch += 1

    return best_value, history, best_history


def run_mboa(args):
    toolbox = base.Toolbox()
    toolbox.register("particle", generate_particle, size=args.size, pmin=args.pminimum, pmax=args.pmaximum)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("evaluate", get_function(args.function))
    pop = toolbox.population(n=args.population)

    swarm_size = int(args.population/args.subswarms)
    epoch = 0
    best_accuracy = float("inf")
    history = []
    best_history = []

    while (args.epoch is None or epoch < args.epoch) and (args.accuracy is None or best_accuracy > args.accuracy):
        swarms = [pop[x:x+swarm_size] for x in range(0, len(pop), swarm_size)]
        best_history_fragment = []
        for swarm in swarms:
            best_value, partial_history, best_partial_history = run_boa(args, swarm, toolbox, epoch)
            merge_best_histories(best_history_fragment, best_partial_history, args.minimum)
            history += partial_history
            accuracy = abs(args.solution - best_value)
            if best_accuracy > accuracy:
                best_accuracy = accuracy
        best_history += best_history_fragment
        random.shuffle(pop)
        epoch += 1
    return epoch, best_accuracy, history, best_history


def main():
    args = parse_args()

    creator.create("Maximum", base.Fitness, weights=(1.0,))
    creator.create("Minimum", base.Fitness, weights=(-1.0,))
    fitness = creator.Minimum if args.minimum else creator.Maximum
    creator.create("Particle", list, fitness=fitness, fragrance=float)

    best_histories = []
    epochs = []
    accuracies = []
    for i in range(args.iteration):
        result = run_mboa(args)
        save_fitness_history("../results/" + args.logCatalog + "/", result[2])
        best_histories.append(result[3])
        accuracies.append(result[1])
        if accuracies[i] <= args.accuracy:
            epochs.append(result[0] * args.swarmEpochs)

    save_best_fitness_history("../results/" + args.logCatalog + "/", best_histories)
    display_and_save_results(epochs, accuracies, args.accuracy, "../results/" + args.logCatalog + "/")


if __name__ == "__main__":
    main()
