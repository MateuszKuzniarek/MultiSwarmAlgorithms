import random

from common import get_common_parser, display_and_save_results, set_creator, get_toolbox, evaluate_particles, get_pso_parameters, \
    save_fitness_history, add_pso_args_to_parser, merge_best_histories, save_best_fitness_history


def parse_args():
    parser = get_common_parser()
    add_pso_args_to_parser(parser)
    parser.add_argument("-se", "--swarmEpochs", type=int, default=3, help='number of epochs per swarm')
    args = parser.parse_args()
    return args


def run_pso(args, pop, toolbox, mpso_epoch):
    epoch = 0
    history = []
    best_history = []

    while epoch < args.swarmEpochs:
        weight, phi1, phi2 = get_pso_parameters(args)
        best = evaluate_particles(pop, toolbox)
        best_history.append([best.fitness.values[0], mpso_epoch * args.swarmEpochs + epoch])
        for part in pop:
            history.append([part.fitness.values[0], mpso_epoch * args.swarmEpochs + epoch])
            toolbox.update(part, best, weight, phi1, phi2)
        epoch += 1

    return best, history, best_history


def run_mpso(args):
    toolbox = get_toolbox(args.size, args.pminimum, args.pmaximum, args.function)
    pop = toolbox.population(n=args.population)

    history = []
    best_history = []
    swarm_size = int(args.population/args.subswarms)
    epoch = 0
    best_accuracy = float("inf")
    while (args.epoch is None or epoch < args.epoch) and (args.accuracy is None or best_accuracy > args.accuracy):
        swarms = [pop[x:x+swarm_size] for x in range(0, len(pop), swarm_size)]
        best_history_fragment = []
        for swarm in swarms:
            best, partial_history, best_partial_history = run_pso(args, swarm, toolbox, epoch)
            history += partial_history
            merge_best_histories(best_history_fragment, best_partial_history, args.minimum)
            accuracy = abs(args.solution - toolbox.evaluate(best)[0])
            if best_accuracy > accuracy:
                best_accuracy = accuracy
        best_history += best_history_fragment
        random.shuffle(pop)
        epoch += 1
    return epoch, best_accuracy, history, best_history


def main():
    args = parse_args()

    set_creator(args.minimum)
    best_histories = []
    epochs = []
    accuracies = []
    for i in range(args.iteration):
        result = run_mpso(args)
        save_fitness_history("../results/" + args.logCatalog + "/", result[2])
        best_histories.append(result[3])
        accuracies.append(result[1])
        if accuracies[i] <= args.accuracy:
            epochs.append(result[0] * args.swarmEpochs)

    save_best_fitness_history("../results/" + args.logCatalog + "/", best_histories)
    display_and_save_results(epochs, accuracies, args.accuracy, "../results/" + args.logCatalog + "/")


if __name__ == "__main__":
    main()
