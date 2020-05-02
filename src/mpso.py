import csv
import random
from pathlib import Path

from common import get_common_parser, display_results, set_creator, get_toolbox, evaluate_particles, get_pso_parameters


def parse_args():
    parser = get_common_parser()
    parser.add_argument("-se", "--swarmEpochs", type=int, default=3, help='number of epochs per swarm')
    args = parser.parse_args()
    return args


def run_pso(args, pop, toolbox, writer, mpso_epoch):
    epoch = 0
    bestValue = None

    while epoch < args.swarmEpochs:
        weight, phi1, phi2 = get_pso_parameters(args)
        best = evaluate_particles(pop, toolbox)

        for part in pop:
            writer.writerow([part.fitness.values[0], mpso_epoch * args.swarmEpochs + epoch])
            toolbox.update(part, best, weight, phi1, phi2)

        bestValue = toolbox.evaluate(best)[0]
        epoch += 1

    return bestValue


def run_mpso(args):
    Path("../results/mpso/").mkdir(parents=True, exist_ok=True)
    fitness_file = open('../results/mpso/fitness.csv', 'w', newline='')
    with fitness_file:
        writer = csv.writer(fitness_file)
        writer.writerow(('fitness', 'epoch'))
        toolbox = get_toolbox(args.size, args.pminimum, args.pmaximum, args.function)
        pop = toolbox.population(n=args.population)

        swarm_size = int(args.population/args.subswarms)
        epoch = 0
        best_accuracy = float("inf")
        while (args.epoch is None or epoch < args.epoch) and (args.accuracy is None or best_accuracy > args.accuracy):
            swarms = [pop[x:x+swarm_size] for x in range(0, len(pop), swarm_size)]
            for swarm in swarms:
                best_value = run_pso(args, swarm, toolbox, writer, epoch)
                accuracy = abs(args.solution - best_value)
                if best_accuracy > accuracy:
                    best_accuracy = accuracy
            random.shuffle(pop)
            epoch += 1

    return epoch, best_accuracy


def main():
    args = parse_args()

    set_creator(args.minimum)

    epochs = []
    accuracies = []
    for i in range(args.iteration):
        result = run_mpso(args)
        accuracies.append(result[1])
        if accuracies[i] <= args.accuracy:
            epochs.append(result[0] * args.swarmEpochs)

    display_results(epochs, accuracies, args.accuracy)


if __name__ == "__main__":
    main()
