import csv
import random
from pathlib import Path

from deap import creator

from common import get_common_parser, display_results, set_creator, get_toolbox, evaluate_particles, get_pso_parameters


def parse_args():
    parser = get_common_parser()
    args = parser.parse_args()
    return args


def update_swarm(args, pop, toolbox, writer, epoch):
    weight, phi1, phi2 = get_pso_parameters(args)

    best = evaluate_particles(pop, toolbox)
    for part in pop:
        if part != best:
            writer.writerow([part.fitness.values[0], epoch])
            toolbox.update(part, best, weight, phi1, phi2)

    best_value = toolbox.evaluate(best)[0]
    return best, best_value


def update_elite_particles(current_swarm, swarms):
    new_position = []
    for i in range(0, len(current_swarm.best)):
        dimension_sum = 0
        for swarm in swarms:
            if swarm != current_swarm:
                dimension_sum += swarm.best[i]
        new_position.append(dimension_sum / (len(swarms) - 1))
    best_index = current_swarm.index(current_swarm.best)
    for i in range(0, len(current_swarm.best)):
        current_swarm[best_index][i] = new_position[i]

def run_mspso(args):
    Path("../results/mspso/").mkdir(parents=True, exist_ok=True)
    fitness_file = open('../results/mspso/fitness.csv', 'w', newline='')
    with fitness_file:
        writer = csv.writer(fitness_file)
        writer.writerow(('fitness', 'epoch'))
        toolbox = get_toolbox(args.size, args.pminimum, args.pmaximum, args.function)
        pop = toolbox.population(n=args.population)
        swarm_size = int(args.population/args.subswarms)
        divided_pop = [pop[x:x+swarm_size] for x in range(0, len(pop), swarm_size)]
        swarms = []
        for pop_fragment in divided_pop:
            swarms.append(creator.Swarm(pop_fragment))
        epoch = 0
        best_accuracy = float("inf")
        while (args.epoch is None or epoch < args.epoch) and (args.accuracy is None or best_accuracy > args.accuracy):
            for swarm in swarms:
                swarm.best, swarm.best_value = update_swarm(args, swarm, toolbox, writer, epoch)
                accuracy = abs(args.solution - swarm.best_value)
                if best_accuracy > accuracy:
                    best_accuracy = accuracy
            for swarm in swarms:
                update_elite_particles(swarm, swarms)
            epoch += 1

    return epoch, best_accuracy


def main():
    args = parse_args()

    set_creator(args.minimum)
    creator.create("Swarm", list, best=list, best_value=None)

    epochs = []
    accuracies = []
    for i in range(args.iteration):
        result = run_mspso(args)
        accuracies.append(result[1])
        if accuracies[i] <= args.accuracy:
            epochs.append(result[0])

    display_results(epochs, accuracies, args.accuracy)


if __name__ == "__main__":
    main()
