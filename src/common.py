import argparse
import random
import operator
import numpy

from deap import creator, tools
from deap import benchmarks
from deap import base


def get_common_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--iteration", type=int, default=1)
    parser.add_argument("-n", "--population", type=int, default=10, help='number of parts')
    parser.add_argument("-N", "--size", type=int, default=2, help='size')
    parser.add_argument("-f", "--function", type=str, required=True, help='selected function')
    parser.add_argument("-pmin", "--pminimum", type=float, default=-100.0, help='partition minimum')
    parser.add_argument("-pmax", "--pmaximum", type=float, default=100.0, help='partition maximum')
    parser.add_argument("-s", "--solution", type=float, default=0.0, help='expected solution (for -a stop condition)')
    parser.add_argument("-ss", "--subswarms", type=int, default=3, help='number of sub-swarms')

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-min", "--minimum", action="store_true", help='minimum searching mode')
    mode.add_argument("-max", "--maximum", action="store_true", help='maximum searching mode')

    parser.add_argument("-e", "--epoch", type=int, help='number of epoch')
    parser.add_argument("-a", "--accuracy", type=float, help='expected accuracy')

    weighOptions = parser.add_mutually_exclusive_group(required=True)
    weighOptions.add_argument("-w", "--weight", type=float, help='constant inertial weight')
    weighOptions.add_argument("-rw", "--randomWeight", action="store_true", help='random inertial weight')

    accelerationOptions = parser.add_mutually_exclusive_group(required=True)
    accelerationOptions.add_argument("-phi" "--phi", type=float, nargs=2, help='acceleration factors')
    accelerationOptions.add_argument("-rphi", "--randomPhi", action="store_true", help='random acceleration factors')

    return parser


def generate_particle(size, pmin, pmax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [0 for _ in range(size)]
    return part


def update_particle(part, best, weight, phi1, phi2):
    baseSpeed = map(lambda v: v * weight, part.speed)
    r1_phi1 = (random.uniform(0, 1) * phi1 for _ in range(len(part)))
    r2_phi2 = (random.uniform(0, 1) * phi2 for _ in range(len(part)))
    v_u1 = list(map(operator.mul, r1_phi1, map(operator.sub, part.best, part)))
    v_u2 = list(map(operator.mul, r2_phi2, map(operator.sub, best, part)))
    part.speed = list(map(operator.add, baseSpeed, map(operator.add, v_u1, v_u2)))

    part[:] = list(map(operator.add, part, part.speed))


def sphere(individual):
    return benchmarks.sphere(individual)


def display_results(epochs, accuracies, expected_accuracy):
    print(epochs)
    is_solution_found_list = list(map(lambda acc: acc <= expected_accuracy, accuracies))
    print(is_solution_found_list)
    print(is_solution_found_list.count(True) / len(is_solution_found_list) * 100)
    if len(epochs) > 0:
        print(numpy.average(epochs))
    else:
        print("-")


def set_creator(is_minimum):
    creator.create("Maximum", base.Fitness, weights=(1.0,))
    creator.create("Minimum", base.Fitness, weights=(-1.0,))
    fitness = creator.Minimum if is_minimum else creator.Maximum
    creator.create("Particle", list, fitness=fitness, speed=list, best=None)


def get_toolbox(size, pminimum, pmaximum, function):
    toolbox = base.Toolbox()
    toolbox.register("particle", generate_particle, size=size, pmin=pminimum, pmax=pmaximum)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", update_particle)
    toolbox.register("evaluate", globals()[function])
    return toolbox


def evaluate_particles(pop, toolbox):
    best = None
    for part in pop:
        part.fitness.values = toolbox.evaluate(part)
        if not part.best or part.best.fitness < part.fitness:
            part.best = creator.Particle(part)
            part.best.fitness.values = part.fitness.values
        if not best or best.fitness < part.fitness:
            best = creator.Particle(part)
            best.fitness.values = part.fitness.values

    if best not in pop:
        print('a')
    return best


def get_pso_parameters(args):
    weight = random.uniform(0.1, 1) if args.randomWeight else args.weight
    phi1 = random.uniform(0.5, 1.5) if args.randomPhi else args.phi__phi[0]
    phi2 = random.uniform(0.5, 1.5) if args.randomPhi else args.phi__phi[1]
    return weight, phi1, phi2