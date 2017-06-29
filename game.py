import argparse
import gym
import ppaquette_gym_super_mario
import os
import numpy as np
from copy import deepcopy
from random import randrange, sample
from neat import nn, population, statistics

parser = argparse.ArgumentParser(description='OpenAI Gym Solver')
parser.add_argument('--max-steps', dest='max_steps', type=int, default=5000,
                    help='The max number of steps to take per genome (timeout)')
parser.add_argument('--episodes', type=int, default=1,
                    help="The number of times to run a single genome. This takes the fitness score from the worst run")
parser.add_argument('--generations', type=int, default=50,
                    help="The number of generations to evolve the network")
parser.add_argument('--checkpoint', type=str,
                    help="Uses a checkpoint to start the simulation")
args = parser.parse_args()



def simulate_species(net, episodes=1, steps=5000):
    fitnesses = []
    actions_before = []
    for runs in range(episodes):
        my_env = gym.make('ppaquette/meta-SuperMarioBros-Tiles-v0')
        my_env.render()
        inputs = my_env.reset()
        cum_reward = 0.0
        cont = 0;
        for j in range(steps):
            inputs = inputs.flatten()
            outputs = net.serial_activate(inputs)
            outputs1 = outputs[:len(outputs)/2]
            outputs2 = outputs[len(outputs)/2:]
            actions1 = get_actions(outputs1)
            inputs, reward, is_finished, info = my_env.step(actions1)
            cum_reward = info['distance']
            if info['life'] == 0:
                break
            actions2 = get_actions(outputs2)
            if not np.array_equal(actions1, actions2):
                my_env.step([0, 0, 0, 0, 0, 0])
            inputs, reward, is_finished, info = my_env.step(actions2)
            cum_reward_before = cum_reward
            cum_reward = info['distance']
            if cum_reward_before == cum_reward:
                cont = cont + 1
            if cont == 50:
                break
            if info['life'] == 0:
                break
        my_env.close()
        fitnesses.append(cum_reward)

    my_env.close()
    fitness = np.array(fitnesses).mean()
    print("Species fitness: %s" % str(fitness))
    return fitness

def get_actions(outputs):
    actions = [0, 0, 0, 0, 0, 0]
    for i in range(6):
        if outputs[i] > 0:
            actions[i] = 1
    return actions


def actions_is_active(actions):
    for i in range(len(actions)):
        if actions[i] == 1:
            return True
    return False

def worker_evaluate_genome(g):
    net = nn.create_feed_forward_phenotype(g)
    return simulate_species(net, args.episodes, args.max_steps)

def train_network():

    def evaluate_genome(g):
        net = nn.create_feed_forward_phenotype(g)
        return simulate_species(net, args.episodes, args.max_steps)

    def eval_fitness(genomes):
        for g in genomes:
            fitness = evaluate_genome(g)
            g.fitness = fitness

    # Simulation
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'game_config')
    pop = population.Population(config_path)
    # Load checkpoint
    if args.checkpoint:
        pop.load_checkpoint(args.checkpoint)
    # Start simulation
    pop.run(eval_fitness, args.generations)

    pop.save_checkpoint("checkpoint")

    # Log statistics.
    statistics.save_stats(pop.statistics)
    statistics.save_species_count(pop.statistics)
    statistics.save_species_fitness(pop.statistics)

    print('Number of evaluations: {0}'.format(pop.total_evaluations))

    # Show output of the most fit genome against training data.
    winner = pop.statistics.best_genome()

    # Save best network
    import pickle
    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

    print('\nBest genome:\n{!s}'.format(winner))
    print('\nOutput:')

    raw_input("Press Enter to run the best genome...")
    winner_net = nn.create_feed_forward_phenotype(winner)
    for i in range(100):
        simulate_species(winner_net, 1, args.max_steps)

train_network()
