import gym
import time
import random
from math import isclose

from multiprocessing import Pool

enable_multithread = True

generation_size = 100
num_generations = 20

crossover_prob = 0.98
mutation_prob = 0.02
select_size = 5
max_iter_simulation = 1000

def run_individual(individual):
    observation = env.reset()
    max_rew = -100
    last_rew = -100
    sum_rew = 0
    last_observation = [0] * num_observations
    ending_state = "success"
    maxi = 0
    for i in range(max_iter_simulation):
        maxi = i
        # if i % 5 == 0:
        #     env.render()
        # time.sleep(0.05)
        actions = []
        for act_ind in range(num_actions):
            action = 0
            for observ_ind in range(num_observations):
                action += individual[act_ind][observ_ind] * observation[observ_ind]
            action = min(1.0, action)
            action = max(-1.0, action)
            actions.append(action)

        observation, reward, done, info = env.step(actions)

        if done:
            ending_state = "fall"
        elif all([isclose(x, y, abs_tol=1e-06) for x, y in zip(observation, last_observation)]):
            ending_state = "stuck"
            reward = -50
            done = True

        last_observation = observation

        max_rew = max(max_rew, reward)
        last_rew = reward
        sum_rew += reward

        if done:
            break

    # if ending_state == "success":
    #     yeahs += 1
    print("%8s" % ending_state, end=" ")
    sum_rew += maxi / 20
    print("->\t{}\t{}".format(maxi, sum_rew))
    return ending_state, max_rew, last_rew, sum_rew
    # with open("log.txt", "a+") as logfile:
    #     logfile.write("{} {} {} {}\n".format(ind, sum_rew, ending_state, generation[ind]))

def run_generation(generation):
    max_rewards = []
    end_rewards = []
    total_rewards = []
    yeahs = 0

    if enable_multithread:
        p = Pool(4)
        result = p.map(run_individual, generation)

        for ind in range(len(result)):

            end_state, max_rew, last_rew, sum_rew = result[ind]

            if end_state == "success":
                yeahs += 1
            max_rewards.append(max_rew)
            end_rewards.append(last_rew)
            total_rewards.append(sum_rew)
    else:
        for ind in range(len(generation)):
            print("Running for individual number: \t{}".format(ind), end=" ")
            individual = generation[ind]

            end_state, max_rew, last_rew, sum_rew = run_individual(individual)

            if end_state == "success":
                yeahs += 1
            max_rewards.append(max_rew)
            end_rewards.append(last_rew)
            total_rewards.append(sum_rew)

    for i in range(len(max_rewards)):
        print("{}: {} {} {}".format(i, total_rewards[i], end_rewards[i], max_rewards[i]))

    return total_rewards, yeahs

def get_weighted_random(weights):
    r = random.uniform(0, sum(weights))

    for i in range(len(weights)):
        if r < weights[i]:
            return i
        r -= weights[i]


def get_new_generation(prev_generation, fitness):
    generation = []

    min_fitness = min(fitness)
    max_fitness = max(fitness)
    fitness = list(map(lambda x: (x - min_fitness)/(max_fitness - min_fitness), fitness))

    for individual in sorted(range(len(fitness)), key=lambda x: fitness[x], reverse=True)[:select_size]:
        generation.append(prev_generation[individual])

    while len(generation) < len(prev_generation):
        i = get_weighted_random(fitness)
        j = get_weighted_random(fitness)
        if i == j:
            continue

        child1 = []
        child2 = []
        for ai in range(num_actions):
            action1 = []
            action2 = []
            for oi in range(num_observations):
                action1.append(prev_generation[i][ai][oi])
                action2.append(prev_generation[j][ai][oi])
            child1.append(action1)
            child2.append(action2)

        if random.uniform(0.0, 1.0) <= crossover_prob:
            pos = random.randint(0, num_actions * num_observations)
            curr_ind = 0
            for ai in range(num_actions):
                for oi in range(num_observations):
                    if curr_ind >= pos:
                        child1[ai][oi], child2[ai][oi] = child2[ai][oi], child1[ai][oi]

        for ai in range(num_actions):
            for oi in range(num_observations):
                if random.uniform(0.0,1.0) <= mutation_prob:
                    child1[ai][oi] = random.uniform(-1.0,1.0)
                if random.uniform(0.0, 1.0) <= mutation_prob:
                    child2[ai][oi] = random.uniform(-1.0, 1.0)

        generation.append(child1)
        generation.append(child2)

    return generation


if __name__ == "__main__":

    env = gym.make('BipedalWalker-v2')

    observation = env.reset()
    print(observation)

    print("Action space:")
    print(env.action_space)
    print(env.action_space.low)
    print(env.action_space.high)

    print("Observation space:")
    print(env.observation_space)
    print(env.observation_space.low)
    print(env.observation_space.high)

    num_actions = env.action_space.shape[0]
    num_observations = env.observation_space.shape[0]


    # fill generation with random individuals
    generation = []
    for genind in range(generation_size):
        individual = []
        for i in range(num_actions):
            params_for_action = []
            for j in range(num_observations):
                params_for_action.append(random.uniform(-1.0, 1.0))
            individual.append(params_for_action)
        generation.append(individual)

    avg_fitnesses = []
    top_fitnesses = []
    yeahs_arr = []
    besties = []

    for generation_ind in range(num_generations):
        with open("log.txt", "a+") as logfile:
            logfile.write("Generation {}\n".format(generation_ind))
        fitness, yeahs = run_generation(generation)
        avg_fitnesses.append(sum(fitness) / len(fitness))
        top_fitnesses.append(max(fitness))
        yeahs_arr.append(yeahs / len(fitness))
        print("generation {}".format(generation_ind))
        print("Current top and avg fitness")
        print(top_fitnesses)
        print(avg_fitnesses)
        print(yeahs_arr)
        generation = get_new_generation(generation, fitness)
        besties.append(generation[0])

    with open("results.txt", "a") as resfile:
        resfile.write(
            "cross: {}, mutation: {}, select: {}, num_iter: {}, gen_size: {}, num_generations: {}\n".format(
                crossover_prob, mutation_prob, select_size, max_iter_simulation, generation_size, num_generations))
        resfile.write("best_fitnesses= {}\n".format(top_fitnesses))
        resfile.write("avg_fitnesses= {}\n".format(avg_fitnesses))
        resfile.write("not falling= {}\n".format(yeahs_arr))
        resfile.write("besties= {}\n".format(besties))
        resfile.write("\n")


