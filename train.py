import gym
gym.logger.set_level(40)
import time
import random
from math import isclose
import matplotlib.pyplot as plt

from multiprocessing import Pool

enable_multithread = True

generation_size = 100
num_generations = 200

crossover_prob = 0.98
mutation_prob = 0.02
select_size = 3
max_iter_simulation = 1000

num_repeat = 5


def run_individual(args):

    individual, params = args

    env = gym.make('BipedalWalker-v2')
    num_actions = env.action_space.shape[0]
    num_observations = env.observation_space.shape[0]

    ending_states = []
    # avg_max_rew = 0
    # avg_last_rew = 0
    avg_sum_rew = 0

    for runind in range(params["repeat"]):

        observation = env.reset()
        max_rew = -100
        last_rew = -100
        sum_rew = 0
        last_observation = [0] * num_observations
        ending_state = "success"
        maxi = 0
        for i in range(params["simiter"]):
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

            # max_rew = max(max_rew, reward)
            # last_rew = reward
            sum_rew += reward

            if done:
                break

    # if ending_state == "success":
    #     yeahs += 1
    # print("%8s" % ending_state, end=" ")
        sum_rew += maxi / 20
    # print("->\t{}\t{}".format(maxi, sum_rew))

    # return ending_states, max_rew, last_rew, sum_rew
    # with open("log.txt", "a+") as logfile:
    #     logfile.write("{} {} {} {}\n".format(ind, sum_rew, ending_state, generation[ind]))
        avg_sum_rew += sum_rew
        ending_states.append(ending_state)
    return ending_states, avg_sum_rew/params["repeat"]


def run_generation(generation, params):
    max_rewards = []
    end_rewards = []
    total_rewards = []
    yeahs = 0

    if enable_multithread:
        result = p.imap(run_individual, map(lambda x: (x, params), generation))

        for item in result:

            # end_state, max_rew, last_rew, sum_rew = item
            end_states, avg_sum_rew = item

            for end_state in end_states:
                if end_state == "success":
                    yeahs += 1
            # max_rewards.append(max_rew)
            # end_rewards.append(last_rew)
            total_rewards.append(avg_sum_rew)
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

    # for i in range(len(max_rewards)):
        # print("{}: {} {} {}".format(i, total_rewards[i], end_rewards[i], max_rewards[i]))
    #
    return total_rewards, yeahs

def get_weighted_random(weights):
    r = random.uniform(0, sum(weights))

    for i in range(len(weights)):
        if r < weights[i]:
            return i
        r -= weights[i]


def get_new_generation(prev_generation, fitness, cross, mut, select):
    generation = []

    min_fitness = min(fitness)
    max_fitness = max(fitness)
    fitness = list(map(lambda x: (x - min_fitness)/(max_fitness - min_fitness), fitness))

    for individual in sorted(range(len(fitness)), key=lambda x: fitness[x], reverse=True)[:select]:
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

        if random.uniform(0.0, 1.0) <= cross:
            pos = random.randint(0, num_actions * num_observations)
            curr_ind = 0
            for ai in range(num_actions):
                for oi in range(num_observations):
                    if curr_ind >= pos:
                        child1[ai][oi], child2[ai][oi] = child2[ai][oi], child1[ai][oi]

        for ai in range(num_actions):
            for oi in range(num_observations):
                if random.uniform(0.0,1.0) <= mut:
                    child1[ai][oi] = random.uniform(-1.0,1.0)
                if random.uniform(0.0, 1.0) <= mut:
                    child2[ai][oi] = random.uniform(-1.0, 1.0)

        generation.append(child1)
        generation.append(child2)

    return generation


if __name__ == "__main__":

    p = Pool(20)

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

    for generation_size, num_generations, crossover_prob, mutation_prob, select_size, max_iter_simulation, num_repeat in [
        (100, 500, 0.95, 0.1, 10, 1000, 3)
    ]:

        # fill generation with random individuals
        generation = []
        for genind in range(generation_size):
            individual = []
            for i in range(num_actions):
                params_for_action = []
                for j in range(num_observations):
                    if num_observations - j <= 10:
                        params_for_action.append(0)
                    else:
                        params_for_action.append(random.uniform(-1, 1))
                individual.append(params_for_action)
            generation.append(individual)

        avg_fitnesses = []
        top_fitnesses = []
        yeahs_arr = []
        besties = []

        for generation_ind in range(num_generations):
            with open("log.txt", "a+") as logfile:
                logfile.write("Generation {}\n".format(generation_ind))
            fitness, yeahs = run_generation(generation, {
                "simiter": max_iter_simulation,
                "repeat": num_repeat,
            })
            avg_fitnesses.append(sum(fitness) / len(fitness))
            top_fitnesses.append(max(fitness))
            yeahs_arr.append(yeahs / len(fitness) / num_repeat)
            print("generation {}".format(generation_ind))
            print("Current top and avg fitness")
            print(top_fitnesses[-5:])
            print(avg_fitnesses[-5:])
            print(yeahs_arr[-5:])
            generation = get_new_generation(generation, fitness, crossover_prob, mutation_prob, select_size)
            besties.append(generation[0])

        with open("results.txt", "a") as resfile:
            resfile.write(
                "cross: {}\nmutation: {}\nselect: {}\nnum_iter: {}\ngen_size: {}\nnum_generations: {}\nnum_repeat: {}\n".format(
                    crossover_prob, mutation_prob, select_size, max_iter_simulation, generation_size, num_generations, num_repeat))
            resfile.write("best_fitnesses= {}\n".format(top_fitnesses))
            resfile.write("avg_fitnesses= {}\n".format(avg_fitnesses))
            resfile.write("not_falling= {}\n".format(yeahs_arr))
            resfile.write("besties= {}\n".format(besties))
            resfile.write("\n")

        plt.plot(top_fitnesses)
        plt.plot(avg_fitnesses)
        plt.savefig("fitness_{}_{}_{}_{}_{}_{}_{}.png".format(
                    crossover_prob, mutation_prob, select_size, max_iter_simulation, generation_size, num_generations, num_repeat))
        plt.clf()


