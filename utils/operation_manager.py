import numpy as np
import gym



def run_one_episode(env, agent, training):

    gym_ver = int(gym.__version__.split('.')[1])

    agent.reset_episode()

    output  = env.reset()
    if  gym_ver >= 25:
        state, _ = output
    else:
        state = output

    reward = None

    rewards = []

    done = False
    while not done:
        action = agent.act(reward, state, training, done)

        output = env.step(action)
        if  gym_ver >= 25:
            state, reward, terminated, truncated, info = output
            done = terminated or truncated
        else:
            state, reward, done, info = output

        rewards.append(reward)
    agent.act(reward, state, training, done)

    num_steps = len(rewards)

    return num_steps


def run_episodes(env, agent, num_episodes, new_agent, training):

    assert not new_agent or training, "A new agent must be trained"

    if new_agent:
        agent.reset()

    episode_lengths = []

    episode = 0

    while num_episodes > episode:
        num_steps = run_one_episode(env, agent, training)
        episode += 1
        if episode % 10 == 0: 
            print(num_steps)

        episode_lengths.append(num_steps)

    return episode_lengths


def run_repeatedly(env, agent, num_episodes, num_repetitions,
                   new_agent, training):

    avg_episode_lengths = [0] * num_episodes

    for i in range(num_repetitions):
        print("repetition: ",i)
        episode_lengths = run_episodes(env, agent, num_episodes,
                                       new_agent, training)

        avg_episode_lengths = [ x+y for x,y in zip(avg_episode_lengths,
                                                   episode_lengths)]

    avg_episode_lengths = [ x / num_repetitions for x in avg_episode_lengths]

    return avg_episode_lengths

