import time
from collections import deque
from ddpg_agent import Agent
from unityagents import UnityEnvironment
import numpy as np
import torch
import matplotlib.pyplot as plt


def ddpg(n_episodes=100, max_t=100, print_every=10):
    scores_deque = deque(maxlen=print_every)
    scores = []
    last_ep_score = 0
    for i_episode in range(1, n_episodes +1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = 0
        for t in range(max_t):
            actions= agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            agent.step(states, actions, reward, next_state, done)
            states = next_state
            score += np.average(reward)
            if np.sum(done) != 0:
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if last_ep_score <= score:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            with open('scores.txt', 'w') as f:
                f.write(scores)
    return scores


if __name__ == "__main__":
    env = UnityEnvironment(file_name='Reacher_2.app', no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # size of each action
    action_size = brain.vector_action_space_size
    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print(states.shape)
    print(np.average(env_info.rewards))
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)

    scores = ddpg()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores ) +1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('scores_{}.png'.format(int(time.time())))
