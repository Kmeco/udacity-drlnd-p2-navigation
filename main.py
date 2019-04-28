import time
from collections import deque
from ddpg_agent import Agent
from unityagents import UnityEnvironment
import numpy as np
import torch
import matplotlib.pyplot as plt


def ddpg(n_episodes=150, max_t=1000, print_every=10):
    session = int(time.time())
    scores_deque = deque(maxlen=print_every)
    mean_scores = []  # list of mean scores from each episode
    min_scores = []  # list of lowest scores from each episode
    max_scores = []  # list of highest scores from each episode
    moving_avgs = []                               # list of moving averages
    last_ep_score = 0
    for i_episode in range(1, n_episodes +1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        scores = np.zeros(num_agents)
        start_time = time.time()
        for t in range(max_t):
            actions= agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            agent.step(states, actions, reward, next_state, done, t)
            states = next_state
            scores += reward
            if np.sum(done) != 0:
                break
        duration = time.time() - start_time
        min_scores.append(np.min(scores))             # save lowest score for a single agent
        max_scores.append(np.max(scores))             # save highest score for a single agent
        mean_scores.append(np.mean(scores))           # save mean score for the episode

        scores_deque.append(mean_scores[-1])         # save mean score to window
        moving_avgs.append(np.mean(scores_deque))    # save moving average

        scores_deque.append(mean_scores[-1])
        print('\rEpisode {} ({} sec)  -- \tMin: {:.1f}\tMax: {:.1f}\tMean: {:.1f}\tMov. Avg: {:.1f}'.format(i_episode,
                            round(duration), min_scores[-1], max_scores[-1], mean_scores[-1], moving_avgs[-1]), end="")
        if last_ep_score <= mean_scores[-1]:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {} ({} sec)  -- \tMin: {:.1f}\tMax: {:.1f}\tMean: {:.1f}\tMov. Avg: {:.1f}'.format(
                    i_episode,round(duration), min_scores[-1], max_scores[-1], mean_scores[-1], moving_avgs[-1]))

            with open('scores_{}.txt'.format(session), 'w') as f:
                f.write(str(scores))
        if moving_avgs[-1] >= 30:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_final.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_final.pth')
            with open('scores_final_{}.txt'.format(session), 'w') as f:
                f.write(str(scores))
        last_ep_score = mean_scores[-1]
    return mean_scores, moving_avgs


if __name__ == "__main__":
    load = 0

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
    num_agents = len(env_info.agents)
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)

    if load:
        agent.actor_local.load_state_dict(torch.load('ch_a.pth'))
        agent.actor_target.load_state_dict(torch.load('ch_a.pth'))
        agent.critic_local.load_state_dict(torch.load('ch_c.pth'))
        agent.critic_target.load_state_dict(torch.load('ch_c.pth'))

    scores = ddpg()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores ) +1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('scores_{}.png'.format(int(time.time())))
