from collections import deque
from ddpg_agent import Agent
from unityagents import UnityEnvironment
import torch
import matplotlib.pyplot as plt
import numpy as np
env = UnityEnvironment(file_name='Reacher_Linux_NoVis/Reacher.x86_64')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)
print("episode done ",env_info.local_done)
# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
agent = Agent(state_size=state_size, action_size=action_size, random_seed=3)
#scores = np.zeros(1)                          # initialize the score (for each agent)
print("hello")
def ddpg(n_episodes=5000, max_t=2000, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        #/print("mello")
        agent.reset()
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
        #print("kello")
        state = env_info.vector_observations                 # get the current state (for each agent)
        score = np.zeros(1)
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]           # send all actions to tne environment
            next_state = env_info.vector_observations         # get next state (for each agent)
            reward = env_info.rewards                        # get reward (for each agent)
            done = env_info.local_done                        # see if episode finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done[0]:
                break
        avg_score = np.mean(score)
        scores_deque.append(avg_score)
        scores.append(avg_score)
        print('\rEpisode {}\tAverage Score: {:.8f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_inter_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_inter_critic.pth') 
        if np.mean(scores_deque) >= 30:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    return scores

scores = ddpg()
fig = plt.figure()
ax = fig.add_subplot(111)
print(scores)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig("graph.png")
plt.show()
