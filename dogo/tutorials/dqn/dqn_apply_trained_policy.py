import gym
from d3rlpy.algos import DQN

##########################################
# Baseed on the simple OpenAI Gym tutorial
# https://gym.openai.com/docs/
##########################################

# Load OpenAI Gym environment directly
env = gym.make('CartPole-v0')
# env = gym.make('Hopper-v3')

# Load algorithm
dqn = DQN()
dqn.build_with_env(env)

# Load model
dqn.load_model('d3rlpy_logs/DQN_20220510100030/model_24740.pt')

# Loop through however many episodes
for i_episode in range(1):
    observation = env.reset()
    for t in range(10000):

        # Cannot render from HPC terminal
        # env.render()

        print(observation)
        
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
