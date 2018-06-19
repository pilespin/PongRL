
import gym
import time

def random_episode(env):
	x = []
	y = []
	observation = env.reset()
	for i in range(1000):
		# time.sleep(0.01)
		env.render()
		action = env.action_space.sample()
		x.append(observation)
		y.append(action)
		observation, reward, done, info = env.step(action)
		# print(env.unwrapped.get_action_meanings())
		if done or reward == 1:
			return x, y, i+1
		if reward == -1:
			break
	return [], [], 0




env = gym.make('Pong-v0')
allAction = env.unwrapped.get_action_meanings()
print(allAction)

for episode in range(200):
	x, y, i = random_episode(env)
	print('x', i)

