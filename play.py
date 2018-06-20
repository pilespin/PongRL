
import gym
import time
import os
import random
import numpy as np

import cv2

def mkdir_p(path):
	if not os.path.exists(path):
		os.mkdir(path)

def save_episode(frame_nb, observation, action, path):
	mkdir_p(path + str(action))
	new_path = path + str(action) + '/' + str(frame_nb) + '.png'
	# print('saved: ' + new_path)
	cv2.imwrite(new_path, observation[:,:,::-1])

def random_episode(env):
	x = []
	y = []
	observation = env.reset()
	for i in range(1000):
		# time.sleep(0.01)
		# env.render()
		action = env.action_space.sample()
		# action = random.random(0,2)
		observation, reward, done, info = env.step(action)
		x.append(observation)
		y.append(action)
		# print(env.unwrapped.get_action_meanings())
		if done or reward == 1:
			return x, y, i+1
		if reward == -1:
			break
	# return x, y, i+1
	return [], [], 0




env = gym.make('Pong-v0')
allAction = env.unwrapped.get_action_meanings()
print(allAction)

folder = "new/"
mkdir_p(folder)

for episode in range(1, 2000):
	x, y, i = random_episode(env)
	if i > 0:
		path = str(episode) + '/'
		mkdir_p(folder + path)
		# i = 0
		for i, (observation, action) in enumerate(zip(x,y)):
			# i+=1
			if i > 20:
				save_episode(i, observation, action, folder + path)

	print('x', i)

exit(0)

