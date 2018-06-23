
import tensorflow as tf
from tfHelper import tfHelper

import gym
import time
import random
import numpy as np
import cv2

import predict
import common

tfHelper.log_level_decrease()
c = common

def predict_one(model, img):
	# img = tfHelper.image_to_array("new/5/1/28.png", c.convertColor)
	img = np.array([img])
	img = c.normalize(img)
	pred = predict.predict(model, img)
	return (pred[0])

def save_episode(frame_nb, observation, action, path):
	c.mkdir_p(path + str(action))
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
		# print ("observation", observation.shape)
		if i%50 == 0:
			action = random.randint(0,5)
		else:
			action = predict_one(model, observation)
		observation, reward, done, info = env.step(action)
		x.append(observation)
		y.append(action)
		# print(env.unwrapped.get_action_meanings())
		if done or reward == 1:
			return x, y, i+1
		if reward == -1:
			break
	return [], [], 0


model = tfHelper.load_model("model")

env = gym.make('Pong-v0')
allAction = env.unwrapped.get_action_meanings()
print(allAction)

folder = "new/"
c.mkdir_p(folder)

for episode in range(1, 1000):
	x, y, len = random_episode(env)
	if len > 0:
		path = str(len) + '_' + str(c.count_elem_in_folder(folder)) + '/'
		c.mkdir_p(folder + path)
		for i, (observation, action) in enumerate(zip(x,y)):
			if i > 20:
				save_episode(i, observation, action, folder + path)

	print('x', len)

exit(0)

