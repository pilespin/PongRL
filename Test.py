
import tensorflow as tf
from tfHelper import tfHelper

import gym
import time
import random
import numpy as np
import cv2

import predict
import common


class Test:

	k = tf.keras
	c = common

	def __init__(self):
		'tfHelper Initialized'
		tfHelper.log_level_decrease()


	def predict_one(self, model, img):
		# img = tfHelper.image_to_array("new/5/1/28.png", self.c.convertColor)
		img = np.array([img])
		img = self.c.normalize(img)
		pred = predict.predict(model, img)
		return (pred[0])

	def save_episode(self, frame_nb, observation, action, path):
		self.c.mkdir_p(path + str(action))
		new_path = path + str(action) + '/' + str(frame_nb) + '.png'
		# print('saved: ' + new_path)
		cv2.imwrite(new_path, observation[:,:,::-1])

	def random_episode(self, env, model):
		x = []
		y = []
		observation = env.reset()
		for i in range(500):
			# time.sleep(0.01)
			# env.render()
			# print ("observation", observation.shape)
			# if i%4 == 0:
			if i%(10) == 0:
				# print ("Random play")
				action = random.randint(0,3)
			else:
				action = self.predict_one(model, observation)
				# action = random.randint(0,5)
			# action = random.randint(0,5)
			observation, reward, done, info = env.step(action)
			x.append(observation)
			y.append(action)
			# print(env.unwrapped.get_action_meanings())
			if done or reward == 1:
				return x, y, i+1
			if reward == -1:
				break
		return [], [], 0


	def test(self, model):

	# model = tfHelper.load_model("model")

		env = gym.make('Pong-v0')
		allAction = env.unwrapped.get_action_meanings()
		print(allAction)

		folder = "new/"
		self.c.mkdir_p(folder)

		nb_episode = 100
		win = 0
		for episode in range(1, nb_episode):
			x, y, len = self.random_episode(env, model)
			if len > 0:
				win += 1
				path = str(len) + '_' + str(self.c.count_elem_in_folder(folder)) + '/'
				self.c.mkdir_p(folder + path)
				for i, (observation, action) in enumerate(zip(x,y)):
					if i > 20:
						self.save_episode(i, observation, action, folder + path)

			# print('x', len)
		print ("WIN " + str(win) + "/" + str(nb_episode))

	# exit(0)

