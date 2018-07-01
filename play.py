
from Test import Test
from tfHelper import tfHelper

import model

import os


te = Test()

while True:
	if os.path.exists("model.h5"):
		model = tfHelper.load_model("model")

	te.test(model)
