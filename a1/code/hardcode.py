import numpy as np

class HardCodedTree:

	def predict(x):
		if (x[0] > -80.305106):
			if (x[1] > 36.453576):
				return 0
			else:
				return 0
		else:
			if (x[1] > 37.669007):
				return 0
			else:
				return 1
			