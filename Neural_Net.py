

class Neural_Net(object):
	"""docstring for Neural_Net"""
	def __init__(self):
		# Defining Hyperparameters
		self.inputLayerSize 	= 2
		self.outputLayerSize 	= 1
		self.hiddenLayerSize 	= 3

		# Weights (parameters)
		self.w1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.w2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

	def forward(self, X):
		#Propagate inputs through network

		self.z2 = np.dot(X, self.w1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.w2)

		yHat	= self.sigmoid(self.z3)
		return yHat

	def sigmoid(self, z):
		#Apply sigmoid activation function to scalar, vector, or matrix
		return 1/(1+np.exp(-z))
		



