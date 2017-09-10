class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, model=None, X=None, y=None, learning_rate=1e-3, batch_size=10, epoch_nums=1000):
		
		self.model = model
		self.X = X
		self.y = y
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.iter_nums = iter_nums	
		
	def _step(self, batch_x, batch_y):

		loss, grads = self.model.loss(batch_x, batch_y)
		
		for param_key, grad in grads.items():
			self.model.params[param_key] -= learning_rate * grad

	def train(self):

		N = self.X.shape[0]
		
		iters_per_epoch = N // self.batch_size

		
