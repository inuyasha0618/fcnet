import numpy as np

class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, model, data, learning_rate=1e-3, batch_size=10, epoch_nums=100):
		
		self.model = model
		self.X_train = data['X_train']
		self.y_train = data['y_train']
		self.X_val = data['X_val']
		self.y_val = data['y_val']
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.epoch_nums = epoch_nums	
		
	def _step(self):

		N = self.X_train.shape[0]

		mask = np.random.choice(N, self.batch_size)

		X_batch = self.X_train[mask]

		y_batch = self.y_train[mask]

		loss, grads = self.model.loss(X_batch, y_batch)
		
		for param_key, grad in grads.items():

			self.model.params[param_key] -= self.learning_rate * grad

		return loss

	def train(self):

		N = self.X_train.shape[0]
		
		iters_per_epoch = max(1, N // self.batch_size)

		iter_nums = self.epoch_nums * iters_per_epoch

		loss_cache = []

		for i in range(iter_nums):

			loss = self._step()

			loss_cache.append(loss)

			if (i + 1) % iters_per_epoch == 0:
				print('epoch%d: train acc: %.2f val acc %.2f'%((i + 1) // iters_per_epoch, self.check_train_accuracy(), self.check_val_accuracy()))

		return loss_cache

	def check_train_accuracy(self):

		predict = self.model.predict(self.X_train)

		accuracy = np.mean(predict == self.y_train)

		return accuracy

	def check_val_accuracy(self):

		predict = self.model.predict(self.X_val)

		accuracy = np.mean(predict == self.y_val)

		return accuracy
		
