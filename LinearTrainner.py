from TrainData import TrainData
import matplotlib.pyplot as plt

class LinearTrainner():
	def __init__(self):
		self.train_count = 0
		self.learning_rate = 0.1
		self.weight = 0
		self.bias = 0

	def estimate(self, x):
		return self.weight * x + self.bias

	def get_gradient_weight(self, data: TrainData):
		"""
		cost 함수의 weight에 대한 현재지점의 기울기를 구한다.
		"""
		sum = 0
		x = data.normalized_data[data.index[0]]
		y = data.normalized_data[data.index[1]]
		for i in range(data.len):
			sum = sum + ((self.estimate(x[i]) - y[i]) * x[i])
		return (sum / data.len)

	def get_gradient_bias(self, data: TrainData):
		"""
		cost 함수의 bias에 대한 현재지점의 기울기를 구한다.
		"""
		sum = 0
		x = data.normalized_data[data.index[0]]
		y = data.normalized_data[data.index[1]]
		for i in range(data.len):
			sum = sum + ((self.estimate(x[i])) - y[i])
		return (sum / data.len)

	def get_cost(self, data: TrainData):
		"""
		실제 데이터와 예측값의 오차 제곱의 합을 반환한다.
		"""
		sum = 0
		x = data.normalized_data[data.index[0]]
		y = data.normalized_data[data.index[1]]
		for i in range(data.len):
			sum = sum + pow((y[i] - self.estimate(x[i])), 2)
		return (sum / data.len)
	
	def batch_gradient_descent(self, data: TrainData):
		grad_b = self.get_gradient_bias(data)
		grad_w = self.get_gradient_weight(data)
		self.bias = self.bias - self.learning_rate * grad_b
		self.weight = self.weight - self.learning_rate * grad_w

	def update(self, data: TrainData):
		self.batch_gradient_descent(data)
		self.train_count += 1

	def train(self, data: TrainData, epoch):
		for i in range(epoch):
			self.update(data)
			if ((i + 1) % 10 == 0):
				print(f"epoch: {self.train_count} w: {self.weight} b: {self.bias} cost: {self.get_cost(data)}")

	def train_logs(self, data: TrainData, epoch) -> None:
		x = data.data[data.index[0]]
		y = data.data[data.index[1]]
		norm_x = data.normalize(x)
		for i in range(epoch):
			self.update(data)
			if ((i + 1) % 100 == 0):
				y_pred_norm = list(map(self.estimate, norm_x))
				y_pred = data.denormalize(y_pred_norm, data.y_min, data.y_max)
				plt.scatter(x, y)
				plt.xlim(data.x_min, data.x_max)
				plt.ylim(data.y_min, data.y_max)
				plt.plot(x, y_pred, color='green')
				print(f"epoch: {self.train_count} w: {self.weight:.4f} b: {self.bias:.4f} cost: {self.get_cost(data):.6f} dc/dw: {self.get_gradient_weight(data):.4f} dc/db: {self.get_gradient_bias(data):.4f}")
				plt.show()

	def save(self, filename, x_min, x_max, y_min, y_max) -> None:
		fd = open(filename, 'wt')
		save_str = f"{self.weight},{self.bias},{x_min},{x_max},{y_min},{y_max}"
		fd.write(save_str)
