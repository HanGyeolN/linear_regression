import matplotlib.pyplot as plt

class TrainData():
	def __init__(self):
		self.index = []
		self.data = {}
		self.normalized_data = {}
		self.len = 0
	
	def read_csv(self, filepath):
		fd = open(filepath, 'rt')
		first_line = fd.readline()
		self.index = first_line[:-1].split(',')
		for idx in self.index:
			self.data[idx] = []
			self.normalized_data[idx] = []
		for line in fd.readlines():
			splited = line[:-1].split(',')
			length = len(splited)
			for i in range(length):
				self.data[self.index[i]].append(splited[i])
		for idx in self.index:
			self.normalized_data[idx] = self.normalize(self.data[idx])
		self.len = len(self.data[self.index[0]])	

	def normalize(self, values):
		normalized_list = []
		min = min(values)
		max = max(values)
		for x in values:
			normalized_list.append((x - min) / (max - min))
		return normalized_list

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
		x = data.data[data.index[0]]
		y = data.data[data.index[1]]
		for i in range(data.len):
			sum = sum + ((self.estimate(x[i]) - y[i]) * x[i])
		return (sum / data.len)

	def get_gradient_bias(self, data: TrainData):
		"""
		cost 함수의 bias에 대한 현재지점의 기울기를 구한다.
		"""
		sum = 0
		x = data.data[data.index[0]]
		y = data.data[data.index[1]]
		for i in range(data.len):
			sum = sum + ((self.estimate(x[i])) - y[i])
		return (sum / data.len)

	def get_cost(self, data: TrainData):
		"""
		실제 데이터와 예측값의 오차 제곱의 합을 반환한다.
		"""
		sum = 0
		x = data.data[data.index[0]]
		y = data.data[data.index[1]]
		for i in range(data.len):
			sum = sum + pow((data[i]['price'] - self.estimate(data[i]['km'])), 2)
		return (sum / data.len)
	
	def batch_gradient_descent(self, data):
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
			print(f"epoch: {self.train_count} w: {self.weight} b: {self.bias} cost: {self.get_cost(data)}")

# 1. 훈련 과정 시각화
# 2. 정규화
if __name__ == "__main__":
	data = read_data('./data.csv')
	plot_data(data)
	x = []
	y = []
	i = 0
	t0 = 0
	t1 = 0
	learning_rate0 = 0.01
	learning_rate1 = 0.00000000015
	while (i < 2000):
		cost = get_cost(t1, t0, data)

		tmp_t0 = get_tmp_t0(learning_rate0, data, t0, t1)
		tmp_t1 = get_tmp_t1(learning_rate1, data, t0, t1)

		t1 = t1 - tmp_t1
		t0 = t0 - tmp_t0


		# x.append(t1)
		# y.append(cost)
		print(f"epoch: {i} w: {t1} b: {t0} cost: {cost}")
		i += 1
		# t0 = tmp_t0
		# t1 = tmp_t1
		# print(i, ": [", t0, ", ", t1, "]")
		# i += 1

	# 	x = [0,10, 300000]
	# 	y = []
	# 	for val in x:
	# 		y.append(t0 + t1 * val)
	# 	plt.plot(x, y)
		
	x = [0,10, 300000]
	y = []
	for val in x:
		y.append(t0 + t1 * val)
	line = plt.plot(x, y)
	plt.setp(line, color='r', linewidth=3.0)
	plt.ylim(0, 10000)
	# print(x)
	# print(y)
	# plt.plot(x, y)
	plt.show()