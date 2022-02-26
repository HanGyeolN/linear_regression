class TrainData():
	def __init__(self):
		self.index = []
		self.data = {}
		self.normalized_data = {}
		self.len = 0
		self.x_min = 0
		self.x_max = 0
		self.y_min = 0
		self.y_max = 0
	
	def read_csv(self, filepath: str):
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
				self.data[self.index[i]].append(float(splited[i]))
		self.__normalize()
		self.len = len(self.data[self.index[0]])

	def __normalize(self):
		x_index = self.index[0]
		y_index = self.index[1]
		self.normalized_data[x_index] = self.normalize(self.data[x_index])
		self.x_min, self.x_max = min(self.data[x_index]), max(self.data[x_index])
		self.normalized_data[y_index] = self.normalize(self.data[y_index])
		self.y_min, self.y_max = min(self.data[y_index]), max(self.data[y_index])

	def normalize(self, values: list) -> list:
		normalized_list = []
		min_val = min(values)
		max_val = max(values)
		for x in values:
			normalized_list.append((x - min_val) / (max_val - min_val))
		return normalized_list
	
	def denormalize(self, values: list, min_val, max_val) -> list:
		denorm_list = []
		for x in values:
			denorm_list.append(x * (max_val - min_val) + min_val)
		return denorm_list
