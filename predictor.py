class Predictor():
	def __init__(self) -> None:
		self.weight = 0
		self.bias = 0
		self.x_min = 0
		self.x_max = 0
		self.y_min = 0
		self.y_max = 0

	def predict(self, x: float) -> float:
		norm_x = (x - self.x_min) / (self.x_max - self.x_min)
		norm_y = self.weight * norm_x + self.bias
		pred_y = norm_y * (self.y_max - self.y_min) + self.y_min
		return pred_y

	def load(self, filepath: str) -> None:
		fd = open(filepath, "rt")
		datas = fd.readline().split(',')
		self.weight = float(datas[0])
		self.bias = float(datas[1])
		self.x_max = float(datas[2])
		self.x_max = float(datas[3])
		self.y_min = float(datas[4])
		self.y_max = float(datas[5])
		fd.close()

if __name__ == "__main__":
	predictor = Predictor()
	predictor.load("./predictor_const.txt")
	x = float(input("Input mileage(km): "))
	print("Predicted price: ", predictor.predict(x))
