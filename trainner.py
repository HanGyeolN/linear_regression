import matplotlib.pyplot as plt
from LinearTrainner import LinearTrainner
from TrainData import TrainData

if __name__ == "__main__":
	train_data = TrainData()
	train_data.read_csv("./data.csv")

	trainner = LinearTrainner()
	trainner.train_logs(train_data, 1000)
	trainner.save("predictor_const.txt", train_data.x_min, train_data.x_max, train_data.y_min, train_data.y_max)

