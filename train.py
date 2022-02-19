import matplotlib.pyplot as plt

def read_data(data_path):
	fd = open(data_path, 'rt')

	index = fd.readline()
	index = index[:-1]
	cols = index.split(',')
	data = []

	lines = fd.readlines()
	for line in lines:
		temp = line[:-1]
		temp = temp.split(',')
		data.append({cols[0]: int(temp[0]), cols[1]: int(temp[1])})

	fd.close()
	return data

def plot_data(data):
	for dot in data:
		plt.scatter(dot['km'], dot['price'])

def estimate(t0, t1, x):
	return t0 + t1 * x

def get_tmp_t0(learning_rate, data, t0, t1):
	i = 0
	sum = 0
	m = len(data)
	while (i < m):
		sum = sum + ((estimate(t0, t1, data[i]['km'])) - data[i]['price'])
		i += 1
	tmp_t0 = learning_rate * (sum / m)
	
	return tmp_t0

def get_tmp_t1(learning_rate, data, t0, t1):
	i = 0
	sum = 0
	m = len(data)
	while (i < m):
		sum = sum + ((estimate(t0, t1, data[i]['km']) - data[i]['price']) * data[i]['km'])
		i += 1
	tmp_t1 = learning_rate * (sum / m)
	return tmp_t1

def get_cost(w, b, data):
	i = 0
	m = len(data)
	sum = 0
	while (i < m):
		sum = sum + pow((data[i]['price'] - estimate(b, w, data[i]['km'])), 2)
		i += 1
	cost = sum / m
	return cost

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