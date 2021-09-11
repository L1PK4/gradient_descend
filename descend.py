import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

EPS = 1e-5



def grad_desc(A, x0, f):
	x = np.copy(x0)
	omega = np.matmul(A, x) - f
	x_arr = [np.copy(x)]
	while True:
		y = np.matmul(A, omega)
		r = np.dot(omega, omega)
		s = np.dot(y, omega)
		if s < EPS * EPS:
			return x_arr
		t = r / s
		x = x - t * omega
		x_arr.append(np.copy(x))
		omega = omega - t * y

def main():
	n = 5
	test_A = np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])
	test_f = np.array(list(map(sum, test_A)))
	
	print(f"Начальные условия:\n{test_A}\n\n{test_f}\n")
	x = grad_desc(test_A, np.zeros(n), test_f)
	print(f"Решение:\n{x[-1]}\n\nНайдено на {len(x)} итерации")
	if n != 2:
		return
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	X = np.arange(-2, 2, 0.25)
	Y = np.arange(-2, 2, 0.25)
	X, Y = np.meshgrid(X, Y)
	# print(test_A, test_f)
	def func(x : float, y : float):
		a, b, c, d = tuple(np.reshape(test_A, 4))
		f1, f2 = tuple(test_f)
		return a * x * x + (b + c) * x * y + d * y * y - 2 * (f1 * x + f2 * y)
	Z = func(X, Y)
	lineX, lineY = zip(*x)
	lineX = np.array(lineX)
	lineY = np.array(lineY)
	lineZ = func(lineX, lineY)
	lineZ += 0.4
	ax.scatter(lineX, lineY, lineZ,  c = 'black', s = 5, marker='o', alpha=1)
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
	plt.show()


if __name__ == "__main__":
	main()