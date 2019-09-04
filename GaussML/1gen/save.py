import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
x = np.arange(0,10)
y = np.sin(x)
plt.plot(x,y)
plt.show()
plt.savefig("hoge.png")

plt.plot(x,sgd[x],label = "SGD")
plt.plot(x,momentum[x],label = "Momentum")
plt.plot(x,adagrad[x],label = "AdaGrad")
plt.plot(x,adam[x],label = "Adam")
plt.legend()
plt.xlabel("iteration")
plt.ylabel("loss_function")
plt.show()
