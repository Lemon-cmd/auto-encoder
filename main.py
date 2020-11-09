from network import * 

net = network(3, "square-mean")
net.add_layer(10, "teq")
net.add_layer(30, "relu")
net.add_layer(10, "sigmoid")
net.add_layer(25, "tanh")
net.add_layer(max_length, "relu")

net.train(X[:len(X) * 3//4], Y[:len(X) * 3//4], 50000)
net.test(X[len(X)*3//4:], Y[len(X)*3//4:])