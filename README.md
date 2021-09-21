# Digit classification on mnist_small dataset only using numpy
Multi-layer perceptron only using numpy to classify characters from 0 to 9

The neural network has two hidden layers. The first hidden layer has 400 neurons and the second has 200 neurons. I used relu as activation function and Mean squared error (MSE) as loss function. I used an bias in our network and momentum as gradient descent optimizer. With the hyperparameters learing rate alpha = 0.00395, momentum rate beta = 0.275 and batch size = 1 the network get an error of about 7 %.

I tested relu and sigmoid as activation functions, MSE and categorical crossentropy as loss functions. To find the right hyperparameters I used random search.


![error of NN](https://github.com/dannlos/neuralnetwork/blob/main/ErrorRateNN.jpg?raw=true)