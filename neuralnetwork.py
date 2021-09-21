"""
Implementing Multi.layer perseptron with numpy
@author: Danilo Walenta (danilowalenta.com)
"""
#%%
# imports
import numpy as np
import matplotlib.pyplot as plt

#%%
# load data
mnist_small_train_input = np.genfromtxt('dataSets/mnist_small_train_in.txt', delimiter=',')
mnist_small_train_output = np.genfromtxt('dataSets/mnist_small_train_out.txt', delimiter=',')
mnist_small_test_input = np.genfromtxt('dataSets/mnist_small_test_in.txt', delimiter=',')
mnist_small_test_output = np.genfromtxt('dataSets/mnist_small_test_out.txt', delimiter=',')
print(mnist_small_train_input.shape)
print(mnist_small_train_input)
print(mnist_small_train_output.shape)
print(mnist_small_train_output)

print(mnist_small_test_input.shape)
print(mnist_small_test_input)
print(mnist_small_test_output.shape)
print(mnist_small_test_output)

#%%
# one hot encoding
def one_hot_encoding(input):
    result = np.zeros((len(input), 10))
    for i in range(len(input)):
        result[i][int(input[i])] = 1
    return result

mnist_small_train_output_prep = one_hot_encoding(mnist_small_train_output)
mnist_small_test_output_prep = one_hot_encoding(mnist_small_test_output)

#%%
# shuffle data
def shuffle_data(x, y):
    p = np.random.permutation(len(x))
    return x[p], y[p]

train_input_shuffle, train_output_shuffle = shuffle_data(mnist_small_train_input,mnist_small_train_output)
train_input_shuffle.shape
train_output_shuffle_prep = one_hot_encoding(train_output_shuffle)
train_output_shuffle_prep

#%%
# define loss and activation functions
# loss function:
def loss_simple(y_hat, y_star):
    result = 0.1 * (y_hat - y_star) ** 2  # (1/len(y_hat))
    return np.sum(result)

def derivertive_loss_simple(y_hat, y_star):
    result = y_hat - y_star
    return result

def categorical_crossentropy(y_hat, y_star):
    result = y_star * np.log(y_hat)
    return - np.sum(result)

def derivertive_categorical_crossentropy(y_hat, y_star):
    return y_hat - y_star

# activation function
def relu_prep(x):
    if x <= 0:
        return 0  # .1 *x
    else:
        return x

def relu(x):
    return np.array(list(map(lambda input: relu_prep(input), x)))


def derivertive_relu_prep(x):
    if x <= 0:
        return 0  # .1
    else:
        return 1


def derivertive_relu(x):
    # print(x.shape)
    result = np.array(list(map(lambda input: derivertive_relu_prep(input), x)))[np.newaxis].T
    # print(result.shape)
    return result


def sigmoid_prep(x):
    return 1 / (1 + np.exp(-x))


def sigmoid(x):
    # print(map(lambda input: relu_prep(input), x))
    return np.array(list(map(lambda input: sigmoid_prep(input), x)))


def derivertive_sigmoid(x):
    return x * (1 - x)

#%%
# initiate weights
number_of_inputs = 28 * 28
mu = 0.0
sigma = 1
hiddenlayer_one_weights = np.random.normal(mu,sigma,(785,400)) /np.sqrt(785/2)
hiddenlayer_two_weights = np.random.normal(mu,sigma,(401,200)) / np.sqrt(401/2)
outputlayer_weights = np.random.normal(mu,sigma,(201,10)) /np.sqrt(251/2)

w = [hiddenlayer_one_weights, hiddenlayer_two_weights, outputlayer_weights]

#%%
# neural network and backpropagation

# return the output of the to_layer
def fully_connection(from_layer, weights, activation_function):
    from_layer = np.append(from_layer, [1])
    result = np.dot(from_layer, weights)
    return activation_function(result)


def forward_check(x_input, w):
    output = [x_input]
    for i in range(len(w)):
        # if i == (len(w)-1):
        #    tmp = fully_connection(output[i], w[i], sigmoid)
        # else:
        tmp = fully_connection(output[i], w[i], relu)  # sigmoid)#(lambda x: relu(x)+sigmoid(x)))
        output.append(tmp)
    return output, w


# returns the new weights and the backpropagation until this layer
# w_new = w_old - learning_rate * derivertive_until_current * output_from_layer
def backpropagation_fully_connection(output_current_layer,
                                     output_from_layer, old_weights, learning_rate,
                                     derivertive_until_current, momentum_rate, momentum):
    momentum = momentum_rate * momentum - learning_rate * np.dot(np.append(output_from_layer, [1])[np.newaxis].T,
                                                                 derivertive_until_current.T)
    new_weights = old_weights + momentum
    derivertive_until_current = derivertive_until_current * derivertive_relu(output_current_layer)
    new_derivertive_until_current = np.dot(old_weights[:-1], derivertive_until_current)
    return new_weights, new_derivertive_until_current, momentum


def back_propagation(y_hat, y_star, list_of_weights, list_of_outputs,
                     alpha, deriv_loss_func, beta, momentum):
    deriv_loss = deriv_loss_func(y_hat, y_star) * derivertive_relu(y_hat)  # loss function
    new_weights = []
    temp_derivertive = deriv_loss
    new_momentum = []
    for i in range((len(list_of_weights) - 1), -1, -1):
        temp_weights, temp_derivertive, current_momentum = backpropagation_fully_connection(
            list_of_outputs[i + 1][np.newaxis].T,
            list_of_outputs[i][np.newaxis].T,
            list_of_weights[i], alpha,
            temp_derivertive, beta, momentum[i])
        new_weights.insert(0, temp_weights)
        new_momentum.insert(0, current_momentum)
    return new_weights, new_momentum


def run_backpropagation(x_input_array, y_output_array, loss_func,
                        deriv_loss_func, alpha, beta, number_of_epochs,
                        batch_size, weights, show_progress=True):
    number_of_layers = len(weights)
    momentum = np.zeros(number_of_layers)
    error = []
    for y in range(number_of_epochs):
        average_loss = []
        outputs = None
        y_stars = None
        for i in range(len(x_input_array)):
            output, weights = forward_check(x_input_array[i], weights)
            if outputs == None:
                outputs = output
                y_stars = y_output_array[i]
            else:
                outputs = np.add(outputs, output)
                y_stars = np.add(y_stars, y_output_array[i])

            current_loss = loss_func(output[number_of_layers][np.newaxis].T, y_output_array[i][np.newaxis].T)
            average_loss.append(current_loss)

            if (i + 1) % batch_size == 0 or i == len(x_input_array) - 1:
                if batch_size > 1:
                    outputs = outputs / batch_size
                    y_stars = y_stars / batch_size
                weights, momentum = back_propagation(outputs[number_of_layers][np.newaxis].T,
                                                     y_stars[np.newaxis].T, weights, outputs,
                                                     alpha, deriv_loss_func, beta, momentum)
                outputs = None
                y_stars = None
            if show_progress:
                if (i + 1) % 100 == 0:
                    print("-", end="")  # progress bar
        average_loss = sum(average_loss) / len(average_loss)
        if show_progress:
            print("")
            print("The train loss in Epoch {} is: {}".format(y + 1, average_loss))
        test_loss = []
        true_positive = 0
        for i in range(len(mnist_small_test_input)):
            output, weights = forward_check(mnist_small_test_input[i], weights)
            test_loss.append(loss_func(output[number_of_layers][np.newaxis].T,
                                       mnist_small_test_output[i][np.newaxis].T))
            if np.argmax(output[number_of_layers]) == mnist_small_test_output[i]:
                true_positive += 1
        test_loss = sum(test_loss) / len(test_loss)
        if show_progress:
            print("The test loss in Epoch {} is: {}".format(y + 1, test_loss))
            print("The Accuracy is: {}".format(true_positive / len(mnist_small_test_input)))
        error.append((1 - (true_positive / len(mnist_small_test_input))) * 100)
    return weights, error


#%%
# random search and run neural network
x_input_array = train_input_shuffle
y_output_array = train_output_shuffle_prep
loss_func = loss_simple#categorical_crossentropy
deriv_loss_func = derivertive_loss_simple#derivertive_categorical_crossentropy
number_of_epochs = 20
weights = w
number_of_random_search_iterations = 500
lr = 0.00395
mr = 0.275
bs = 1
random_search = False
errors = []
combination = []
if random_search:
    number_of_epochs = 20
    for i in range(number_of_random_search_iterations):
        lr = round(np.random.rand(1)[0],2) * 0.005
        mr = round(np.random.rand(1)[0],2) *0.5
        bs = round((round(np.random.rand(1)[0],2) * 20),0)
        w1, error = run_backpropagation(x_input_array, y_output_array, loss_func, deriv_loss_func, lr, mr, number_of_epochs, bs, weights, False) # alpha 0.001
        errors.append(min(error))
        combination_text = "Learning rate: {}, Momentum_rate: {}, batch size: {}, Min error: ".format(lr, mr, bs)
        print(combination_text, min(error))
        combination.append(combination_text)
    print("")
    print(combination[np.argmin(errors)], errors[np.argmin(errors)])

if not random_search:
    w1, error = run_backpropagation(x_input_array, y_output_array, loss_func, deriv_loss_func, lr, mr, number_of_epochs, bs, weights, True) # alpha 0.001
    x_range = np.arange(1, number_of_epochs + 1 ,1)
    plt.plot(x_range,error)
    plt.xlabel("Epoch")
    plt.ylabel("Error in %")
    plt.savefig("ErrorRateNN.jpg", dpi=300)
    plt.show()