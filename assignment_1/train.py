import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from keras.datasets import fashion_mnist
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

import wandb
import argparse

################################################################################

def weights_initialization(num_neurons, initializer):
  """
  num_neurons = list of number of neurons at each layer starting from the input layer and ending at output layer
  initializer = 'random' or 'xavier'

  Returns: initialized weight matrices and bias vectors
  """
  mean=0
  std_dev=1

  W_matrices = []
  b_vectors = []

  for i in range(len(num_neurons)-1):
    rows = num_neurons[i+1]
    cols = num_neurons[i]

    if initializer == 'random':

      weight_matrix = np.random.normal(mean, std_dev, size=(rows, cols))

    elif initializer == 'Xavier':

      upper_bound = np.sqrt(6.0/(rows + cols))
      lower_bound = -1*upper_bound
      weight_matrix = np.random.uniform(low = lower_bound, high = upper_bound, size = (rows, cols))

    else:
      print('initializer invalid')


    bias_vector = np.zeros((rows,1))

    W_matrices.append(weight_matrix)
    b_vectors.append(bias_vector)


  return W_matrices, b_vectors

################################################################################

# ACTIVATION FUNCTIONS
def relu(x):
  """
  Rectified Linear Unit (ReLU) activation function
  """
  return np.maximum(0, x)

def sigmoid(x):
  """
  Sigmoid activation function
  """
  # x = np.float128(x)
  return 1 / (1 + np.exp(-x))

def tanh(x):
  """
  Hyperbolic tangent (tanh) activation function
  """
  # x = np.float128(x)
  return np.tanh(x)
def softmax(x):

  """
  Softmax function for output layer
  """
  # x = np.float128(x)
  return np.exp(x) / np.sum(np.exp(x), axis=0)

def activation_output(x, activation_function):
  """
  activation_function = 'ReLU', 'sigmoid', 'tanh'
  """
  if activation_function == 'ReLU':
    return relu(x)
  elif activation_function == 'sigmoid':
    return sigmoid(x)
  elif activation_function == 'tanh':
    return tanh(x)
  elif activation_function == 'softmax':
    return softmax(x)
  else:
    print('activation function invalid')

# DERIVATIVE OF ACTIVATION FUNCTION
def sigmoid_derivative(x):
  s = sigmoid(x)
  return s * (1 - s)

def tanh_derivative(x):
  t = tanh(x)
  return 1 - t**2

def relu_derivative(x):
  return 1*(x>0)

def activation_derivative(x, activation_function):
  """
  activation_function = 'ReLU', 'sigmoid', 'tanh'
  """
  if activation_function == 'ReLU':
    return relu_derivative(x)
  elif activation_function == 'sigmoid':
    return sigmoid_derivative(x)
  elif activation_function == 'tanh':
    return tanh_derivative(x)
  else:
    print('activation function invalid')

################################################################################

def layer_output_FP(x, weight_matrix, bias_vector, activation_function):
  pre_activation = np.add(np.matmul(weight_matrix, x), bias_vector)
  post_activation = activation_output(pre_activation, activation_function)
  return pre_activation, post_activation

def forward_propagation(ip_data, W_matrices, b_vectors, activation_functions):
  """
  forward propagation
  """

  layer_op = []
  layer_op.append(ip_data)

  layer_ip = []

  for i in range(len(W_matrices)):

    weight_matrix = W_matrices[i]
    bias_vector = b_vectors[i]

    activation_function = activation_functions[i]

    pre_activation, post_activation = layer_output_FP(layer_op[i], weight_matrix, bias_vector, activation_function)

    layer_op.append(post_activation)
    layer_ip.append(pre_activation)

  return layer_ip, layer_op

################################################################################

def back_propagation(W_matrices, b_vectors, y_true, layer_ip, layer_op, activation_functions, batch_size, w_d, loss_function):

  DWs = []
  Dbs = []
  for i in range(len(W_matrices)):
    k = len(W_matrices) - i

    if k == len(W_matrices):
      if loss_function == 'cross_entropy':
        Da = -np.add(y_true, -layer_op[k])
      elif loss_function == 'squared_error':
        Da = (layer_op[k] - y_true)*layer_op[k]*(1-layer_op[k])

      Dw = (np.matmul(Da, layer_op[k-1].T) + w_d*W_matrices[k-1])/batch_size
    else:

      Dh = np.matmul(W_matrices[k].T, Da)
      Dg = activation_derivative(layer_ip[k-1], activation_functions[k-1])
      Da = np.multiply(Dh, Dg)
      Dw = (np.matmul(Da, layer_op[k-1].T) + w_d*W_matrices[k-1])/batch_size
    Db = np.sum(Da, axis=1, keepdims=True)/batch_size

    DWs.append(Dw)
    Dbs.append(Db)

  return DWs, Dbs


################################################################################


def update_weights_gd(W_matrices, b_vectors, DWs, Dbs, learning_rate = 0.1):

  DWs.reverse()
  Dbs.reverse()

  for i in range(len(DWs)):

    W_matrices[i] = W_matrices[i] - learning_rate*DWs[i]
    b_vectors[i] = b_vectors[i] - learning_rate*Dbs[i]
  return W_matrices, b_vectors

def update_weights_momentum(W_matrices, b_vectors, DWs, Dbs, u_past_w, u_past_b, learning_rate = 0.1, beta = 0.5):
  DWs.reverse()
  Dbs.reverse()
  u_w = u_past_w
  u_b = u_past_b
  for i in range(len(DWs)):

    u_w[i] = beta*u_past_w[i] + DWs[i]
    u_b[i] = beta*u_past_b[i] + Dbs[i]

    W_matrices[i] = W_matrices[i] - learning_rate*u_w[i]
    b_vectors[i] = b_vectors[i] - learning_rate*u_b[i]

  return W_matrices, b_vectors, u_w, u_b

def update_weights_adagrad(W_matrices, b_vectors, DWs, Dbs, u_past_w, u_past_b, learning_rate = 0.1):
  DWs.reverse()
  Dbs.reverse()

  u_w = u_past_w
  u_b = u_past_b
  eps = 1e-8
  for i in range(len(DWs)):
    u_w[i] = u_past_w[i] + DWs[i]**2
    u_b[i] = u_past_b[i] + Dbs[i]**2

    W_matrices[i] = W_matrices[i] - learning_rate*DWs[i]/(np.sqrt(u_w[i]) + eps)
    b_vectors[i] = b_vectors[i] - learning_rate*Dbs[i]/(np.sqrt(u_b[i]) + eps)

  return W_matrices, b_vectors, u_w, u_b

def update_weights_rmsprop(W_matrices, b_vectors, DWs, Dbs, u_past_w, u_past_b, learning_rate = 0.1, beta = 0.5):
  DWs.reverse()
  Dbs.reverse()

  u_w = u_past_w
  u_b = u_past_b
  eps = 1e-8
  for i in range(len(DWs)):
    u_w[i] = beta*u_past_w[i] + (1-beta)*DWs[i]**2
    u_b[i] = beta*u_past_b[i] + (1-beta)*Dbs[i]**2

    W_matrices[i] = W_matrices[i] - learning_rate*DWs[i]/(np.sqrt(u_w[i]) + eps)
    b_vectors[i] = b_vectors[i] - learning_rate*Dbs[i]/(np.sqrt(u_b[i]) + eps)

  return W_matrices, b_vectors, u_w, u_b

def update_weights_adam(W_matrices, b_vectors, DWs, Dbs, mw_past, mb_past, vw_past, vb_past, t, learning_rate = 0.1, beta1 = 0.5, beta2 =0.5):
  DWs.reverse()
  Dbs.reverse()
  mw = mw_past
  mb = mb_past
  vw = vw_past
  vb = vb_past
  eps = 1e-8

  for i in range(len(DWs)):
    mw[i] = beta1*mw_past[i] + (1-beta1)*DWs[i]
    mb[i] = beta1*mb_past[i] + (1-beta1)*Dbs[i]

    mw_cap = mw[i]/(1 - beta1**t)
    mb_cap = mb[i]/(1 - beta1**t)

    vw[i] = beta2*vw_past[i] + (1-beta2)*DWs[i]**2
    vb[i] = beta2*vb_past[i] + (1-beta2)*Dbs[i]**2
    vw_cap = vw[i]/(1 - beta2**t)
    vb_cap = vb[i]/(1 - beta2**t)

    W_matrices[i] = W_matrices[i] - learning_rate*mw_cap/(np.sqrt(vw_cap) + eps)
    b_vectors[i] = b_vectors[i] - learning_rate*mb_cap/(np.sqrt(vb_cap) + eps)

  return W_matrices, b_vectors, mw, mb, vw, vb

def update_weights_nadam(W_matrices, b_vectors, DWs, Dbs, mw_past, mb_past, vw_past, vb_past,t,  learning_rate = 0.1, beta1 = 0.5, beta2 =0.5):
  DWs.reverse()
  Dbs.reverse()
  mw = mw_past
  mb = mb_past
  vw = vw_past
  vb = vb_past
  eps = 1e-8

  for i in range(len(DWs)):
    mw[i] = beta1*mw_past[i] + (1-beta1)*DWs[i]
    mb[i] = beta1*mb_past[i] + (1-beta1)*Dbs[i]

    mw_cap = mw[i]/(1 - beta1**(t+1))
    mb_cap = mb[i]/(1 - beta1**(t+1))

    vw[i] = beta2*vw_past[i] + (1-beta2)*DWs[i]**2
    vb[i] = beta2*vb_past[i] + (1-beta2)*Dbs[i]**2
    vw_cap = vw[i]/(1 - beta2**(t+1))
    vb_cap = vb[i]/(1 - beta2**(t+1))

    W_matrices[i] = W_matrices[i] - learning_rate*(beta1*mw_cap + ((1-beta1)/(1 - beta1**(t+1)))*DWs[i])/(np.sqrt(vw_cap) + eps)
    b_vectors[i] = b_vectors[i] - learning_rate*(beta1*mb_cap + ((1-beta1)/(1 - beta1**(t+1)))*Dbs[i])/(np.sqrt(vb_cap) + eps)

  return W_matrices, b_vectors, mw, mb, vw, vb

def look_ahead_nag(W_s, b_s, u_past_w, u_past_b, beta = 0.5):
  for i in range(len(W_s)):
    W_s[i] = W_s[i] - beta*u_past_w[i]
    b_s[i] = b_s[i] - beta*u_past_b[i]
  return W_s, b_s

################################################################################


def one_hot_encode(integers, num_classes=None):
  if num_classes is None:
      num_classes = np.max(integers) + 1
  return np.eye(num_classes)[integers]

def cross_entropy_loss(y_true, y_pred, batch_size):
  # Clip the predicted probabilities to avoid numerical instability
  y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
  loss_value = np.sum(np.sum(y_true*np.log(y_pred), axis=0))/batch_size
  return loss_value*(-1)

def squared_error_loss(y_true, y_pred, batch_size):
  # Clip the predicted probabilities to avoid numerical instability
  loss_value = 0.5*np.sum(la.norm(y_true-y_pred, axis=0)**2)/batch_size
  return loss_value

def accuracy(y_true, y_pred, batch_size):
  n_correct = 0
  for i in range(0, batch_size, 1) :
    if y_true[:,i].argmax() == y_pred[:,i].argmax() :
      n_correct += 1
  return 100 * n_correct / batch_size

################################################################################

def load_split_dataset_fasion_mnist(test_ratio=0.3):
  # Load Fashion MNIST dataset
  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

  # Split the training set into training and validation sets
  X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size=test_ratio, random_state=42)

  data_size = X_train.shape[0]
  X_train = (X_train.reshape(data_size, -1).T)/255
  Y_train = one_hot_encode(Y_train, 10).T

  data_size = X_val.shape[0]
  X_val = (X_val.reshape(data_size, -1).T)/255
  Y_val = one_hot_encode(Y_val, 10).T

  data_size = test_images.shape[0]
  X_test = (test_images.reshape(data_size, -1).T)/255
  Y_test = one_hot_encode(test_labels, 10).T

  return X_train, Y_train, X_val, Y_val, X_test, Y_test

def load_split_dataset_mnist(test_ratio=0.3):
  # Load Fashion MNIST dataset
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

  # Split the training set into training and validation sets
  X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size=test_ratio, random_state=42)

  data_size = X_train.shape[0]
  X_train = (X_train.reshape(data_size, -1).T)/255
  Y_train = one_hot_encode(Y_train, 10).T

  data_size = X_val.shape[0]
  X_val = (X_val.reshape(data_size, -1).T)/255
  Y_val = one_hot_encode(Y_val, 10).T

  data_size = test_images.shape[0]
  X_test = (test_images.reshape(data_size, -1).T)/255
  Y_test = one_hot_encode(test_labels, 10).T

  return X_train, Y_train, X_val, Y_val, X_test, Y_test

################################################################################


def train_model(X_train,Y_train, X_test, Y_test, epoch=1,batch_size=25, num_neurons_hidden = [10], activation_functions = ['sigmoid'], loss_function = 'cross_entropy',
                weights_init_type='random', optimizer = 'sgd', learning_rate = 0.1, opti_beta = [0.5, 0.5], w_d = 0, plot_acc_loss = False):

  """
  X has shape (number of features, number of samples in train data set)
  Y has shape (number of classes, number of samples in train data set)

  num_neurons_hidden = list of number of neurons at each hidden layer

  """
  num_ip_neurons = X_train.shape[0]
  num_op_neurons = Y_train.shape[0]
  num_neurons = [num_ip_neurons] + num_neurons_hidden + [num_op_neurons]
  activation_functions = activation_functions + ['softmax']

  W_s, b_s = weights_initialization(num_neurons, weights_init_type)
  print('Hyper parameters: \n')
  print("Weight initialization type : ", weights_init_type)
  print("Optimizer : ", optimizer)
  print("Learning rate (initial): ", learning_rate)
  print("Batch size: ", batch_size)
  print("-------------------")
  print("Architecture Description:\n")

  for i in range(len(num_neurons)-1):
    print("Layer: ", i+1, " ; number of neurons: ", num_neurons[i+1], " ; activation function: ", activation_functions[i])
    print("Weight matrix dimention", W_s[i].shape, "Bias vector dimention", b_s[i].shape)
    print("----------------")

  num_batches = np.floor(X_train.shape[1]/batch_size)
  print(num_batches)


  if optimizer == 'momentum':
    u_past_w = [x * 0 for x in W_s]
    u_past_b = [x * 0 for x in b_s]

  elif optimizer == 'nag':
    u_past_w = [x * 0 for x in W_s]
    u_past_b = [x * 0 for x in b_s]

  elif optimizer == 'rmsprop':
    u_past_w = [x * 0 for x in W_s]
    u_past_b = [x * 0 for x in b_s]

  elif optimizer == 'adagrad':
    u_past_w = [x * 0 for x in W_s]
    u_past_b = [x * 0 for x in b_s]

  elif optimizer == 'adam':
    mw_past = [x * 0 for x in W_s]
    mb_past = [x * 0 for x in b_s]
    vw_past = [x * 0 for x in W_s]
    vb_past = [x * 0 for x in b_s]
    t = 1

  elif optimizer == 'nadam':
    mw_past = [x * 0 for x in W_s]
    mb_past = [x * 0 for x in b_s]
    vw_past = [x * 0 for x in W_s]
    vb_past = [x * 0 for x in b_s]
    t = 1

  print('\n Start of training')

  ip_all, op_all = forward_propagation(X_train, W_s, b_s, activation_functions)
  ce_loss_tr = cross_entropy_loss(Y_train, op_all[-1], X_train.shape[1])
  se_loss_tr = squared_error_loss(Y_train, op_all[-1], X_train.shape[1])
  acc_tr = accuracy(Y_train, op_all[-1], Y_train.shape[1])

  ip_all, op_all = forward_propagation(X_test, W_s, b_s, activation_functions)
  ce_loss_ts = cross_entropy_loss(Y_test, op_all[-1], X_test.shape[1])
  se_loss_ts = squared_error_loss(Y_test, op_all[-1], X_test.shape[1])
  acc_ts = accuracy(Y_test, op_all[-1], Y_test.shape[1])

  print("CE Training Loss: ", ce_loss_tr, "SE Training Loss: ", se_loss_tr,"Training Accuracy: ", acc_tr, "CE Test Loss: ",
        ce_loss_ts, "SE Test Loss: ", se_loss_ts ,  "Testing Accuracy: ", acc_ts)

  wandb.log({'tr_loss_CE' : ce_loss_tr, 'tr_loss_SE' : se_loss_tr, 'tr_accuracy' : acc_tr, 'val_loss_CE' : ce_loss_ts, 'val_loss_SE' : se_loss_ts, 'val_accuracy' : acc_ts})

  train_loss = np.array([ce_loss_tr])
  train_acc = np.array([acc_tr])

  val_loss = np.array([ce_loss_ts])
  val_acc = np.array([acc_ts])

  for i in range(epoch):
    print('Epoch: ', i+1)

    for j in tqdm(range(int(num_batches))):
      batch_X = X_train[:,j*batch_size:(j+1)*batch_size]
      batch_Y = Y_train[:,j*batch_size:(j+1)*batch_size]


      if optimizer == 'sgd':
        ip, op = forward_propagation(batch_X, W_s, b_s, activation_functions)
        DWs, Dbs = back_propagation(W_s, b_s, batch_Y, ip, op, activation_functions, batch_size, w_d, loss_function)
        W_s, b_s = update_weights_gd(W_s, b_s, DWs, Dbs, learning_rate)

      elif optimizer == 'momentum':
        ip, op = forward_propagation(batch_X, W_s, b_s, activation_functions)
        DWs, Dbs = back_propagation(W_s, b_s, batch_Y, ip, op, activation_functions, batch_size, w_d, loss_function)
        W_s, b_s, u_past_w, u_past_b  = update_weights_momentum(W_s, b_s, DWs, Dbs, u_past_w, u_past_b, learning_rate, opti_beta[0])

      elif optimizer == 'adagrad':
        ip, op = forward_propagation(batch_X, W_s, b_s, activation_functions)
        DWs, Dbs = back_propagation(W_s, b_s, batch_Y, ip, op, activation_functions, batch_size, w_d, loss_function)
        W_s, b_s, u_past_w, u_past_b  = update_weights_adagrad(W_s, b_s, DWs, Dbs, u_past_w, u_past_b, learning_rate)

      elif optimizer == 'rmsprop':
        ip, op = forward_propagation(batch_X, W_s, b_s, activation_functions)
        DWs, Dbs = back_propagation(W_s, b_s, batch_Y, ip, op, activation_functions, batch_size, w_d, loss_function)
        W_s, b_s, u_past_w, u_past_b  = update_weights_rmsprop(W_s, b_s, DWs, Dbs, u_past_w, u_past_b, learning_rate, opti_beta[0])

      elif optimizer == 'adam':
        ip, op = forward_propagation(batch_X, W_s, b_s, activation_functions)
        DWs, Dbs = back_propagation(W_s, b_s, batch_Y, ip, op, activation_functions, batch_size, w_d, loss_function)
        W_s, b_s, mw_past, mb_past, vw_past, vb_past = update_weights_adam(W_s, b_s, DWs, Dbs, mw_past, mb_past, vw_past, vb_past, t, learning_rate, opti_beta[0], opti_beta[1])
        t =t +1

      elif optimizer == 'nadam':
        ip, op = forward_propagation(batch_X, W_s, b_s, activation_functions)
        DWs, Dbs = back_propagation(W_s, b_s, batch_Y, ip, op, activation_functions, batch_size, w_d, loss_function)
        W_s, b_s, mw_past, mb_past, vw_past, vb_past = update_weights_nadam(W_s, b_s, DWs, Dbs, mw_past, mb_past, vw_past, vb_past, t, learning_rate, opti_beta[0], opti_beta[1])
        t =t +1
      elif optimizer == 'nag':
        PWs, Pbs = look_ahead_nag(W_s, b_s, u_past_w, u_past_b, opti_beta[0])
        ip, op = forward_propagation(batch_X, W_s, b_s, activation_functions)
        DWs, Dbs = back_propagation(PWs, Pbs, batch_Y, ip, op, activation_functions, batch_size, w_d, loss_function)
        W_s, b_s, u_past_w, u_past_b  = update_weights_momentum(W_s, b_s, DWs, Dbs, u_past_w, u_past_b, learning_rate, opti_beta[0])


    ip_all, op_all = forward_propagation(X_train, W_s, b_s, activation_functions)
    ce_loss_tr = cross_entropy_loss(Y_train, op_all[-1], X_train.shape[1])
    se_loss_tr = squared_error_loss(Y_train, op_all[-1], X_train.shape[1])
    acc_tr = accuracy(Y_train, op_all[-1], Y_train.shape[1])

    ip_all, op_all = forward_propagation(X_test, W_s, b_s, activation_functions)
    ce_loss_ts = cross_entropy_loss(Y_test, op_all[-1], X_test.shape[1])
    se_loss_ts = squared_error_loss(Y_test, op_all[-1], X_test.shape[1])
    acc_ts = accuracy(Y_test, op_all[-1], Y_test.shape[1])

    print("CE Training Loss: ", ce_loss_tr, "SE Training Loss: ", se_loss_tr,"Training Accuracy: ", acc_tr, "CE Test Loss: ",
          ce_loss_ts, "SE Test Loss: ", se_loss_ts ,  "Testing Accuracy: ", acc_ts)

    train_loss = np.append(train_loss, [ce_loss_tr])
    train_acc = np.append(train_acc, [acc_tr])

    val_loss = np.append(val_loss, [ce_loss_ts])
    val_acc = np.append(val_acc, [acc_ts])

    wandb.log({'tr_loss_CE' : ce_loss_tr, 'tr_loss_SE' : se_loss_tr, 'tr_accuracy' : acc_tr, 'val_loss_CE' : ce_loss_ts, 'val_loss_SE' : se_loss_ts, 'val_accuracy' : acc_ts})

  if plot_acc_loss == True:

    fig, ax = plt.subplots()  # Create a figure and axes object
    ax.plot(np.arange(0, epoch + 1, 1), train_acc, color='r', label='training')
    ax.plot(np.arange(0, epoch + 1, 1), val_acc, color='g', label='validation')
    ax.set_title("Accuracy")  # Set title on the axes object
    ax.legend()
    plt.grid()
    plt.show()

    fig, ax = plt.subplots()  # Create a figure and axes object for the second plot
    ax.plot(np.arange(0, epoch + 1, 1), train_loss, color='r', label='training')
    ax.plot(np.arange(0, epoch + 1, 1), val_loss, color='g', label='validation')
    ax.set_title("Loss")  # Set title on the axes object
    ax.legend()
    plt.grid()
    plt.show()


################################################################################


def main(args):
    # Start a W&B Run
    run = wandb.init(config=args)
    
    # Access values from config dictionary and store them into variables for readability

    dataset=wandb.config.dataset
    epochs=wandb.config.epochs
    batch_size=wandb.config.batch_size
    loss=wandb.config.loss
    optimizer=wandb.config.optimizer
    learning_rate=wandb.config.learning_rate
    momentum=wandb.config.momentum
    beta=wandb.config.beta
    beta1=wandb.config.beta1
    beta2=wandb.config.beta2
    epsilon=wandb.config.epsilon
    lambda_val=wandb.config.weight_decay
    init=wandb.config.weight_init
    num_hidden_layers=wandb.config.num_layers
    num_neurons=wandb.config.hidden_size
    activation_function=wandb.config.activation

    neuros_num = []
    act_func = []
    for i in range(num_hidden_layers):
      neuros_num.append(num_neurons)
      act_func.append(activation_function)

    if beta != None:
      opti_beta = [beta]
    elif beta1 != None:
      opti_beta = [beta1, beta2]

    # Dataset 
    if dataset=='fashion_mnist':
       X_train, Y_train, X_val, Y_val, X_test, Y_test = load_split_dataset_fasion_mnist()
    elif dataset=='mnist':
       X_train, Y_train, X_val, Y_val, X_test, Y_test = load_split_dataset_mnist()

    wandb.run.name = "e_{}_hl_{}_hs_{}_lr_{}_opt_{}_bs_{}_init_{}_ac_{}_l2_{}".format(epochs, num_hidden_layers, batch_size, learning_rate, optimizer, 
                                                                                      batch_size, init, activation_function, lambda_val)

    train_model(X_train, Y_train, X_val, Y_val, epoch=epochs, batch_size= batch_size, num_neurons_hidden = neuros_num, activation_functions = act_func, loss_function = loss,
                weights_init_type=init, optimizer = optimizer, learning_rate = learning_rate, opti_beta = opti_beta, w_d = lambda_val)



################################################################################
if __name__ == "__main__":
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument('-wp',"--wandb_project",type=str,default="Assignment-1")
  parser.add_argument("-we","--wandb_entity",type=str,default="shashwat_mm19b053")
  parser.add_argument("-d","--dataset",type=str,default='fashion_mnist',help="Datasets: fashion_mnist or mnist")
  parser.add_argument("-e","--epochs",type=int,default=30,help='Number of epochs')
  parser.add_argument("-b","--batch_size",type=int,default=32,help='Batch Size')
  parser.add_argument("-l","--loss",type=str,default="cross_entropy",help="Cross-entropy loss/ Mean Squared Error loss")
  parser.add_argument("-o","--optimizer",type=str,default="adam",help="sgd/momentum/nesterov/rmsprop/adam/nadam")
  parser.add_argument("-lr","--learning_rate",type=float,default=0.0001,help="Learning Rate")
  parser.add_argument("-m","--momentum",type=float,default=0.9,help="Momentum used by nesterov & momentum gd")
  parser.add_argument("-beta","--beta",type=float,default=0.9,help="Beta used by rmsprop")
  parser.add_argument("-beta1","--beta1",type=float,default=0.9,help="Beta1 used by adam & nadam")
  parser.add_argument("-beta2","--beta2",type=float,default=0.999,help="Beta2 used by adam & nadam")
  parser.add_argument("-eps","--epsilon",type=float,default=0.000001)
  parser.add_argument("-w_d","--weight_decay",type=float,default=0,help="L2 Regularizer")
  parser.add_argument("-w_i","--weight_init",type=str,default="Xavier",help="Xavier or Random Initialization")   
  parser.add_argument("-nhl","--num_layers",type=int,default=3,help="Number of hidden layers in the network")
  parser.add_argument("-sz","--hidden_size",type=int,default=128,help="Number of neurons in each hidden layer")
  parser.add_argument("-a","--activation",type=str,default='tanh',help="Activation Function")


args = parser.parse_args()