#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import time

tf.enable_eager_execution()
tf.executing_eagerly()


# In[2]:


train, val, test = input_data.read_data_sets('data/fashion',one_hot=True)


# In[3]:


## Permuted MNIST

# Generate the tasks specifications as a list of random permutations of the input pixels.
num_tasks_to_run = 10
task_permutation = []
for task in range(num_tasks_to_run):
  np.random.seed(5+task)
  task_permutation.append( np.random.permutation(784) )
each_task_data = np.asarray([train.images[:,task_permutation[i]] for i in range(num_tasks_to_run)])
each_task_test = np.asarray([test.images[:,task_permutation[i]] for i in range(num_tasks_to_run)])
  


# # L1 Loss

# In[4]:


class MLP(object):
  def __init__(self, size_input, size_hidden_1, size_hidden_2 , size_output, device=None):
    """
    size_input: int, size of input layer
    size_hidden: int, size of hidden layer
    size_output: int, size of output layer
    device: str or None, either 'cpu' or 'gpu' or None. If None, the device to be used will be decided automatically during Eager Execution
    """
    self.size_input, self.size_hidden_1,self.size_hidden_2, self.size_output, self.device =    size_input, size_hidden_1,size_hidden_2, size_output, device
    
    # Initialize weights between input layer and hidden layer
    self.W1 = tf.Variable(tf.random_normal([self.size_input, self.size_hidden_1]))
    # Initialize biases for hidden layer
    self.b1 = tf.Variable(tf.random_normal([1, self.size_hidden_1]))
     # Initialize weights between hidden layer and output layer
    self.W2 = tf.Variable(tf.random_normal([self.size_hidden_1, self.size_hidden_2]))
    # Initialize biases for output layer
    self.b2 = tf.Variable(tf.random_normal([1, self.size_hidden_2]))
    
    self.W3 = tf.Variable(tf.random_normal([self.size_hidden_2, self.size_output]))
    # Initialize biases for output layer
    self.b3 = tf.Variable(tf.random_normal([1, self.size_output]))
    
    # Define variables to be updated during backpropagation
    self.variables = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]
    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)

    
  def forward(self, X):
    """
    forward pass
    X: Tensor, inputs
    """
    if self.device is not None:
      with tf.device('gpu:0' if self.device=='gpu' else 'cpu'):
        self.y = self.compute_output(X)
    else:
      self.y = self.compute_output(X)
      
    return self.y
  
  def loss(self, y_pred, y_true):
    '''
    y_pred - Tensor of shape (batch_size, size_output)
    y_true - Tensor of shape (batch_size, size_output)
    '''
    
    y_true_tf = tf.cast(y_true, dtype=tf.float32)
    y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
    
    return tf.reduce_mean(tf.reduce_sum(tf.math.abs((y_true_tf-y_pred_tf))))
#     return tf.losses.softmax_cross_entropy(, )
#   loss_op = 
  def backward(self, X_train, y_train):
    """
    backward pass
    """
#     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    with tf.GradientTape() as tape:
      predicted = self.forward(X_train)
      current_loss = self.loss(predicted, y_train)
    grads = tape.gradient(current_loss, self.variables)
    self.optimizer.apply_gradients(zip(grads, self.variables),
                              global_step=tf.train.get_or_create_global_step())
        
        
  def compute_output(self, X):
    """
    Custom method to obtain output tensor during forward pass
    """
    # Cast X to float32
    X_tf = tf.cast(X, dtype=tf.float32)
    #Remember to normalize your dataset before moving forward
    # Compute values in hidden layer
    what1 = tf.matmul(X_tf, self.W1) + self.b1
    hhat1 = tf.nn.relu(what1)
    what2 = tf.matmul(hhat1, self.W2) + self.b2
    hhat2 = tf.nn.relu(what2)
    # Compute output
    output = tf.matmul(hhat2, self.W3) + self.b3
    
    output = tf.nn.softmax(output)
    #Now consider two things , First look at inbuild loss functions if they work with softmax or not and then change this
    #Second add tf.Softmax(output) and then return this variable
    return output


# In[5]:


num_epochs_per_task = 20
n_epochs = 20
minibatch_size = 100
batch_size = 100
num_tasks_to_run = 10
size_input = 784
size_hidden_1 = 256
size_hidden_2 = 256
size_output = 10

R_matrix_L1=list()
#Initialize model using CPU
mlp_on_cpu = MLP(size_input, size_hidden_1,size_hidden_2, size_output, device='cpu')
time_start = time.time()
for task in range(num_tasks_to_run):
  input_kir = each_task_data[task]
  label_kir = train.labels
  input_kir = np.asarray(input_kir,dtype='float32')


#   for i in range(1):
#     if i == task:
#       continue

#     input_kir = pd.read_csv(f"task{i}.csv",header = None) 
#     label_kir = train.labels
#     input_kir = np.asarray(input_kir,dtype='float32')

  if task == 0:
    n_epochs = 50
  else:
    n_epochs = 20
  for epoch in range(n_epochs):
    loss_total = tf.Variable(0, dtype=tf.float32)
    for index, offset in enumerate(range(0, 55000, batch_size)):
      img , label=input_kir[offset: offset + batch_size], label_kir[offset: offset + batch_size]
      
      preds = mlp_on_cpu.forward(img)
      loss_total = loss_total + mlp_on_cpu.loss(preds, label)
  #     print(mlp_on_cpu.loss(preds, label))
      mlp_on_cpu.backward(img, label)  
    print('Number of Epoch = {} - Average MSE:= {:.4f}'.format(epoch + 1, loss_total.numpy() / train.images.shape[0]))
  row_of_R=list()
  for tsk in range(num_tasks_to_run):
    input_kir = each_task_test[tsk]
    label_kir = test.labels
    input_kir = np.asarray(input_kir, dtype = 'float32')
    preds = mlp_on_cpu.forward(input_kir)
    correct_preds = tf.equal(tf.argmax(tf.nn.softmax(preds), 1), tf.argmax(label_kir, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
    print(f'Accuracy of task {tsk+1} is {accuracy}')    
    row_of_R.append(accuracy)
  R_matrix_L1.append(row_of_R)
#   for i in range(num_tasks_to_run):
#     if i == task:
#       continue
#     loss_total = tf.Variable(0, dtype=tf.float32)
#     input_data = pd.read_csv(f"task{task}.csv",header = None) 
#     label_value=train.labels
#     preds = (mlp_on_cpu.forward(np.asarray(input_data,dtype='float32')/255))
#     loss_total = mlp_on_cpu.loss(preds, label)
#     print(f'for task {i} accuracy is {loss_total}')



  time_taken = time.time() - time_start

  print('\nTotal time taken (in seconds): {:.2f}'.format(time_taken))
  #For per epoch_time = Total_Time / Number_of_epochs


# In[6]:


ACC=list()
BWT=list()
R_matrix = np.asarray(R_matrix_L1)
for i in range(num_tasks_to_run):
  ACC.append((1/(i+1))*np.sum(R_matrix[i,:i+1],axis=0))
  try:
    BWT.append((1/i)*(np.sum(R_matrix[i,:i])-R_matrix[i,i]))
  except:
      BWT.append(0)


# In[13]:


print(ACC)


# In[14]:


print(BWT)


# # L2 Loss

# In[7]:


class MLP(object):
  def __init__(self, size_input, size_hidden_1, size_hidden_2 , size_output, device=None):
    """
    size_input: int, size of input layer
    size_hidden: int, size of hidden layer
    size_output: int, size of output layer
    device: str or None, either 'cpu' or 'gpu' or None. If None, the device to be used will be decided automatically during Eager Execution
    """
    self.size_input, self.size_hidden_1,self.size_hidden_2, self.size_output, self.device =    size_input, size_hidden_1,size_hidden_2, size_output, device
    
    # Initialize weights between input layer and hidden layer
    self.W1 = tf.Variable(tf.random_normal([self.size_input, self.size_hidden_1]))
    # Initialize biases for hidden layer
    self.b1 = tf.Variable(tf.random_normal([1, self.size_hidden_1]))
     # Initialize weights between hidden layer and output layer
    self.W2 = tf.Variable(tf.random_normal([self.size_hidden_1, self.size_hidden_2]))
    # Initialize biases for output layer
    self.b2 = tf.Variable(tf.random_normal([1, self.size_hidden_2]))
    
    self.W3 = tf.Variable(tf.random_normal([self.size_hidden_2, self.size_output]))
    # Initialize biases for output layer
    self.b3 = tf.Variable(tf.random_normal([1, self.size_output]))
    
    # Define variables to be updated during backpropagation
    self.variables = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]
    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)

    
  def forward(self, X):
    """
    forward pass
    X: Tensor, inputs
    """
    if self.device is not None:
      with tf.device('gpu:0' if self.device=='gpu' else 'cpu'):
        self.y = self.compute_output(X)
    else:
      self.y = self.compute_output(X)
      
    return self.y
  
  def loss(self, y_pred, y_true):
    '''
    y_pred - Tensor of shape (batch_size, size_output)
    y_true - Tensor of shape (batch_size, size_output)
    '''
    
    y_true_tf = tf.cast(y_true, dtype=tf.float32)
    y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
    
    return tf.reduce_mean(tf.reduce_sum(tf.square((y_true_tf-y_pred_tf))))
#     return tf.losses.softmax_cross_entropy(, )
#   loss_op = 
  def backward(self, X_train, y_train):
    """
    backward pass
    """
#     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    with tf.GradientTape() as tape:
      predicted = self.forward(X_train)
      current_loss = self.loss(predicted, y_train)
    grads = tape.gradient(current_loss, self.variables)
    self.optimizer.apply_gradients(zip(grads, self.variables),
                              global_step=tf.train.get_or_create_global_step())
        
        
  def compute_output(self, X):
    """
    Custom method to obtain output tensor during forward pass
    """
    # Cast X to float32
    X_tf = tf.cast(X, dtype=tf.float32)
    #Remember to normalize your dataset before moving forward
    # Compute values in hidden layer
    what1 = tf.matmul(X_tf, self.W1) + self.b1
    hhat1 = tf.nn.relu(what1)
    what2 = tf.matmul(hhat1, self.W2) + self.b2
    hhat2 = tf.nn.relu(what2)
    # Compute output
    output = tf.matmul(hhat2, self.W3) + self.b3
    
    output = tf.nn.softmax(output)
    #Now consider two things , First look at inbuild loss functions if they work with softmax or not and then change this
    #Second add tf.Softmax(output) and then return this variable
    return output


# In[8]:


num_epochs_per_task = 20
n_epochs = 20
minibatch_size = 100
batch_size = 100
num_tasks_to_run = 10
size_input = 784
size_hidden_1 = 256
size_hidden_2 = 256
size_output = 10

R_matrix_L2=list()
#Initialize model using CPU
mlp_on_cpu = MLP(size_input, size_hidden_1,size_hidden_2, size_output, device='cpu')
time_start = time.time()
for task in range(num_tasks_to_run):
  input_kir = each_task_data[task]
  label_kir = train.labels
  input_kir = np.asarray(input_kir,dtype='float32')


#   for i in range(1):
#     if i == task:
#       continue

#     input_kir = pd.read_csv(f"task{i}.csv",header = None) 
#     label_kir = train.labels
#     input_kir = np.asarray(input_kir,dtype='float32')

  if task == 0:
    n_epochs = 50
  else:
    n_epochs = 20
  for epoch in range(n_epochs):
    loss_total = tf.Variable(0, dtype=tf.float32)
    for index, offset in enumerate(range(0, 55000, batch_size)):
      img , label=input_kir[offset: offset + batch_size], label_kir[offset: offset + batch_size]
      
      preds = mlp_on_cpu.forward(img)
      loss_total = loss_total + mlp_on_cpu.loss(preds, label)
  #     print(mlp_on_cpu.loss(preds, label))
      mlp_on_cpu.backward(img, label)  
    print('Number of Epoch = {} - Average MSE:= {:.4f}'.format(epoch + 1, loss_total.numpy() / train.images.shape[0]))
  row_of_R=list()
  for tsk in range(num_tasks_to_run):
    input_kir = each_task_test[tsk]
    label_kir = test.labels
    input_kir = np.asarray(input_kir, dtype = 'float32')
    preds = mlp_on_cpu.forward(input_kir)
    correct_preds = tf.equal(tf.argmax(tf.nn.softmax(preds), 1), tf.argmax(label_kir, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
    print(f'Accuracy of task {tsk+1} is {accuracy}')    
    row_of_R.append(accuracy)
  R_matrix_L2.append(row_of_R)
#   for i in range(num_tasks_to_run):
#     if i == task:
#       continue
#     loss_total = tf.Variable(0, dtype=tf.float32)
#     input_data = pd.read_csv(f"task{task}.csv",header = None) 
#     label_value=train.labels
#     preds = (mlp_on_cpu.forward(np.asarray(input_data,dtype='float32')/255))
#     loss_total = mlp_on_cpu.loss(preds, label)
#     print(f'for task {i} accuracy is {loss_total}')



  time_taken = time.time() - time_start

  print('\nTotal time taken (in seconds): {:.2f}'.format(time_taken))
  #For per epoch_time = Total_Time / Number_of_epochs


# In[15]:


ACC=list()
BWT=list()
R_matrix = np.asarray(R_matrix_L2)
for i in range(num_tasks_to_run):
  ACC.append((1/(i+1))*np.sum(R_matrix[i,:i+1],axis=0))
  try:
    BWT.append((1/i)*(np.sum(R_matrix[i,:i])-R_matrix[i,i]))
  except:
      BWT.append(0)


# In[18]:


print(ACC)


# In[19]:


print(BWT)


# # L1+L2

# In[10]:


class MLP(object):
  def __init__(self, size_input, size_hidden_1, size_hidden_2 , size_output, device=None):
    """
    size_input: int, size of input layer
    size_hidden: int, size of hidden layer
    size_output: int, size of output layer
    device: str or None, either 'cpu' or 'gpu' or None. If None, the device to be used will be decided automatically during Eager Execution
    """
    self.size_input, self.size_hidden_1,self.size_hidden_2, self.size_output, self.device =    size_input, size_hidden_1,size_hidden_2, size_output, device
    
    # Initialize weights between input layer and hidden layer
    self.W1 = tf.Variable(tf.random_normal([self.size_input, self.size_hidden_1]))
    # Initialize biases for hidden layer
    self.b1 = tf.Variable(tf.random_normal([1, self.size_hidden_1]))
     # Initialize weights between hidden layer and output layer
    self.W2 = tf.Variable(tf.random_normal([self.size_hidden_1, self.size_hidden_2]))
    # Initialize biases for output layer
    self.b2 = tf.Variable(tf.random_normal([1, self.size_hidden_2]))
    
    self.W3 = tf.Variable(tf.random_normal([self.size_hidden_2, self.size_output]))
    # Initialize biases for output layer
    self.b3 = tf.Variable(tf.random_normal([1, self.size_output]))
    
    # Define variables to be updated during backpropagation
    self.variables = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]
    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)

    
  def forward(self, X):
    """
    forward pass
    X: Tensor, inputs
    """
    if self.device is not None:
      with tf.device('gpu:0' if self.device=='gpu' else 'cpu'):
        self.y = self.compute_output(X)
    else:
      self.y = self.compute_output(X)
      
    return self.y
  
  def loss(self, y_pred, y_true):
    '''
    y_pred - Tensor of shape (batch_size, size_output)
    y_true - Tensor of shape (batch_size, size_output)
    '''
    
    y_true_tf = tf.cast(y_true, dtype=tf.float32)
    y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
    
    return tf.reduce_mean(tf.reduce_sum(tf.square(y_true_tf-y_pred_tf)+tf.math.abs(y_true_tf-y_pred_tf)))
#     return tf.losses.softmax_cross_entropy(, )
#   loss_op = 
  def backward(self, X_train, y_train):
    """
    backward pass
    """
#     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    with tf.GradientTape() as tape:
      predicted = self.forward(X_train)
      current_loss = self.loss(predicted, y_train)
    grads = tape.gradient(current_loss, self.variables)
    self.optimizer.apply_gradients(zip(grads, self.variables),
                              global_step=tf.train.get_or_create_global_step())
        
        
  def compute_output(self, X):
    """
    Custom method to obtain output tensor during forward pass
    """
    # Cast X to float32
    X_tf = tf.cast(X, dtype=tf.float32)
    #Remember to normalize your dataset before moving forward
    # Compute values in hidden layer
    what1 = tf.matmul(X_tf, self.W1) + self.b1
    hhat1 = tf.nn.relu(what1)
    what2 = tf.matmul(hhat1, self.W2) + self.b2
    hhat2 = tf.nn.relu(what2)
    # Compute output
    output = tf.matmul(hhat2, self.W3) + self.b3
    
    output = tf.nn.softmax(output)
    #Now consider two things , First look at inbuild loss functions if they work with softmax or not and then change this
    #Second add tf.Softmax(output) and then return this variable
    return output


# In[11]:


num_epochs_per_task = 20
n_epochs = 20
minibatch_size = 100
batch_size = 100
num_tasks_to_run = 10
size_input = 784
size_hidden_1 = 256
size_hidden_2 = 256
size_output = 10

R_matrix_L2_L1=list()
#Initialize model using CPU
mlp_on_cpu = MLP(size_input, size_hidden_1,size_hidden_2, size_output, device='cpu')
time_start = time.time()
for task in range(num_tasks_to_run):
  input_kir = each_task_data[task]
  label_kir = train.labels
  input_kir = np.asarray(input_kir,dtype='float32')


#   for i in range(1):
#     if i == task:
#       continue

#     input_kir = pd.read_csv(f"task{i}.csv",header = None) 
#     label_kir = train.labels
#     input_kir = np.asarray(input_kir,dtype='float32')

  if task == 0:
    n_epochs = 50
  else:
    n_epochs = 20
  for epoch in range(n_epochs):
    loss_total = tf.Variable(0, dtype=tf.float32)
    for index, offset in enumerate(range(0, 55000, batch_size)):
      img , label=input_kir[offset: offset + batch_size], label_kir[offset: offset + batch_size]
      
      preds = mlp_on_cpu.forward(img)
      loss_total = loss_total + mlp_on_cpu.loss(preds, label)
  #     print(mlp_on_cpu.loss(preds, label))
      mlp_on_cpu.backward(img, label)  
    print('Number of Epoch = {} - Average MSE:= {:.4f}'.format(epoch + 1, loss_total.numpy() / train.images.shape[0]))
  row_of_R=list()
  for tsk in range(num_tasks_to_run):
    input_kir = each_task_test[tsk]
    label_kir = test.labels
    input_kir = np.asarray(input_kir, dtype = 'float32')
    preds = mlp_on_cpu.forward(input_kir)
    correct_preds = tf.equal(tf.argmax(tf.nn.softmax(preds), 1), tf.argmax(label_kir, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
    print(f'Accuracy of task {tsk+1} is {accuracy}')    
    row_of_R.append(accuracy)
  R_matrix_L2_L1.append(row_of_R)
#   for i in range(num_tasks_to_run):
#     if i == task:
#       continue
#     loss_total = tf.Variable(0, dtype=tf.float32)
#     input_data = pd.read_csv(f"task{task}.csv",header = None) 
#     label_value=train.labels
#     preds = (mlp_on_cpu.forward(np.asarray(input_data,dtype='float32')/255))
#     loss_total = mlp_on_cpu.loss(preds, label)
#     print(f'for task {i} accuracy is {loss_total}')



  time_taken = time.time() - time_start

  print('\nTotal time taken (in seconds): {:.2f}'.format(time_taken))
  #For per epoch_time = Total_Time / Number_of_epochs


# In[12]:


ACC=list()
BWT=list()
R_matrix = np.asarray(R_matrix_L2_L1)
for i in range(num_tasks_to_run):
  ACC.append((1/(i+1))*np.sum(R_matrix[i,:i+1],axis=0))
  try:
    BWT.append((1/i)*(np.sum(R_matrix[i,:i])-R_matrix[i,i]))
  except:
      BWT.append(0)


# In[16]:


print(ACC)


# In[17]:


print(BWT)

