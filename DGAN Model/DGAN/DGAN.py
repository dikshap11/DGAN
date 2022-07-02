#!/usr/bin/env python
# coding: utf-8

# In[35]:


from __future__ import print_function, division, unicode_literals, absolute_import, annotations
# Place this before directly or indirectly importing tensorflow
import tensorflow as tf
import numpy as np
import scipy.io
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[36]:


Dict = {}
def cmdArg(arg = 'cmd_input'):
    print("arg: ", arg)
    
    Dict['Extension'] = '.csv'
    #help:Input file type supported format is comma-separated values, default: .csv
    
    Dict['Error'] = True
    #help:Print the different types of error for the model, default: true
    
    Dict['log']= True
    #help:Collect log for debugging if needed, default: true
    
    Dict['collect_log_upto'] = 10
    #help:Printing steps upto this counter, default: 5
    
    Dict['hidden_units']= 2500   
    #'--hidden_units', type=int, default=2000, help="Size of hidden layer or latent space dimensions."
    
    Dict['hidden_neurons'] = 900   
    #help:Number of neurons in the hidden layer of model, default: 1000
    
    Dict['1E10'] = '1e-10'
    #help:Tuning the cost function by upto this value,default:1e-10
    
    Dict['learning_cost'] = 0.0001
    #help:The amount that the weights are updated during training for the model, default:0.0001
    
    Dict['epoch'] = 25
    #help:Number of iteration required to train the model, default:50 (To save the bandwidth)
    
    Dict['batch_size'] = 100
    #help: the number of samples that will be propagated through the network, default:100
    
    Dict['latent_space'] = 350
    #help: encoder compress the data from initial space to encoded space, default:500
    
    Dict['threshold'] = 0.001        
    #help: the cut off value of the function used to quantify the output of a neuron in the output layer, default:0.001 
    
    Dict['input_data'] = 'blakeley.csv'    
    #help: the input dataset on which want to run the model script, default:blakeley.csv    
    
    Dict['masking'] = True
    #help:to check tell sequence-processing layers that certain timesteps in an input are missing, and thus should be skipped when processing the data (matrix test), defult: True
    
    Dict['mask_condition'] = 0.5      
    #help:masking condition, default:50%
    
    Dict['model_save_at'] = 'file_location/'   
    #help:save the model index and meta files at, default:file_location/
    
    Dict['log_file'] = 'log.log'
    #help:for debugging save the stpes per iteration in log file, default:log.log
    
    Dict['denoised_matrix'] = 'denoised_matrix'
    #help:save the denoised/imputed matrix in the name as denoised_matrix, default: denoised_matrix

    Dict['masked_matrix'] = 'masked_matrix'
    #help:save the masked matrix in the name of masked_matrix, default:masked_matrix
    
    if arg == 'cmd_input':
        if len(sys.argv) == 1:
            exit("No arguments passed exiting")
        if sys.argv[1] == 'Extension':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
        if sys.argv[1] == 'Error':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
        if sys.argv[1] == 'hidden_units':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
        if sys.argv[1] == 'hidden_neurons':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
        if sys.argv[1] == '1E10':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
        if sys.argv[1] == 'learning_cost':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
        if sys.argv[1] == 'epoch':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
        if sys.argv[1] == 'batch_size':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
        if sys.argv[1] == 'latent_space':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
        if sys.argv[1] == 'threshold':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
        if sys.argv[1] == 'input_data':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
        if sys.argv[1] == 'masking':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
        if sys.argv[1] == 'mask_condition':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
        if sys.argv[1] == 'model_save_at':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
        if sys.argv[1] == 'log_file':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
        if sys.argv[1] == 'load_saved':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
        if sys.argv[1] == 'denoised_matrix':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
        if sys.argv[1] == 'masked_matrix':
            if len(sys.argv) == 3:
                Dict[sys.argv[1]] = sys.argv[2]
    
if __name__ == '__main__':
    cmdArg()


# In[37]:


#Reading the input data
if(Dict['Extension'] == '.csv'):
    #Keep your data under input_data/ folder to read it successfully
    input_matrix = pd.read_csv('C:/VAE_Project/10X/preprocessing_10x.csv', header = 0) 
    first_column = input_matrix.columns[0]
    input_matrix = input_matrix.drop([first_column], axis=1)
    input_matrix = input_matrix.to_numpy()
    #input_matrix = np.transpose(input_matrix)
    print("Read {0} data successfully".format(Dict['input_data']))
else:
    print("Please provide data in csv format")


# In[38]:


# Dimensionality of input matrix
ip_r = np.size(input_matrix, 0)
ip_c = np.size(input_matrix, 1)
print(ip_c)
shape = input_matrix.shape
print("Shape of input_matrix : {0}".format(shape))


# In[39]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# Defining placeholder to feed data into the graph at later stage
x_true = tf.placeholder(tf.float32, shape=[None, ip_c], name = 'input')
mask = tf.placeholder(tf.float32, shape=[None, ip_c], name = 'input')


# In[40]:


#Masking to handle the missing, invalid, or unwanted entries in our array or dataset/dataframe
if(Dict['masking']):
    masked_matrix = input_matrix.copy()
    #Masking the invalid valid data from input matrix
    masked_matrix = np.ma.masked_invalid(masked_matrix)
    
    #Defining masking function
    def masking(arr1, arr2):
        mask_check = arr1 >= Dict['mask_condition']
        mm = np.ma.masked_where(mask_check, arr2, copy = True)
        return mm
    
    # main function
    if __name__ == '__main__':
    # calling masking function to get masked array
        masked_matrix = masking(masked_matrix, input_matrix)

    masked_matrix[masked_matrix != 0] = 1
    
    ##Save masked matrix into a file for future reference
    if(Dict['masked_matrix']):
        np.savetxt(Dict['masked_matrix'] + Dict['Extension'], masked_matrix, 
              delimiter = ",")
        
        print("Masked {0} matrix saved at {1}".format(Dict['masked_matrix'] + Dict['Extension'],
                                                      Dict['model_save_at']))


# In[41]:


#Parameter
learning_cost = Dict['learning_cost']
num_epoch = Dict['epoch']
batch_size = Dict['batch_size']

# Network Parameters
input_capacity = ip_c 
hidden_capacity = Dict['hidden_neurons']
latent_capacity = Dict['latent_space']

print("Summary of HyperParameter of Network!")
print("Learning Cost : {0}".format(Dict['learning_cost']))
print("Num of Epoch : {0}".format(Dict['epoch']))
print("Batch Size : {0}".format(Dict['batch_size']))
print("Input Dimension : {0}".format(ip_c))
print("Hidden Dimension : {0}".format(Dict['hidden_neurons']))
print("Latent Dimension : {0}".format(Dict['latent_space']))

# Defination for glorot_init
def xavier_init(shape):
    return tf.random.normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


# In[42]:


# Define weights and biases for our network
weights = {
    'encoder_h1': tf.Variable(xavier_init([input_capacity, hidden_capacity])),
    'z_mean': tf.Variable(xavier_init([hidden_capacity, latent_capacity])),
    'z_std': tf.Variable(xavier_init([hidden_capacity, latent_capacity])),
    'decoder_h1': tf.Variable(xavier_init([latent_capacity, hidden_capacity])),
    'decoder_out': tf.Variable(xavier_init([hidden_capacity, input_capacity]))
}
biases = {
    'encoder_b1': tf.Variable(xavier_init([hidden_capacity])),
    'z_mean': tf.Variable(xavier_init([latent_capacity])),
    'z_std': tf.Variable(xavier_init([latent_capacity])),
    'decoder_b1': tf.Variable(xavier_init([hidden_capacity])),
    'decoder_out': tf.Variable(xavier_init([input_capacity]))
}


# In[43]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras import layers

# Building the encoder with latent space z_mean (mean) & z_std (variance or std deviation)
input_data = tf.placeholder(tf.float32, shape=[None, ip_c])
encoder = tf.matmul(input_data, weights['encoder_h1']) + biases['encoder_b1']
encoder = tf.nn.tanh(encoder)
z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']

#Defining the Sampling
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Sampler: Normal (gaussian) random distribution
eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,
                       name='epsilon')
# Sampled from z_mean & z_std
z = z_mean + tf.exp(z_std / 2) * eps

# Building the decoder
decoder = tf.matmul(z, weights['decoder_h1']) + biases['decoder_b1']
decoder = tf.nn.tanh(decoder)
decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
decoder = tf.nn.sigmoid(decoder)


# In[44]:


# Define loss function of our model in form of reconstrution & KL divergence loss
@tf.function
def vae_loss(x_generative, x_true):
    # Reconstruction loss
    reconstruction_loss = x_true * tf.log(1e-10 + x_generative)                          + (1 - x_true) * tf.log(1e-10 + 1 - x_generative)
    reconstruction_loss = -tf.reduce_sum(reconstruction_loss, 1)

    # KL Divergence loss
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(reconstruction_loss + kl_div_loss)

model_loss = vae_loss(decoder, input_data)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_cost)
train_opt = optimizer.minimize(model_loss)


# In[45]:


import numpy
import csv
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

if __name__ == '__main__':
    # Start a new TF session to run our model
    y_pred = decoder
    sess = tf.Session()

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Run the initializer
    sess.run(init)
    
    # Training our model
    for i in range(1, num_epoch+1):
        # Train
        feed_dict_session = {input_data: input_matrix}
        rmse_loss = tf.pow(tf.norm(input_data - decoder * mask), 2)
        _, ls = sess.run([train_opt, model_loss], feed_dict=feed_dict_session)
        lspercell = ls/ip_r
        if i % 1 == 0 or i == 1:
            #print('Epoch %i [========] Total Loss: %f Reconstrction Loss %f' % (i, ls, lspercell))
            if(Dict['log']):
                print('Epoch %i [========] Total Loss: %f Reconstrction Loss %f' % (i, ls, lspercell))
                with open(Dict['log_file'], 'a') as log:
                    log.write('{0}\t{1}\t{2}\n'.format(i, ls, lspercell))
    
    #Get the denoised matrix
    denoised_matrix = sess.run([decoder], feed_dict={input_data: input_matrix, mask: masked_matrix})
    scipy.io.savemat( "mahesh_preprocessing_model_imputed_10x.mat", mdict = {"arr" : denoised_matrix})
    
    #Calculate the different types of errors to evaluate model.
    if(Dict['Error']):
        act = input_matrix
        pred = denoised_matrix
        
        def root_mean_squared_error(act, pred):
            diff = pred - act
            differences_squared = diff ** 2
            mean_diff = differences_squared.mean()
            rmse_val = np.sqrt(mean_diff)
            return rmse_val

        def mean_absolute_error(act, pred):
            diff = pred - act
            abs_diff = np.absolute(diff)
            mean_diff = abs_diff.mean()
            return mean_diff

        def mean_absolute_percentage_error(act, pred):
            diff = pred - act
            abs_diff = np.absolute(diff)
            mean_diff = abs_diff.mean()
            return (mean_diff*100)

        def hinge_loss_error(act, pred):
            diff = 1 - act * pred
            abs_diff = np.absolute(diff)
            mean_diff = abs_diff.mean()
            return (mean_diff*100)
                
        print("--Error Summary--")
        print("RMSE :",root_mean_squared_error(act,pred))
        print("MAE  :",mean_absolute_error(act,pred))
        print("MAPE :",mean_absolute_percentage_error(act,pred))
        print("Hinge:",hinge_loss_error(act,pred))


# In[ ]:




