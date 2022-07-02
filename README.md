# DGAN

## Deep Generative Autoencoder Network: An imputation variational autoencoder model for downstream functional analysis of single-cell RNA-sequence (scRNA-seq) data.
Diksha Pandey, Onkara Perumal P

## Getting Started
DGAN has been implemented in Python 2.7. The recommended version is Python3.
These instructions will get you a copy of the project up and running on your local machine.

## Prerequisites
* For imputation and preprocessing  
   For Python (2.7):
    > numpy, pandas, scipy, scikit-learn, tensorflow, matplotlib.

## Chapters
* `DGAN/DGAN Model/DGAN/DGAN.py` - Imputation model.
* `DGAN/DGAN Model/Data/Karen_raw.csv` - raw data in `.csv` file format.
* `DGAN/DGAN Model/Pre-processing/dataPreprocessing.py` - Python scripts for pre-processing.

##Run
DGAN can be run through tensorflow (Jupyter) or command line.

Choices:

          Extension
          #help:Input file type supported format is comma-separated values, default: .csv

          Error
          #help:Print the different types of error for the model, default: true

          log
          #help:Collect log for debugging if needed, default: true

          collect_log_upto
          #help:Printing steps upto this counter, default: 5

          hidden_units 
          #help:Size of hidden layer or latent space dimensions, default:2000

          hidden_neurons
          #help:Number of neurons in the hidden layer of model, default: 1000

          1E10
          #help:Tuning the cost function by upto this value,default:1e-10

          learning_cost
          #help:The amount that the weights are updated during training for the model, default:0.0001

          epoch
          #help:Number of iteration required to train the model, default:50 (To save the bandwidth)

          batch_size
          #help:the number of samples that will be propagated through the network, default:100

          latent_space
          #help:encoder compress the data from initial space to encoded space, default:500

          threshold       
          #help:the cut off value of the function used to quantify the output of a neuron in the output layer, default:0.001 

          input_data   
          #help:the input dataset on which want to run the model script, default:Karen_raw.csv    

          masking
          #help:to check tell sequence-processing layers that certain timesteps in an input are missing, and thus should be skipped when processing the data (matrix test), defult: True

          mask_condition      
          #help:masking condition, default:50%

          model_save_at   
          #help:save the model index and meta files at, default:file_location/

          log_file
          #help:for debugging save the stpes per iteration in log file, default:log.log

          denoised_matrix
          #help:save the denoised/imputed matrix in the name as denoised_matrix, default: denoised_matrix

          masked_matrix
          #help:save the masked matrix in the name of masked_matrix, default:masked_matrix

ex: python DGAN.py <data> <choices>

A more detailed usage of DGAN's functionality is available in the iPython Notebook DGAN.ipynb
