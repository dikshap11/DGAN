# DGAN
===========
## Deep Generative Autoencoder Network [DGAN]: An imputation variational autoencoder model for downstream functional analysis of single-cell RNA-sequence (scRNA-seq) data.

Diksha Pandey, Onkara Perumal P

DGAN has been implemented in Python3.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

## Prerequisites
* For imputation and preprocessing  
   For Python (3):
    > numpy, pandas, scipy, scikit-learn, tensorflow, matplotlib.


## Chapters
* `DGAN/DGAN Model/DGAN/DGAN.py` - Imputation model.
* `DGAN/DGAN Model/Data/Karen_raw.csv` - raw data in `.csv` format.
* `DGAN/DGAN Model/Pre-processing/dataPreprocessing.py` - Python scripts for pre-processing.

##Run
DGAN can be run through tensorflow (Jupyter) or command line.

Choices:
           Dict['Extension'] = '.csv'
          #help:Input file type supported format is comma-separated values, default: .csv

          Dict['Error'] = True
          #help:Print the different types of error for the model, default: true

          Dict['log']= True
          #help:Collect log for debugging if needed, default: true

          Dict['collect_log_upto'] = 10
          #help:Printing steps upto this counter, default: 5

          Dict['hidden_units']= 2000   
          #'--hidden_units', type=int, default=2000, help="Size of hidden layer or latent space dimensions."

          Dict['hidden_neurons'] = 1000
          #help:Number of neurons in the hidden layer of model, default: 1000

          Dict['1E10'] = '1e-10'
          #help:Tuning the cost function by upto this value,default:1e-10

          Dict['learning_cost'] = 0.00001
          #help:The amount that the weights are updated during training for the model, default:0.0001

          Dict['epoch'] = 50
          #help:Number of iteration required to train the model, default:50 (To save the bandwidth)

          Dict['batch_size'] = 100
          #help: the number of samples that will be propagated through the network, default:100

          Dict['latent_space'] = 500
          #help: encoder compress the data from initial space to encoded space, default:500

          Dict['threshold'] = 0.001        
          #help: the cut off value of the function used to quantify the output of a neuron in the output layer, default:0.001 

          Dict['input_data'] = 'karen_raw.csv'    
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

ex: python DGAN.py <data> <choices>

A more detailed usage of DGAN's functionality is available in the iPython Notebook DGAN.ipynb
