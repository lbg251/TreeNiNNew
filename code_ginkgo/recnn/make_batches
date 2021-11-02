import argparse
import os
from subprocess import check_call
import sys
import numpy as np
import utils
import time
import logging
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
import pickle
import model.data_loader as dl
import model.dataset as dataset
from model import recNet as net
from model import preprocess 

### Preprocess the code with transformer, and split into 
# train, val, and test sets
##----------------------------------------------------------------------------------------------------------
if __name__=='__main__':  

  print('Preprocessing jet trees ...')
  print('==='*20)
 

  
  ##-------------------
##------------------------------------------------------------  
algo='ptdesc'
filename = 'ginkgo_'+str(algo)+'_48jets.pkl'

parser = argparse.ArgumentParser()
parser.add_argument('--jet_algorithm', default=algo, help="jet algorithm")
parser.add_argument('--data_dir', default='../data/reclustered_trees/', help="Directory containing the input dataset")
parser.add_argument('--eval_data_dir', default='../data/preprocessed_trees/', help="Directory containing the input batches")
parser.add_argument('--in_filename', default=filename,help="Name of the input data file")

args = parser.parse_args()
data_dir= args.data_dir
algo=args.jet_algorithm
in_filename = args.in_filename
eval_data_dir = args.eval_data_dir
sample_name=str(in_filename).split('.')[0]
print('sample name is',sample_name)
logging.info('sample_name={}'.format(sample_name))
logging.info('----'*20)

train_data=eval_data_dir+'train_'+sample_name+'.pkl'
val_data=eval_data_dir+'dev_'+sample_name+'.pkl'
test_data=eval_data_dir+'test_'+sample_name+'.pkl'

transformer_filename=sample_name
transformer_data=eval_data_dir+'transformer_'+transformer_filename+'.pkl'

start_time = time.time()  
##-----------------------------------------------------------------------------------------------------------
data_loader=dl.DataLoader # Main class with the methods to load the raw data, create and preprocess the trees


# loading dataset_params and make trees
logging.info('Loading dataset={}'.format(str(data_dir)+in_filename))
with open(data_dir+in_filename, "rb") as f: data_list =pickle.load(f)  

    
 ## Splitting dataset into signal and background

sig_list=[data_list[k][0] for k in range(len(data_list))if data_list[k][1]==1]
bkg_list=[data_list[k][0] for k in range(len(data_list))if data_list[k][1]==0] 

    ##-------------------
# Split into train+validation+test and shuffle
logging.info("Splitting into train, validation and test datasets, and shuffling...")
train_x, train_y, dev_x, dev_y, test_x, test_y = data_loader.split_shuffle_sample(sig_list, bkg_list, 0.6, 0.2, 0.2)

##-------------------
# Apply RobustScaler (remove outliers, center and scale data)
transformer=data_loader.get_transformer(train_x)

# Save transformer
with open(transformer_data, "wb") as f: pickle.dump(transformer, f)

#Scale features using the training set transformer
train_x = data_loader.transform_features(transformer,train_x)
dev_x = data_loader.transform_features(transformer,dev_x)
test_x = data_loader.transform_features(transformer,test_x)

    
##---------------------------------   
elapsed_time=time.time()-start_time
logging.info('Split sample time (minutes) ={}'.format(elapsed_time/60))

# Save trees
with open(train_data, "wb") as f: pickle.dump(zip(train_x,train_y), f)
with open(val_data, "wb") as f: pickle.dump(zip(dev_x,dev_y), f)
with open(test_data, "wb") as f: pickle.dump(zip(test_x,test_y), f)
