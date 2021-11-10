"""Peform hyperparemeters search"""


# Comments
# info : This goes into the name of the batched dataset that we use to train/evaluate/test
# name: this goes into the name of the dir with the results (fpr/trp, evaluate, train log files, etc)

# DON'T CHANGE THE ARCHITECTURE, i.e. recNet.py for a run in between train and evaluate routines!!!
#-------------------------------------------------------------------------------------------------------------
import argparse
import os
from subprocess import check_call
import sys
import numpy as np
import utils
import time
#-------------------------------------------------------------------------------------------------------------
# Global variables
#-----------------------------

#Directory with the input trees
# sample_name='top_qcd_jets_antikt_antikt'

# jet_algorithm=''

#----------------
# architecture='gatedRecNN'
# architecture='simpleRecNN'
# architecture = 'leaves_inner_RecNN'
# architecture = 'NiNRecNN'
architecture = 'NiNRecNNReLU'
# architecture = 'NiNRecNN2L3W'
# architecture = 'NiNgatedRecNN'
#-------------------------------------------------------

#-----------
# TRAIN_and_EVALUATE
TRAIN_and_EVALUATE=True

load_weights=False
# load_weights=True

#-----------
EVALUATE=False
# EVALUATE=False

# restore_file='last'
restore_file='best'

#-------------------------------------------------------
PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=2,
                    help='Select the GPU')
parser.add_argument('--parent_dir', default='/Users/laurengreenspan/GitDLs/TreeNiNNew/code_ginkgo/recnn/experiments',
                    help='Directory containing params.json')
#parser.add_argument('--data_dir', default='../data/inputTrees/'+sample_name, help="Directory containing the raw datasets")
parser.add_argument('--eval_data_dir', default='../data/preprocessed_trees/', help="Directory containing the input batches")
#parser.add_argument('--sample_name', default=sample_name, help="Sample name")

#parser.add_argument('--jet_algorithm', default=jet_algorithm, help="jet algorithm")

parser.add_argument('--architecture', default=architecture, help="RecNN architecture")

parser.add_argument('--NrunStart', default=0, help="Initial Model Number for the scan")
parser.add_argument('--NrunFinish', default=25, help="Final Model Number for the scan")
parser.add_argument('--sample_type', default='ginkgo', help="sample type")

#-------------------------------------------------------------------------------------------------------------
#//////////////////////    FUNCTIONS     //////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------
#------------------------------------------

#------------------------------------------
# TRAINING
def launch_training_job(parent_dir, data_dir, job_name, params, GPU,sample_name, algo):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    start_time = time.time()
    print('search_hyperparams.py sample_name=',sample_name)
    print('----'*20)
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print('Model dir=',model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    
    if load_weights==False: 
      #---------------
      # Launch training with this config
      cmd_train = "CUDA_VISIBLE_DEVICES={gpu} {python} train.py --model_dir={model_dir} --data_dir={data_dir} --jet_algorithm={algo} --architecture={architecture}".format(gpu=GPU, python=PYTHON, model_dir=model_dir, data_dir=data_dir, algo=algo, architecture=architecture)
      print(cmd_train)
      check_call(cmd_train, shell=True)
    
    else:
      # Launch training with this config and restore previous weights(use --restore_file=best or --restore_file=last)
      cmd_train = "CUDA_VISIBLE_DEVICES={gpu} {python} train.py --model_dir={model_dir} --data_dir={data_dir}  --restore_file=best --jet_algorithm={algo} --architecture={architecture}".format(gpu=GPU, python=PYTHON, model_dir=model_dir, data_dir=data_dir, algo=algo, architecture=architecture)
      print(cmd_train)
      check_call(cmd_train, shell=True)


    elapsed_time=time.time()
    print('Training time (minutes) = ',(elapsed_time-start_time)/60)


#------------------------------------------
# EVALUATION
def launch_evaluation_job(parent_dir, data_dir, eval_data_dir, job_name, params, GPU,sample_name, algo):
    """Launch evaluation of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    elapsed_time = time.time()
    print('Running evaluation of the model')
    print('----'*20)
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print('Model dir=',model_dir)


    #--------------
    # Launch evaluation with this config
    cmd_eval = "CUDA_VISIBLE_DEVICES={gpu} {python} evaluate.py --model_dir={model_dir} --data_dir={data_dir} --sample_name={sample_name} --jet_algorithm={algo} --architecture={architecture} --restore_file={restore_file}".format(gpu=GPU, python=PYTHON, model_dir=model_dir, data_dir=eval_data_dir,sample_name=sample_name, algo=algo, architecture=architecture, restore_file=restore_file)
    print(cmd_eval)
    check_call(cmd_eval, shell=True)

    eval_time=time.time()
    print('Evaluation time (minutes) = ',(eval_time-elapsed_time)/60)
    
    
#-------------------------------------------------------------------------------------------------------------
###///////////////////////////////////////////////////////////////////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'template_params.json')
    print("path is"+str(json_path))
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    NrunStart= int(args.NrunStart)
    NrunFinish= int(args.NrunFinish)

    # Perform hyperparameters scans
    def multi_scan(jet_algorithm,learning_rates,decays, batch_sizes,num_epochs,hidden_dims,jet_numbers,Nfeatures,dir_name,name, info, sample_name, Nrun_start,Nrun_finish):
    
      parent_dir=args.parent_dir+str(dir_name)+'/'
      if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
#           os.system('mkdir -p '+parent_dir)

      #-------------------------------------------------------------
      # Loop to scan over the hyperparameter space
      for jet_algo in jet_algorithm:
          for jet_number in jet_numbers:
            for hidden_dim in hidden_dims:
              for num_epoch in num_epochs:
                for batch_size in batch_sizes:
                  for decay in decays: 
                    for learning_rate in learning_rates:

                      params.learning_rate=learning_rate
                      params.decay=decay
                      params.batch_size=batch_size
                      params.num_epochs=num_epoch
                      params.hidden=hidden_dim
                      params.features=Nfeatures
        #           params.number_of_labels_types=1
                      params.myN_jets=jet_number
                      params.info=info #This goes into the name of the batched dataset that we use to train/evaluate/test
                      params.nrun_start=Nrun_start
                      params.nrun_finish=Nrun_finish
                      params.jet_algorithm = jet_algo 
                  #-----------------------------------------
                  # Launch job (name has to be unique)
                      sample_name = args.sample_type+'_'+jet_algo+'_'+str(params.myN_jets)+'_jets'
                      job_name = str(sample_name)+'_'+str(name)+'_lr_'+str(learning_rate)+'_decay_'+str(decay)+'_batch_'+str(batch_size)+'_epochs_'+str(num_epoch)+'_hidden_'+str(hidden_dim)+'_Njets_'+str(jet_number)+'_features_'+str(params.features)

                 
                  
                  #-----------------------------------------                
                  # Run training, evaluation 
                      if TRAIN_and_EVALUATE:
                        for n_run in np.arange(Nrun_start,Nrun_finish):
                          launch_training_job(parent_dir, args.data_dir, args.eval_data_dir, job_name+'/run_'+str(n_run), params, args.gpu, sample_name, jet_algorithm)        

                          launch_evaluation_job(parent_dir, args.data_dir, args.eval_data_dir, job_name+'/run_'+str(n_run), params, args.gpu, sample_name, jet_algorithm)


                      if EVALUATE:
                        for n_run in np.arange(Nrun_start,Nrun_finish):
                          launch_evaluation_job(parent_dir, args.data_dir, args.eval_data_dir, job_name+'/run_'+str(n_run), params, args.gpu, sample_name, jet_algorithm) 

multi_scan(
jet_algorithm=['kt', 'antikt','ptdesc','truth','ptasc'],
learning_rates=[2e-3],
decays=[0.9],
batch_sizes=[64],
num_epochs=[40],
hidden_dims=[64,128,256,512,1024],
jet_numbers=[1200000], 
Nfeatures=7,
dir_name=str(args.eval_data_dir),
name=architecture,
sample_name=str(args.sample_name),
Nrun_start=NrunStart,
Nrun_finish=NrunFinish) #gpu1   




                        
###########################################################

#-------------------------------


#Simple
#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128], num_epochs=[40], hidden_dims=[40], jet_numbers=[1200000], Nfeatures=7, dir_name='top_tag_reference_dataset', name=architecture+'_kt_R_0.3_rot_boost_rot_flip', info='R_0.3_rot_boost_rot_flip', sample_name=args.sample_name, Nrun_start=6, Nrun_finish=9) #gpu1 



'''
antikt_antikt
learning_rates=[1e-2, 5e-3,2e-3,5e-4] 
decays=[0.9,0.8,0.7]
batch_sizes=[64,128,256,512,1024]
num_epochs=[30]
hidden_dims=[20,40,80,160,320,640]
jet_numbers = [40000,80000,160000]

-----------------------------------
antikt_kt
learning_rates=[1e-2, 5e-3,2e-3,1e-3] 
decays=[0.9,0.8,0.7]
batch_sizes=[64,128,256,512,1024]
num_epochs=[35]
hidden_dims=[20,40,80,160,320,640]
jet_numbers = [40000,80000,160000]
'''       

