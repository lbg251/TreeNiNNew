import wandb
import argparse
from model import recNet as net
import logging
import torch.optim as optim
import torch
import time
import os
import numpy as np
from sklearn.metrics import roc_curve, auc
from numpy import interp
import utils
import model.data_loader as dl
import model.dataset as dataset
from tqdm import trange
import pickle

#-------------------------------------------------------------------------------------------------------------
#/////////////////////    TRAINING AND EVALUATION FUNCTIONS     //////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------
def train(model, optimizer, loss_fn, data_iterator, metrics, params, num_steps):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network superclass
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
#     wandb.init(project="Ginkgo Tree", entity="lbg251")
#     wandb.watch(model)
    model.train()
    
    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    
    ##-----------------------------
    # Use tqdm for progress bar
    t = trange(num_steps) 
    
    data_iterator_iter = iter(data_iterator)
    
    for i in t:
    
        time_before_batch=time.time() 
        
        # fetch the next training batch
        levels, children, n_inners, contents, n_level, labels_batch=next(data_iterator_iter)

        # shift tensors to GPU if available
        if params.cuda:
          levels = levels.cuda()
          children=children.cuda()
          n_inners=n_inners.cuda()
          contents=contents.cuda()
          n_level= n_level.cuda()
          labels_batch =labels_batch.cuda()
      
        # convert them to Variables to record operations in the computational graph
        levels=torch.autograd.Variable(levels)
        children=torch.autograd.Variable(children)
        n_inners=torch.autograd.Variable(n_inners)
        contents = torch.autograd.Variable(contents)
        n_level=torch.autograd.Variable(n_level)
        labels_batch = torch.autograd.Variable(labels_batch)    
    
        time_after_batch=time.time()
#         logging.info("Batch creation time" + str(time_after_batch-time_before_batch))
        
        ##-----------------------------
        # Feedforward pass through the NN
        output_batch = model(params, levels, children, n_inners, contents, n_level)
        
        
#         logging.info("Batch usage time" + str(time.time()-time_after_batch))
#         logging.info('####'*20)
        
        # compute model output and loss
        labels_batch = labels_batch.float()  #Uncomment if using torch.nn.BCELoss() loss function
        output_batch=output_batch.view((params.batch_size)) # For 1 final neuron 
        loss = loss_fn(output_batch, labels_batch)
        
#         print('output_batch=',output_batch)
#         print('labels_batch=',labels_batch)
#         print('y_pred=',output_batch)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()
        
        ##-----------------------------
        # Evaluate summaries only once in a while
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)
        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg())) #Uncomment once tqdm is installed
     
#     print('summ=',summ)    
    ##-----------------------------
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.4f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
#     print('metrics_mean=',metrics_mean)
#     print('metrics_string=',metrics_string)
    return metrics_mean
    
#-------------------------------------------------------------------------------------------------------------
def evaluate(model, loss_fn, data_iterator, metrics, params, num_steps):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network superclass
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    
    output_all=[]
    labels_all=[]
    ##-----------------------------
    # compute metrics over the dataset
    
    data_iterator_iter = iter(data_iterator)

    for _ in range(num_steps):
    
        # fetch the next evaluation batch
        levels, children, n_inners, contents, n_level, labels_batch=next(data_iterator_iter)

        # shift tensors to GPU if available
        if params.cuda:
          levels = levels.cuda()
          children=children.cuda()
          n_inners=n_inners.cuda()
          contents=contents.cuda()
          n_level= n_level.cuda()
          labels_batch =labels_batch.cuda()

        # convert them to Variables to record operations in the computational graph
        levels=torch.autograd.Variable(levels)
        children=torch.autograd.Variable(children)
        n_inners=torch.autograd.Variable(n_inners)
        contents = torch.autograd.Variable(contents)
        n_level=torch.autograd.Variable(n_level)
        labels_batch = torch.autograd.Variable(labels_batch)    

        ##-----------------------------
        # Feedforward pass through the NN
        output_batch = model(params, levels, children, n_inners, contents, n_level)


        # compute model output
        labels_batch = labels_batch.float() #Uncomment if using torch.nn.BCELoss() loss function
        output_batch=output_batch.view((params.batch_size)) # For 1 final neuron 
        loss = loss_fn(output_batch, labels_batch)
#         print('labels for loss=',labels_batch)
#         print('y_pred=',output_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # Save labels and output prob of the current batch
        labels_all=np.concatenate((labels_all,labels_batch))        
        output_all=np.concatenate((output_all,output_batch))




        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
#         summary_batch['loss'] = loss.data[0]
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)
        
        
    ##-----------------------------
    
    ##Get the bg rejection at 30% tag eff: 0.05 + 125*(1 - 0.05)/476=0.3). That's why we pick 125
    fpr, tpr, thresholds = roc_curve(labels_all, output_all,pos_label=1, drop_intermediate=False)
    base_tpr = np.linspace(0.05, 1, 476)
    inv_fpr = interp(base_tpr, tpr, 1. / fpr)[125]
#     print('inv_fpr at 30% tag eff=',inv_fpr)
   #wandb.log({'roc': wandb.plots.ROC(labels_all, output_all, pos_label=1)})
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.4f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean, inv_fpr


#-------------------------------------------------------------------------------------------------------------
def train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, model_dir, step_size, restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network superclass
        train_data: array with levels, children, n_inners, contents, n_level and labels_batch lists
        val_data: array levels, children, n_inners, contents, n_level and labels_batch lists
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log files
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
        
    best_val_acc = 0.0
#     best_val_acc = np.inf
    
    #Save loss, accuracy history
    history={'train_loss':[],'val_loss':[],'train_accuracy':[],'val_accuracy':[],'val_bg_reject':[]}
    
    ##------
    #Create lists to access the lenght below
    train_data=list(train_data)
    val_data=list(val_data)    
#     print('train data lenght=',len(train_data))
   
    num_steps_train=len(train_data)//params.batch_size
    num_steps_val=len(val_data)//params.batch_size
      
    # We truncate the dataset so that we get an integer number of batches    
    train_x=np.asarray([x for (x,y) in train_data][0:num_steps_train*params.batch_size])
    train_y=np.asarray([y for (x,y) in train_data][0:num_steps_train*params.batch_size])        
    val_x=np.asarray([x for (x,y) in val_data][0:num_steps_val*params.batch_size])
    val_y=np.asarray([y for (x,y) in val_data][0:num_steps_val*params.batch_size])
    
    ##------
    # Create tain and val datasets. Customized dataset class: dataset.TreeDataset that will create the batches by calling data_loader.batch_nyu_pad. 
    train_data = dataset.TreeDataset(data=train_x,labels=train_y,transform=data_loader.batch_nyu_pad,batch_size=params.batch_size,features=params.features)
    
    val_data = dataset.TreeDataset(data=val_x,labels=val_y,transform=data_loader.batch_nyu_pad,batch_size=params.batch_size,features=params.features,shuffle=False)
  
    ##------
    # Create the dataloader for the train and val sets (default Pytorch dataloader). Paralelize the batch generation with num_workers. BATCH SIZE SHOULD ALWAYS BE = 1 (batches are only loaded here as a single element, and they are created with dataset.TreeDataset).
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False,
                                               num_workers=4, pin_memory=True, collate_fn=dataset.customized_collate) 
                                               
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False,
                                               num_workers=4, pin_memory=True, collate_fn=dataset.customized_collate) 
    
    ##------
    # Train/evaluate for each epoch
    for epoch in range(params.num_epochs):
    
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
      
        # Train one epoch
        train_metrics = train(model, optimizer, loss_fn, train_loader, metrics, params, num_steps_train)
            
        # Evaluate for one epoch on validation set
        val_metrics, inv_fpr = evaluate(model, loss_fn, val_loader, metrics, params, num_steps_val)      

          # Minimize the accuracy on the val set  
#         val_acc = val_metrics['accuracy']
#         is_best = val_acc >= best_val_acc

#         
#         # Minimize the loss on the val set
#         val_acc = val_metrics['loss']
#         is_best = val_acc <= best_val_acc
        
        
        # Maximize the bg rejection at 30% tag eff on the val set
        val_acc = inv_fpr
        is_best = val_acc >= best_val_acc
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_bg_reject'].append(inv_fpr)
        wandb.log({"train_loss":train_metrics['loss'],"train_accuracy":train_metrics['accuracy'],"val_loss":val_metrics['loss'],"val_accuracy":val_metrics['accuracy']})
        scheduler.step()
        step_size = step_size * decay
        
        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()}, 
                               is_best=is_best,
                               checkpoint=model_dir)
            
        # If best_eval, best_save_path        
        if is_best:
#             logging.info("- Found new best accuracy")
#             logging.info("- Found new lowest loss")
            best_val_acc = val_acc
            logging.info('- Found new best bg rejection = {}'.format(best_val_acc))
            
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        # Save loss history in a json file in the model directory
#         print('loss_hist=',loss_hist)
        hist_json_path = os.path.join(model_dir, "metrics_history.json")
        utils.save_dict_list_to_json(history, hist_json_path)    

#-------------------------------------------------------------------------------------------------------------
###///////////////////////////////////////////////////////////////////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------
if __name__=='__main__':  
  ##----------------------------------------------------------------------------------------------------------
  # Global variables
  ##-------------------
  data_dir='../data/'
  os.system('mkdir -p '+data_dir)
  
  # Select the right dir for jets data
  trees_dir='preprocessed_trees/'
  os.system('mkdir -p '+data_dir+'/'+trees_dir)


  sample_name='ginkgo'
  algo=''

  sg='ttbar' 
  bg='qcd'

  ##------------------------------------------------------------  
  parser = argparse.ArgumentParser()
  parser.add_argument('--sample_name', default='ginkgo', help="Directory containing the raw datasets")
  parser.add_argument('--data_dir', default='../data/preprocessed_trees/', help="Directory containing the raw datasets")
  parser.add_argument('--model_dir', default='experiments/ginkgo', help="Directory containing params.json")
  parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'last'

  parser.add_argument('--jet_algorithm', default=algo, help="jet algorithm")
  parser.add_argument('--architecture', default='simpleRecNN', help="RecNN architecture")

  parser.add_argument('--gpu', default=2,
                  help='Select the GPU')
  parser.add_argument('--parent_dir', default='experiments/',
                    help='Directory containing params.json')
  parser.add_argument('--eval_data_dir', default='../data/preprocessed_trees/', help="Directory containing the input batches")

  parser.add_argument('--NrunStart', default=0, help="Initial Model Number for the scan")
  parser.add_argument('--NrunFinish', default=1, help="Final Model Number for the scan")


  args = parser.parse_args()
  jet_algorithm = args.jet_algorithm
  architecture=args.architecture

  ##----------------------------------------------------------------------
  sweep_config = {
  "name" : str(jet_algorithm)+"sweep",
  "method" : "random",
  "parameters" : {
    "epochs" : {
      "values" : [10, 20, 50]
    },
    "learning_rate" :{
      "min": 0.0001,
      "max": 0.1
    },
    "batch_size":{
      "values":[16,32,64,128,256,512,1024]
      },
      "decays":{"min":.6,"max":1.},
      "hidden_dims":{"values":[20,40,80,160,320,640]},
      "optimizer":{"values":["adam"]}
  }
  }
  params = sweep_config
  data_loader=dl.DataLoader # Main class with the methods to load the raw data, create and preprocess the trees
  # use GPU if available
  params.cuda = torch.cuda.is_available()

  # Set the random seed for reproducible experiments
  #   torch.manual_seed(230)
  #   if params.cuda: torch.cuda.manual_seed(230)
  if params.cuda: torch.cuda.seed()
  ##-----------------------------
  # Create the input data pipeline 
  logging.info('---'*20)
  logging.info("Loading the datasets...")

  # Load data 
  with open(train_data, "rb") as f: train_data=pickle.load(f)
  with open(val_data, "rb") as f: val_data=pickle.load(f) 
  logging.info("- done loading the datasets") 
  logging.info('---'*20)   

  def train_sweep():
    with wandb.init() as run:
      config=wandb.config
      ## Architecture

      # Define the model and optimizer

      ## a) Simple RecNN 
      if architecture=='simpleRecNN': 
        model = net.PredictFromParticleEmbedding(params,make_embedding=net.GRNNTransformSimple).cuda() if params.cuda else net.PredictFromParticleEmbedding(params,make_embedding=net.GRNNTransformSimple) 

      ##----
      ## b) Gated RecNN
      elif architecture=='gatedRecNN':
        model = net.PredictFromParticleEmbeddingGated(params,make_embedding=net.GRNNTransformGated).cuda() if params.cuda else net.PredictFromParticleEmbeddingGated(params,make_embedding=net.GRNNTransformGated) 

      ## c) Leaves/inner different weights -  RecNN 
      elif architecture=='leaves_inner_RecNN': 
        model = net.PredictFromParticleEmbeddingLeaves(params,make_embedding=net.GRNNTransformLeaves).cuda() if params.cuda else net.PredictFromParticleEmbeddingLeaves(params,make_embedding=net.GRNNTransformLeaves) 

      ##----
      ## d) Network in network (NiN) - Simple RecNN
      elif architecture=='NiNRecNN':
        model = net.PredictFromParticleEmbeddingNiN(params,make_embedding=net.GRNNTransformSimpleNiN).cuda() if params.cuda else net.PredictFromParticleEmbeddingNiN(params,make_embedding=net.GRNNTransformSimpleNiN)  

      ##-----
      ## e) Network in network (NiN) - Simple RecNN
      elif architecture=='NiNRecNN2L3W':
        model = net.PredictFromParticleEmbeddingNiN2L3W(params,make_embedding=net.GRNNTransformSimpleNiN2L3W).cuda() if params.cuda else net.PredictFromParticleEmbeddingNiN2L3W(params,make_embedding=net.GRNNTransformSimpleNiN2L3W)  

      ##-----
      ## f) Network in network (NiN) - Gated RecNN
      elif architecture=='NiNgatedRecNN':
        model = net.PredictFromParticleEmbeddingGatedNiN(params,make_embedding=net.GRNNTransformGatedNiN).cuda() if params.cuda else net.PredictFromParticleEmbeddingGatedNiN(params,make_embedding=net.GRNNTransformGatedNiN) 


      ##-----
      ## g) Network in network (NiN) -- NiN RecNN ReLU
      elif architecture=='NiNRecNNReLU':
        model = net.PredictFromParticleEmbeddingNiNReLU(params,make_embedding=net.GRNNTransformSimpleNiNReLU).cuda() if params.cuda else net.PredictFromParticleEmbeddingNiNReLU(params,make_embedding=net.GRNNTransformSimpleNiNReLU) 


      ##----------------------------------------------------------------------
      # Output number of parameters of the model
      pytorch_total_params = sum(p.numel() for p in model.parameters())
      pytorch_total_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)

      logging.info("Total parameters of the model= {}".format(pytorch_total_params))
      logging.info("Total weights of the model= {}".format(pytorch_total_weights))

      ##----------------------------------------------------------------------
      ## Optimizer and loss function

      logging.info("Model= {}".format(model))
      logging.info("---"*20)  
      logging.info("Building optimizer...")

      step_size=params.learning_rate
      decay=params.decay
      #   optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
      optimizer = optim.Adam(model.parameters(), lr=step_size)#,eps=1e-05)
      scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)

      # fetch loss function and metrics
      loss_fn = torch.nn.BCELoss()
      #   loss_fn = torch.nn.CrossEntropyLoss()
      metrics = net.metrics

      ##----------------------
      # Train the model
      logging.info("Starting training for {} epoch(s)".format(params.num_epochs))

      train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, args.model_dir, step_size,
                          args.restore_file)   

      elapsed_time=time.time()-start_time
      logging.info('Total time (minutes) ={}'.format(elapsed_time/60))

      for epoch in range(config["epochs"]):
        loss = loss_fn
        wandb.log({"loss":loss,"epoch":epoch})
  count = 5 
  wandb.agent(sweep_id, function=train,count=count)