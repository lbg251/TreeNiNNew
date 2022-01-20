import numpy as np
import torch
import pprint
import sys
import pickle
import argparse
import logging
import os
import random
import time


import sys
# sys.path.append("/Users/laurengreenspan/GitDLs/ginkgo/src/")
# sys.path.append("/Users/laurengreenspan/GitDLs/ClusterTrellis/src")
sys.path.append("/scratch/lbg251/GitDLs/ginkgo/src")
sys.path.append("/scratch/lbg251/GitDLs/ClusterTrellis/src")
from ginkgo import invMass_ginkgo
#logger = get_logger(level = logging.WARNING)
parser = argparse.ArgumentParser()
parser.add_argument("--job_num", help="The number of the job that's running", type=int)
parser.add_argument("--Nsamples", help="Number of Generated Jets", type=int)
args = parser.parse_args()

from ginkgo import likelihood_invM as likelihood

#logger = get_logger(level=logging.WARNING) 

i = args.job_num
#i=1

params = {}
minLeaves = 1
maxLeaves = 15
Nsamples = args.Nsamples
#Nsamples = 10
maxNTry = 20000

QCD_mass = 14.#TeV, from Top Tagging paper 
M2start = torch.tensor(QCD_mass)**2
jetM = np.sqrt(M2start.numpy())

jetdir = np.array([1,1,1])
jetP = random.randint(550,650) #GeV, from Top Tagging Paper
jetvec = jetP * jetdir / np.linalg.norm(jetdir)

jet4vec = np.concatenate(([np.sqrt(jetP**2 + jetM**2)], jetvec))

pt_min = 36.

tree_rate = random.uniform(1,5)
root_rate =random.uniform(1,5)

params['Nsamples']=Nsamples
params['maxLeaves']=maxLeaves
params['jetP']=jetP
params['jetM']=jetM
params['pt_min']=pt_min
params['tree_rate']=tree_rate
params['root_rate']=root_rate

pickle.dump(params, open("/Users/laurengreenspan/GitDLs/TreeNiNNew/code_ginkgo/data/raw_trees/params"+str(i)+".pkl","wb"), protocol = 2)
simulator = invMass_ginkgo.Simulator(jet_p=jet4vec,
                                     pt_cut=float(pt_min),
                                     Delta_0=M2start,
                                     M_hard=jetM ,
                                     num_samples=Nsamples,
                                     minLeaves =minLeaves,
                                     maxLeaves = maxLeaves,
                                     maxNTry = maxNTry)

#### QCD-like jets
rate_QCD=torch.tensor([tree_rate,tree_rate]) #Entries: [root node, every other node] decaying rates. Choose same values for a QCD jet

QCD_jet_list = simulator(rate_QCD)
for count,value in enumerate(QCD_jet_list):
    p = ((value['content'][0,0:3]**2).sum())**0.5
    px = value['content'][0,1]
    py = value['content'][0,2]
    pz = value['content'][0,3]    
    pT = np.asarray([np.linalg.norm(const[1:3]) for const in value["leaves"]])   
    eta = 0.5 * (np.log(p+pz) - np.log(p-pz))
    phi = np.arctan2(py,px)
    
    value['label']=0
    value['pt']=pT
    value['root_id']=QCD_jet_list[count]['root_id']
    value['energy']=QCD_jet_list[count]['content'][0,0]
    value['tree']=QCD_jet_list[count]['tree']
    value['content']=QCD_jet_list[count]['content']##Ordering depends on pytorch or tf implementation
    value['eta']=eta
    value['mass']=QCD_jet_list[count]['M_Hard']
    value['phi']=phi
    value['algorithm']= 'truth'
    
#### W-like jets
rate_W=torch.tensor([tree_rate,root_rate]) #Entries: [root node, every other node] decaying rates. Choose same values for a QCD jet

#### restructure dictionary to align with TreeNiN input format
W_jet_list = simulator(rate_W)
for count,value in enumerate(W_jet_list):
    p = ((value['content'][0,0:3]**2).sum())**0.5
    px = value['content'][0,1]
    py = value['content'][0,2]
    pz = value['content'][0,3]    
    pT = np.asarray([np.linalg.norm(const[1:3]) for const in value["leaves"]])   
    eta = 0.5 * (np.log(p+pz) - np.log(p-pz))
    phi = np.arctan2(py,px)
    
    value['label']=1
    value['pt']=pT
    value['root_id']=W_jet_list[count]['root_id']
    value['energy']=W_jet_list[count]['content'][0,0]
    value['tree']=W_jet_list[count]['tree']
    value['content']=W_jet_list[count]['content']##Ordering depends on pytorch or tf implementation
    value['eta']=eta
    value['mass']=W_jet_list[count]['M_Hard']
    value['phi']=phi
    value['algorithm']= 'truth'


jets_list = QCD_jet_list + W_jet_list
pickle.dump(jets_list, open("/Users/laurengreenspan/GitDLs/TreeNiNNew/code_ginkgo/data/raw_trees/ginkgo_jets"+str(i)+".pkl","wb"), protocol = 2)
