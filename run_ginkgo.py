import numpy as np
import torch
import pprint
import sys
import pickle
import argparse
import logging
import os
import random
import sys
sys.path.append("/scratch/lbg251/GitDLs/ginkgo/src")
from ginkgo import invMass_ginkgo
from ginkgo.utils import get_logger

logger = get_logger(level = logging.WARNING)
parser = argparse.ArgumentParser()
parser.add_argument("job_num", help="The number of the job that's running", type=int)
args = parser.parse_args()
         
        
class ginkgo_simulator():
    def __init__(self,rate,pt_cut,M2start,Nsamples,minLeaves,maxLeaves,maxNTry,jetType,jetP,root_rate
                ):
        self.root_rate = root_rate
        self.rate = rate
        self.pt_cut = pt_cut
        self.M2start = torch.tensor(M2start)
        self.Nsamples = Nsamples
        self.minLeaves = minLeaves
        self.maxLeaves = maxLeaves
        self.maxNTry = maxNTry
        self.jetType = jetType
        self.jetM = np.sqrt(M2start)
        self.jetdir = np.array([1,1,1])
        self.jetP = jetP
        self.jetvec = self.jetP * self.jetdir/np.linalg.norm(self.jetdir)
        self.jet4vec = np.concatenate(([np.sqrt(self.jetP**2+self.jetM**2)],self.jetvec))
        logger.debug(f"jet4vec = {self.jet4vec}")
    
        if jetType == "W":
            self.rate = torch.tensor([self.root_rate,self.rate])
        ##for W jets. Entries: [root node, every other node] decaying rates. Choose same values for a QCD jets
    
        elif jetType == "QCD":
            self.rate = torch.tensor([self.rate,self.rate])
    
        else: 
            raise ValueError("Choose a valid jet type between W and QCD")
        
    def simulator(self):
        simulator = invMass_ginkgo.Simulator(jet_p = self.jet4vec, pt_cut = float(self.pt_cut), Delta_0 = self.M2start, M_hard = self.jetM, num_samples = int(self.Nsamples), minLeaves = int(self.minLeaves), maxLeaves = int(maxLeaves), maxNTry = int(self.maxNTry))
        
        return simulator
    
    def generate(self):
        
        simulator = self.simulator()
        jet_list = simulator(self.rate)
        
        logger.debug(f"---"*10)
        logger.debug(f"jet_list = {jet_list}")
        
        return jet_list
            
        
            
    


i = args.job_num

minLeaves = 1
maxLeaves = 60
Nsamples = 10000
maxNTry = 20000

jetP = random.randint(550,650) #GeV, from Top Tagging Paper
QCD_mass = 14.#TeV, from Top Tagging paper 
M2start = torch.tensor(QCD_mass)**2


pt_cut = 36.

rate = random.uniform(1,5)
W_rate =random.uniform(1,5)

params = [Nsamples, jetP, M2start, pt_cut, rate, W_rate]

pickle.dump(params, open("/scratch/lbg251/GinkgoTrellis/JetTree/data/params"+str(i)+".pkl","wb"), protocol = 2)

jetType = "QCD"

ginkgo = ginkgo_simulator(rate,pt_cut,M2start,Nsamples,minLeaves,maxLeaves,maxNTry,jetType,jetP,root_rate = rate)
QCD_jets = ginkgo.generate()
    
pickle.dump(QCD_jets, open("/scratch/lbg251/GinkgoTrellis/JetTree/data/QCD_jets"+str(i)+"_jetp_"+str(jetP)+"_lambda_"+str(round(rate,3))+"_ptcut_"+str(round(float(pt_cut),3))+".pkl","wb"), protocol = 2)

jetType = "W"

ginkgo = ginkgo_simulator(rate,pt_cut,M2start,Nsamples,minLeaves,maxLeaves,maxNTry,jetType,jetP,root_rate = W_rate)
W_jets = ginkgo.generate()
    
pickle.dump(W_jets, open("/scratch/lbg251/GinkgoTrellis/JetTree/data/W_jets"+str(i)+"_jetp_"+str(jetP)+"_lambda_"+str(round(rate,3))+"_ptcut_"+str(round(float(pt_cut),3))+"W_rate"+str(W_rate)+".pkl","wb"), protocol = 2)
    
    
