#!/usr/bin/env python
# USAGE

#------------------------------------------------------------------------------------------
# Enable python for fastjet
# [macaluso@hexcms fastjet-3.3.1]$ ./configure --prefix=$PWD/../fastjet-install --enable-pyext
# 
#   make 
#   make check
#   make install
#   cd ..
#   
#------------------------------------------------------------------------------------------
# PYTHONPATH
# There should be two ways:
# 1) tell python where to look: for example at the beginning of my code I have this:
# sys.path.append("/opt/fastjet-install/lib/python2.7/site-packages")
# After this one can do "import fastjet"
# 
# 2) append the fj install path to the PYTHONPATH global variable. Execute something like this in the shell (if you want it to be permanent, add to your ~/.cshrc file)
# setenv PYTHONPATH ${PYTHONPATH}:/opt/fastjet-install/lib/python2.7/site-packages
# 

#------------------------------------------------------------------------------------------
from __future__ import print_function

# import sys
import sys, os, copy
os.environ['TERM'] = 'linux'
#pyroot module
import numpy as np  
# import scipy as sp
import random
# import matplotlib.pyplot as plt
random.seed(1)
# import itertools
# import ROOT as r
# import json 
import time
import pickle
# import ROOT as r

# import copy
# from rootpy.vector import LorentzVector
# from recnn.preprocessing import _pt

start_time = time.time()


# PYTHONPATH
#sys.path.append("/scratch/lbg251/environments/fastjet-3.4.0/../fastjet-install/lib/python3.9/site-packages")

sys.path.append("/Users/laurengreenspan/fastjet-install/lib/python3.8/site-packages")
import fastjet as fj

import analysis_functions as af
import preprocess_functions as pf
import tree_cluster_hist as cluster_h


print('Reclustering and making the jet trees ...')
print('==='*20)
#-----------------------------------------
plots_dir='plots/'
os.system('mkdir -p '+plots_dir)

images_plots_dir='images/'
os.system('mkdir -p '+plots_dir+'/'+images_plots_dir)

#-------------------------------------------------------------------------------------------------------------
#/////////////////////     FUNCTIONS     //////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------
class ParticleInfo(object):
    """class for use in assigning pythonic user information
    to a PseudoJet.
    """
    def __init__(self, type, PID=None, Charge=None, Muon=None):
        self.type = str(type)
        self.PID=PID
        self.Charge=Charge

    def set_PID(self, PID):
        self.PID = PID

    def set_Charge(self, Charge):
        self.Charge = Charge
        
    def set_Muon(self, Muon): #Muon label (yes=1 or no=0)
        self.Muon = Muon

    
#-------------------------------------------------------------------------------------------------------------



cardfile=sys.argv[1]
sampletype=sys.argv[2]  # train, val,test
# root_file=sys.argv[3]
# outfilestring=sys.argv[4]
dir_subjets= sys.argv[3]
# dir_subjets='../data/input/test_subjets/'
out_dir=sys.argv[4]

Rjet = .8
Rtrim = .3
jetdef_tree = kt

# Turn to true if preprocessing (shift, rotate, etc)
rot_boost_rot_flip=True
#rot_boost_rot_flip=False
#-------------------------------------------------------------------------------------------------------------
# #Read cardfile
# with open(cardfile) as f:
#    commands=f.readlines()

# commands = [x.strip().split('#')[0].split() for x in commands] 

# ptmin=-9999999.
# ptmax=9999999.
# maxeta=9999999.
# matchdeltaR=9999999.
# mergedeltaR=9999999.
# N_jets=np.inf
# # N_jets=100000
# # N_jets=100

# for command in commands:
#   if len(command)>=2:
#     if(command[0]=='TRIMMING'):
#        Trimming=int(command[1])
#     if(command[0]=='JETDEF'):
#        jetdef_tree=str(command[1])       
#     if(command[0]=='PTMIN'):
#        ptmin=float(command[1])
#     elif(command[0]=='PTMAX'):
#        ptmax=float(command[1])
#     elif(command[0]=='ETAMAX'):
#        etamax=float(command[1])
#     elif(command[0]=='MATCHDELTAR'):
#        matchdeltaR=float(command[1])
#     elif(command[0]=='MERGEDELTAR'):
#        mergedeltaR=float(command[1])
#     elif(command[0]=='RJET'): #Radius of the jet
#        Rjet=float(command[1])
#     elif(command[0]=='RTRIM'): #Radius for the subjets used for trimming
#        Rtrim=float(command[1])
#     elif(command[0]=='MINPTFRACTION'): #Min pT fraction for the subjets that pass the trimming filter
#        MinPtFraction=float(command[1])
#     elif(command[0]=='PREPROCESS'):
#        preprocess_label=command[1]
#        print('preprocess_label=',preprocess_label)
#     elif(command[0]=='MERGE'):
#        jetmergeflag=int(command[1])
#     elif(command[0]=='NPOINTS'):
#        npoints=int(command[1])
#     elif(command[0]=='DRETA'):
#        DReta=float(command[1])
#     elif(command[0]=='DRPHI'):
#        DRphi=float(command[1])   
#     elif(command[0]=='NCOLORS'):
#        Ncolors=int(command[1])
#     elif(command[0]=='KAPPA'):
#        kappa=float(command[1])
       
# preprocess_cmnd=preprocess_label.split('_')    



# print("ptmin",ptmin)
# print("ptmax",ptmax)
# print("etamax",etamax)
# print("matchdeltaR",matchdeltaR)
# print("mergedeltaR",mergedeltaR)


#counter for current entry
n=-1
# out_dir='../data/output/test_top_qcd/'
if not os.path.exists(out_dir):
  os.makedirs(out_dir)

# 
# #-------------------------------------------------------------------------------------------------------------

print('Loading files for subjets')
print('Subjet array format ([[[pTsubj1],[pTsubj2],...],[[etasubj1],[etasubj2],...],[[phisubj1],[phisubj2],...]])')
print('-----------'*10)

subjetlist = [filename for filename in np.sort(os.listdir(dir_subjets)) if (sampletype in filename and filename.endswith('.pkl'))]
# subjetlist = [filename for filename in np.sort(os.listdir(dir_subjets)) if ('subjets' in filename and eventtype in filename and 'nompi_5' in filename and filename.endswith('.dat'))]

N_analysis=len(subjetlist)
print('Number of subjet files =',N_analysis)
print('Loading subjet files...  \n {}'.format(subjetlist))

images=[]
jetmasslist=[]

Ntotjets=0

#-------------------------------------------------------------------------------------------------------------
#///////////////////////////////////////////////////////////////////
#------------------------------------------------------------------------------------------------------------- 
#------------------------------------------------------------------------------
counter=0 

## Loop over the data
jet_mass_difference=[]

for ifile in range(N_analysis):
print(myN_jets,Ntotjets)
#   outputfile=out_dir+'tree_'+subjetlist[ifile].split('.')[0]+'.pkl'
  
  if(Ntotjets>N_jets):
     break
     
  file=dir_subjets+subjetlist[ifile]
#   out_file = open('top_tag_reference_dataset/tree_list/tree_'+subjetlist[ifile].split('.')[0]+'.pkl', "wb")

#   print('out_file=',out_file)
  
  with open(file, "rb") as f: jets_file =pickle.load(f) 
#   print('jets_file[0][0]=',jets_file[0][0])
#   print('jets_file[0][0]=',jets_file[1][0])   
  
  jet_pT=[]
  jet_mass=[]
  
  reclustered_jets=[]
  
  #Loop over all the events
  for element in jets_file:
    
    event=pf.make_pseudojet(element[0])
    #print('Event const=',event)
    label=element[1]
    print('label=',label)

    # Recluster jet constituents
    out_jet = pf.recluster(event, Rjet,jetdef_tree)  
    print('length out_jet =',len(out_jet)) 
    print('Jets [m,pT,eta,phi,pz]=',[[subjet.m(),subjet.perp(),subjet.eta(),subjet.phi_std(),subjet.pz()] for subjet in out_jet])


    #-----------------------   
    # If preprocessing (shift, rot, etc)
    if rot_boost_rot_flip:
    
        # Keep only the leading jet. Then recluster jets in subjets of R=0.3
        R_preprocess=Rtrim
        subjets = pf.recluster(out_jet[0].constituents(), R_preprocess,jetdef_tree) 
    #     print('---'*20)
    #     print('Subjets [mass]=',[subjet.m() for subjet in subjets])
    #     print('Subjets [mass,pT,eta,phi,pz]=',[[subjet.m(),subjet.perp(),subjet.eta(), subjet.phi_std(),subjet.pz()] for subjet in subjets]) 
      
        #------
        # Preprocess the jet constituents 
        preprocessed_const_fj= pf.preprocess_nyu(subjets) 
    #     print('preprocessed_const_fj [pT,eta,phi,m,pz]=',[[const_fj.perp(),const_fj.eta(),const_fj.phi(),const_fj.m(),const_fj.pz()] for const_fj in preprocessed_const_fj])
        
    #     #--------------------------------
    #     # Cross-check: leading subjet should have eta=0 (pz=0), phi=0. 2nd leading one should have eta=0 (pz=0). The 3rd one should have pz>0
#         preprocessed_subjets = pf.recluster(preprocessed_const_fj, R_preprocess,'kt') 
    #     print(' Number of subjets=',len(preprocessed_subjets))
#         print('---'*20)
#         print('Subjets [mass]=',[subjet.m() for subjet in preprocessed_subjets])
#         print('Subjet mass difference=',np.asarray([subjet.m() for subjet in preprocessed_subjets])-np.asarray([subjet.m() for subjet in subjets]))
#         print('Subjets [mass,pT,eta,phi,pz]=',[[subjet.m(),subjet.perp(),subjet.eta(), subjet.phi_std(),subjet.pz()] for subjet in preprocessed_subjets])
    
    
        # Recluster preprocessed jet constituents 
        preprocessed_subjets = pf.recluster(preprocessed_const_fj, Rjet,jetdef_tree) 
#         print(' Number of subjets=',len(preprocessed_subjets))
#         print('Subjets [mass,pT,eta,phi,pz]=',[[subjet.m(),subjet.perp(),subjet.eta(), subjet.phi_std(),subjet.pz()] for subjet in preprocessed_subjets])   
#         print('Jet mass difference=',np.asarray([subjet.m() for subjet in preprocessed_subjets])-np.asarray([subjet.m() for subjet in out_jet])) 
#         jet_mass_difference.append(np.asarray([subjet.m() for subjet in preprocessed_subjets])-np.asarray([subjet.m() for subjet in out_jet]))
#         jet_mass_difference.append(preprocessed_subjets[0].m()-out_jet[0].m())
    
        out_jet=preprocessed_subjets
#         print('Preprocessed Subjets [mass,pT,eta,phi,pz]=',[[subjet.m(),subjet.perp(),subjet.eta(), subjet.phi_std(),subjet.pz()] for subjet in out_jet])  


    #-----------------------------------
    # #Create a dictionary with all the jet tree info (topology, constituents features: eta, phi, pT, E, muon label)
    jets_tree = pf.make_tree_list(out_jet)
#     print('jets_tree=',jets_tree)
    
    #Keep only the leading jet
    for tree, content, mass, pt in [jets_tree[0]]:
    
#       print('Content=',content)
      jet_pT.append(pt)
      jet_mass.append(mass)
      jet = pf.make_dictionary(tree,content,mass,pt)
    
#       print('jet dictionary=',jet)
      reclustered_jets.append((jet, label))
#       pickle.dump((jet, label), out_file, protocol=2)
      
      counter+=1

#     print('===='*20)
#     print('===='*20)
    
#     if counter>0:
#       break


#------------------------------------------------------------------------------
# print('Max jet_mass_difference with rot=',np.sort(jet_mass_difference)[-50::])

# print('reclustered_jets=',reclustered_jets)

#If preprocessing
if rot_boost_rot_flip:
  out_filename = str(out_dir)+'tree_'+subjetlist[ifile].split('.')[0]+'_'+str(counter)+'_R_'+str(R_preprocess)+'_rot_boost_rot_flip.pkl'
else:
#   out_filename = str(out_dir)+'tree_'+subjetlist[ifile].split('.')[0]+'_'+str(counter)+'.pkl'
  out_filename = str(out_dir)+'tree_'+subjetlist[ifile].split('.')[0]+'.pkl'


# SAVE OUTPUT FILE
print('out_filename=',out_filename)
with open(out_filename, "wb") as f: pickle.dump(reclustered_jets, f, protocol=2) 
    


print('counter=',counter)  
# sys.exit()

  
  
  
  
    
    
    
    
       
#-------------------------------------------------------------------------------------------------------------
# Make histograms: Input= (out_dir,data,bins,plotname,title,xaxis,yaxis)
hist_dir='plots/histograms/top_tag_reference_dataset/'
if not os.path.exists(hist_dir):
  os.makedirs(hist_dir)



