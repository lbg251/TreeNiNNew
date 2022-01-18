import os, sys

num_jets = 2000

os.chdir('code_ginkgo/')
# for i in range(10):
#     os.system('python make_ginkgo.py  --Nsamples=100 --job_num='+str(i))

#algo = ["kt","antikt","ptdesc","truth"]
algo=['truth']
for j in range(len(algo)):
    # os.system('python set_topology.py '+str(algo[j]))

    os.chdir('/Users/laurengreenspan/GitDLs/TreeNiNNew/code_ginkgo/recnn')
    # file_name ='ginkgo_'+str(algo[j])+'_'+str(num_jets)+'jets.pkl'
    # os.system('python make_batches.py --jet_algorithm='+str(algo[j])+' --in_filename='+str(file_name))
# # Load the trained weights and evaluate each model 
    os.system('python search_hyperparams.py --jet_algorithm='+str(algo[j])+' --NrunStart=0 --NrunFinish=1')