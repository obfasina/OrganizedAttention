# generic package imports
import argparse
import numpy as np
import os,sys
import pickle

# specify paths 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
sys.path.append(parent_dir + "/Julia_allcodes_plus_dependencies/pyquest-master")
import diffusion_maps, organize_atten


# Step 0) Specify CLI arguments
parser = argparse.ArgumentParser(description='Specify network inference commands')
parser.add_argument('data',type=str,help='location of data tensor')
parser.add_argument('--nbas',type=int,help='number of top basis vectors to select',default=10)
args = parser.parse_args()

# Step 1) Instantiate and run 3D questionnaire
datapath = args.data
X = np.load(datapath)
orgobj = organize_atten.organize_heads(X)

# Organize Data
iralph = 0
irbeta = 1 
icalph = 0
icbeta = 1
idalph = 0
idbeta = 1
initers = 3

orgobj = organize_atten.organize_heads(X)
orgobj.init_quest_params(iralph,irbeta,icalph,icbeta,idalph,idbeta,initers)
orgobj.run_quest()
orgobj.run_diffusion()


# Step 2) Save qualitative results

#qualdict = {'diff_coord':orgobj,'head_affinity':orgobj.chan_aff}
with open("network_obj.pkl", "wb") as f:
    pickle.dump(orgobj, f)

# Step 3) Save quantitative results

topk = args.nbas
entp = np.sum(orgobj.compute_ntwk_entp(topk))
print(f"The l1 entropy of the network computed with the top {topk} tensor basis vectors is: {entp}")








