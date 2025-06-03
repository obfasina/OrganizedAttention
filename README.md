
# Organized Attention

Included are implementations of:
1) the questionnaire algorithm
2) the paraproduct decomposition
3) examples with the transformer-xl and vit b-16 networks

The 2D questionnaire algorithm organizes a matrix to be mixed Holder, allowing one to subsequently apply the paraproduct decomposition.
The 3D questionnaire algorithm is used for network 3-tensor organization; one can then compute expansion coefficients of the network 3-tensor with respect to the Haar bases supported on the indices of the tensor axes.
'txl_vit_paraproduct_decompositions.ipynb' is an example of the paraproduct decompostion implementation for the transformer-xl and vit b-16 networks
'txl_vit_tensor_organizations.ipynb' is an example notebook for organization of the attention heads for the transformer-xl and vit b-16 networks

NOTES: 
a) The questionnaire algorithm used here was originally implemented by Ankenmann/Mishne
b) For the transformer-xl attention head generation, we only include the scripts modified from the original transformer-xl model

# Requirements

All packages required to run 'network_inference.py' can be found in the /src/network_inference.yml file

# How to run

We have provided a script that generates the l1 entropy of the input network and an object containing the diffusion coordinates, the location of the query, key and head indeces in each tree axis, and pair-wise affinities on the space of queries,keys, and heads.

The script can be found in /src and is run with 'python /path/to/data'. One can also optionally specify the number of tensor basis vectors used to compute the l_1 entropy of the network. This option is specified as 'python /path/to/data --nbas k' where k is the number of desired basis vectors (it is an optional argument, the default value is 10) .

