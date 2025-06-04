
# Organized Attention

Included are implementations of:
1) the questionnaire algorithm (originally implemented by Gal Mishne: [3Dquest](https://github.com/gmishne/pyquest)) 
2) the paraproduct decomposition
3) examples with the transformer-xl and vit b-16 networks

## Package Notes
- The 2D questionnaire algorithm organizes a matrix to be mixed Holder, allowing one to subsequently apply the paraproduct decomposition.
- The 3D questionnaire algorithm is used for network 3-tensor organization; one can then compute expansion coefficients of the network 3-tensor with respect to the Haar bases supported on the indices of the tensor axes.
- 'txl_vit_paraproduct_decompositions.ipynb' is an example of the paraproduct decompostion implementation for the transformer-xl and vit b-16 networks
- 'txl_vit_tensor_organizations.ipynb' is an example notebook for organization of the attention heads for the transformer-xl and vit b-16 networks
- For the transformer-xl attention head extraction, we only include scripts which were modified from the original transformer-xl code which is found here: [transformer-xl](https://github.com/kimiyoung/transformer-xl)

# Requirements

All packages required to run 'network_inference.py' can be found in the /src/network_inference.yml file

# How to run

We provide a script which generates:
- the l1 entropies of each of the attention heads for the input network (admited by query-key tensor basis) and the network entropies (admitted by query-key-head tensor basis)
- the diffusion coordinates
- pair-wise affinities on the space of queries,keys, and heads.

The script can be found in the /src directory and can be run with:
**'python network_inference.py /path/to/data'**.
One can also optionally specify the number of tensor basis vectors used to compute the l_1 entropy of the network. 
This can be run in the CLI with: **'python network_inference.py /path/to/data --nbas k'** where k is the number of desired basis vectors (it is an optional argument, the default value is 10).

