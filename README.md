# M3IB
#################################################################################
This is the code for paper:  A Variational Multi-modal Information Bottleneck Model for Next-item Recommendation
#################################################################################
Requirement: 
python 3.6
tensorflow-gpu==1.14.0
#################################################################################
The main.py describes the hyper-parameter of the proposed method, which can be summaried as follows:
d: The dimension of embeddings.
F: The number of most frequent patterns.
K: Users’ memory capacity.
lamda: the trade-off coefficient in Eq.(2).
sVAE: M3IB-noM2PG
cVAE:  M3IB-noMM
'KL': M3IB
'no-KL': M3IB-noBN
