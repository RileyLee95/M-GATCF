# M-GAT (Modified-Graph Attention Networks)

## Introduction 
This project is my MSc final project and it aimed at studying and improving on a specific GNN-based CF recommender model named NFCF proposed by Wang et al, 2020.  (Note: The Neural Graph Collaborative Filtering (NGCF) model proposed in the paper by Wang et al. is state-of-the-art GNN-based CF model for implicit feedback-based recommendation scenarios)



## Motivation
Although the NGCF model has achieved significant improvement over several state-of-the-art competitors with the help of GNN techniques, we may argue that the original paper failed to clearly state the design motivation of NGCF, nor to conduct sufficient comparative experiments to prove the optimality of each component in NGCF. Motivated by this gap, we first conducted a systematic anatomy of NGCF, whereby identified these core components and design decisions in it. Then we attempted to improve NGCF from those key design aspects identified correspondingly. 

## Outcomes
Experimental results confirmed that we successfully improved the performance (Recall@20) of NGCF by nearly 4.5% and 3% (on original benchmark datasets as the original paper) through adopting a different aggregation rule (M-GAT) and updating rule (context-only updating rule) respectively. 
This project is our early attempt in the GNN-based recommendation area, we hope to further improve our findings by integrating auxiliary information as Knowledge Graphs (KG) in the future.

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:

- tensorflow == 1.8.0
- numpy == 1.14.3
- scipy == 1.1.0
- sklearn == 0.19.1

## Guideline to re-run models
- I run all experiments on the Golab
- All arguments for training different models are included in the "M_GATCF_notebook" nootbook, it's self- explained. 

## Acknowledgement
All our implementations are based on the open-source project  https://github.com/xiangwang1223/neural_graph_collaborative_filtering . We also referred to the official implementation of GATs (Graph Attention Networks) https://github.com/PetarV-/GAT when implementing the M-GAT aggregating layer.
