# M-GAT
This project aimed at studying and improving the GNN-based CF recommender model named NFCF proposed by Wang et al, 2020. 

The recently proposed Neural Graph Collaborative Filtering (NGCF) model has achieved significant improvement over several state-of-the-art competitors with the help of GNN techniques. Despite its success, we may argue that the original paper failed to clearly state the design motivation of NGCF, nor to conduct sufficient comparative experiments to prove the optimality of each component in NGCF. Motivated by this gap, we first conducted a systematic anatomy of NGCF, whereby identified these core components and design decisions in it. Then we attempted to improve NGCF from those key design aspects identified. 

Experimental results confirmed that we successfully improved the performance of NGCF by nearly 4.5% and 3% through adopting a different aggregation rule and updating rule respectively. This project is our early attempt in the GNN-based recommendation area, we hope to further improve our findings by integrating auxiliary information as Knowledge Graphs (KG) in the future.
