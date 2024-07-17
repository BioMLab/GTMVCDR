## GTMVCDR

### GTMVCDR: Cancer drug response prediction with cross-modal multi-level feature learning mechanism.
### Datasets
 
 - GDSC has 568 drugs and 221 drugs with 103336 observed drug-cell line
   responses.  
### Description
 - smilegraph.py: generate molecular graphs for drugs.
 - cellgraph.py: generate cell line graphs.
 - similarity.py: construct similarity_matrix for drugs and cell lines.
 - GTMVCDR.py: our model.
 - main.py: train and test our model.

### Run Step
 1. Run smilegraph.py to generate molecular graphs for drugs.
 2. Run cellgraph.py to generate cell line graphs.
 3. Run datasetload.py to load related data for cancer drug response prediction.
 4. Run main.py to train the model and obtain the predicted scores for cancer drug response prediction.

### Requirements
 - Python == 3.7.0
 - Numpy == 1.21.6
 - Pandas == 1.2.3
 - Scikit-learn == 1.0.2
 - Scipy == 1.7.3
 - Seaborn == 0.12.2
 - Pytorch == 1.10.0
 - Torch-geometric == 2.2.0 
 - Rdkit == 2018.09.2
 - Torch-cluster == 1.5.9
 - Torch-scatter == 2.0.9
 - Torch-sparse == 0.6.12
 - Torch-spline-conv == 1.2.1
### Notation
9606.protein.links.detailed.v11.0.txt and data_new.npy are too large to upload. If you want to download them, please click https://pan.baidu.com/s/1JbL3jBE8TmBvAG_mhKa7cA?pwd=mo7n
### Citation
 If there is a requirement for you to reference the paper, code or dataset, please ensure to cite the source accurately.
