# Code for my Bachelor thesis
Code of the paper "A Practical Investigation of Connections Between Contrastive Learning and Spectral Decomposition"

This is an extension of the repository [Kernel-InfoNCE](https://github.com/yifanzhang-pro/Kernel-InfoNCE), which is the implementation of the paper "Contrastive Learning Is Spectral Clustering On Similarity Graph" (https://arxiv.org/abs/2303.15103) and uses the work of [Pytorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) and [Lightning Bolts](https://github.com/Lightning-Universe/lightning-bolts/tree/master). 


## Installation 

Run `pip install -r requirements.txt` to install all necessary packages. Notice that a cuda 12 environment is needed for some of the scripts, and generally cuda is recommend to boost the performance.



## Preparation for Reproduction
Generally have a look at all scripts for the available argument options, the default arguments are used in all experiments of the thesis.

### 1. Dataset and Similarity graph creation
Before starting to train models, the dataset and the matching similarity graph(s) have to be created. 

To create the custom pre-computed dataset with the augmented views, which is used for the training and the embedding calculation run the following command in the `dataset` folder: 

```shell 
python tiny_mnist_aug_generator.py
```

The created dataset will be stored under `dataset/mnist_subset/chunks` in chunks. 

Similarly you can create the similarity graph adjacency matrix by running the following command in the `matrices` folder: 

```shell 
python similarity_graph_generator.py --block_type="pair"
```
for a pair-wise similarity graph or: 
```shell 
python similarity_graph_generator.py --block_type="aug_group"
```
for a augmentation-group similarity graph. 

The adjacency matrix and normalized adjacency matrix of the graph will be stored under `matrices/generated/`.

### 2. Model training and embedding computation
Generally wandb is used for logging all runs, therefore create a wandb account and login as described [here](https://docs.wandb.ai/quickstart/).

There exists two options for the training process: 

1) Run the `tiny_train.py` (pre-training), `tiny_evaluation.py` (linear evaluation) and `embeddings.py` (embedding computation) scripts individually 
2) **Recommended**: Run the `tiny_full_train.py` script to automatically run the pre-training, evaluation and embedding computation. 

Have a look at the script for the concrete argument options, as an example a SimCLR model (with the NT_Xent loss) can be trained by running: 

```shell
python tiny_full_train.py --loss_type="nt_xent"
```

This trains a model and stores the checkpoint under `results/pretraining/<wandb-run-name>`, runs linear evaluation and stores the checkpoint under `results/eval/<wandb-run-name>` and computes the embeddings given the custom dataset and stores them under `results/embeddings/<wandb-run-name>/chunks`.

> **_NOTE:_** Running `./configs/training_all_losses.sh` trains all necessary models used in all experiments.

### 3. Eigenvector matrix computation 

To compute the eigenvector matrix for the computed normalized adjacency matrix of the similarity graph run: 

```shell 
python eigenvector.py
```
This stores the eigenvector matrix for the similarity graph under the storage path of the graph.

## Reproduce the experiment results 

Generally for all files you need to define the input names, paths and the path suffix. Both name and suffix are used for identification, we recommend e.g. `--path_suffix="rq-nt` when comparing a model trained with the RQMin loss and a model trained with the NT-Xent loss. 

### 1. Ridge regression 
Ridge regression is used to calculate a transformation matrix between two embedding matrices (or eigenvector matrix) and can be run by: 

```shell 
python ridge_regression.py --matrix1_name="<...>" --matrix2_name="<...>" --matrix1_path="<...>" --matrix2_path="<...>" --path_suffix="<...>"
```

The computed transformation matrix and the run configuration are stored under: `results/ridge-regression/<timestamp>-matrices_<path_suffix>`.

### 2. Distance & Similarity Measures 

Make sure to run the ridge regression beforehand, as the transformation matrix is needed to compute both the normed euclidean distance and cosine similarity. 

Both metrics can be computed by running: 

```shell 
python distance_measures.py --matrix1_name="<...>" --matrix2_name="<...>" --matrix1_path="<...>" --matrix2_path="<...>" --matrix1_transformation_path="<...>"  --path_suffix="<...>"
```

The computed metrics with the run configuration and visualizations are stored under: `results/distances/<timestamp>_measures_<path_suffix>`.

### 3. CKA 

CKA can be computed in two ways, either for all layers given two trained model checkpoints or for the embeddings matrices. The latter approach can is the only one available when comparing something with an eigenvector matrix.

#### Model 

To compute the CKA similarity for all layers and also visualize the similarity in form of heatmap run: 

```shell
python cka.py  --model1_name="<...>" --model2_name="<...>" --model1_path="<...>" --model2_path="<...>" --path_suffix="<...>"
```
given the checkpoints from pre-training. The results are stored under: `results/cka/<timestamp>_cka_<path_suffix>`

#### Matrix 

To compute the CKA similarity between the embeddings or eigenvector matrices run: 

```shell
python cka_matrices.py  --matrix1_name="<...>" --matrix2_name="<...>" --matrix1_path="<...>" --matrix2_path="<...>" --path_suffix="<...>"
```
given the embedding (or eigenvector) matrices. The results are stored under: `results/cka/<timestamp>_cka-matrices_<path_suffix>`


### 4. t-SNE visualization 

To visualize a seeded subset of the embeddings or the eigenvector matrix with t-SNE run: 

```shell
python tsne.py  --matrix_name="<...>" --matrix_path="<...>" --path_suffix="<...>"
```

The t-SNE results and a visualization is stored under: `results/tsne/<timestamp>_<path_suffix>`


## Appendix experiments

### 1. Bootstrapping 
To reproduce the bootstrapping results simply run the `distance_measures.py` script with the additional argument `--bootstrapping_iterations=1`. 

### 2. Constraints of RQMin loss 
You can disable individual constraints when running the training with option `--constrained_rqmin` by adding `--orthogonal_constrained` (disables orthogonality constraint) or `--centering_constrained` (disables centering constraint)

### 3. Constraining other losses 
Simply run the shell script `./configs/training_all_losses_constrained.sh`, which trains models for all losses (except RQMin) with constraints. 

### 4. Regularization term of LGMin loss 
Again simply run the shell script `./configs/training_lgmin_appendix.sh`, which trains all shown parameter variations. 

### 5. Experimental loss 
The model can be optimized with the experimental loss by running `python tiny_full_train.py --loss_type="experimental_trace"`

### 6. Overlap spectral decomposition   
You can create similarity graphs with overlap by running `python matrices/overlap_matrix_generator.py`, have a look at the arguments to define the SSL-based construction as well as the ratio and density.
