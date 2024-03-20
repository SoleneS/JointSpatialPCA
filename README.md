# JointSpatialPCA and clustering

Workflow for the paper "Single-cell morphometrics reveals T-box dependent patterns of epithelial tension in the Second Heart field"

There is an input folder with the data of the paper, and an output folder with the output corresponding to the paper's input. The workflow is constituted of 3 steps in python, and one step in matlab that calls the function from Cardoso et al. "Jacobi angles for simultaneous diagonalization"  for joint matrix diagonalization.

Step0.py 
builds the criterion matrices M_k for k samples from the training data (in our paper, it's the condition E9WT embryos). M_k=(1/2n)X^T(L+L^T)X, where X is the features matrix and L is the row normalized adjacency matrix, and n the number of nodes. It concatenates them as a single matrix M_cat that will be used as the input of the matlab function of Cardoso et al.

Step1_call_joint_diagonalization.m
is a matlab function that calls the Cardoso et al. function for joint diagonalization (joint_diag.m). Its outputs are the common eigenvectors V.mat and the eigenvalues for each sample, summarized in all_diags.mat

Step2.py 
buids the transformed coordinates on the first and second joint spatial PC directions. It performs gaussian mixture clustering on jsPC1. It calculates the Aikake score and Bayesian inference scores (aic and bic) for number of clusters n=1 to n=9. The user chooses manually the number of clusters after checking these values. Then the script runs this gaussian mixture clustering multiple times (nb_iter) to check the robustness of the clustering upon random initialization

Step3.py
Now that robustness has been checked and number of clusters decided, this script runs the gaussian mixture on jsPC1 with a seed for initialization. It plots the histograms of the jsPC1 coordinates, as well as the fitted gaussian curves. Then, it uses the trained gaussian mixture model to cluster the cells from all other conditions. 
The final output is df_labels.pkl, with all the labels for all embryos.
