# Privacy Preserving Federated Inference for Genomic Analysis with Homomorphic Encryption

## Overview
This work presents the first full framework linking **fully homomorphic encryption** with **federated analytics**, in the context of DNA nucleotide data. Users can train using the code in this repository, then perform inference. 

##Usage
The scripts and data provided in this repository allow viewers to replicate the findings found in our paper (preprint coming soon!).

###Cloning, Training (on your local machine):
First, make sure that you are in an appropriate virtual environment. 
```
git clone https://github.com/AnishC10/PPFI-DNA.git
cd PPFI-DNA
pip3 install -r requirements.txt
cd models/encrypted_neural_network.py

```
This will begin training and testing on the data in ```data.csv```, and then output the resulting statistics.   

