# Privacy Preserving Federated Inference for Genomic Analysis with Homomorphic Encryption

## Overview
This work presents the first full framework linking **fully homomorphic encryption** with **federated analytics**, in the context of DNA nucleotide data. Users can train using the code in this repository, then perform encrypted inference. 

## How to Cite this Work
The preprint can be accessed [here](https://ia.cr/2025/1515). The citation information for the paper is shown below. 
```
@misc{chakraborty_tsoutsos_2025,
      author = {Anish Chakraborty and Nektarios Georgios Tsoutsos},
      title = {Privacy-Preserving Federated Inference for Genomic Analysis with Homomorphic Encryption},
      howpublished = {Cryptology {ePrint} Archive, Paper 2025/1515},
      year = {2025},
      url = {https://eprint.iacr.org/2025/1515}
}
```
## Usage

The scripts and data provided in this repository allow viewers to replicate the findings found in our paper (preprint coming soon!).

### Cloning, Training (on your local machine):

```
git clone https://github.com/AnishC10/PPFI-DNA.git
cd PPFI-DNA
pip3 install -r requirements.txt
cd models
python3 encrypted_neural_network.py
```
This will begin training and testing on the data in ```data.csv```, and then output the resulting statistics.   

