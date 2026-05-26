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

The scripts and data provided in this repository allow viewers to replicate the findings found in our paper.

### Cloning, Training (on your local machine):

```
git clone https://github.com/AnishC10/PPFI-DNA.git
cd PPFI-DNA
git lfs install
git lfs pull
pip3 install -r requirements.txt
cd models
cd [TFHE, CKKS]
python3 [MODEL].py
```
Once you finish the final step, a dialogue will appear in the terminal, asking which dataset you would like to run on. If you respond with "Promoters" (or "promoters"), the promoter dataset will automatically be used. If you respond with "CVI," the CVI dataset will automatically be used. 

It is **highly** recommended to utilize an external server, rather than a local machine to run these programs.

