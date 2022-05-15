# PSPGO: cross-species heterogeneous network propagation for protein function prediction

This is the code repository for protein automatic function prediction model PSPGO. 

**PSPGO** is a (~~Protein Station Portable for GO~~) cross-species heterogeneous network propagation model, which can **P**ropagate feature and label information on **S**equence similarity (SS) network and **P**PI network for predicting **G**ene **O**ntology terms. The model is evaluated on a large multi-species dataset split based on time, and is compared with several state-of-the-art methods. The results show that PSPGO not only has good performance on multi-species test set but also performs well in prediction for single species.

<div align=center><img width="400" alt="network" src="https://user-images.githubusercontent.com/34743589/168456432-0fb12024-9997-4cdc-894d-966c4cf15328.png"></div>

## Dependencies
* The code was developed and tested using python 3.8.
* To install python dependencies run: `pip install -r requirements.txt`. Some libraries may need to be installed via conda.
* The version of CUDA is `cudatoolkit==11.3.1`

## Data
<div align=center><img width="147" alt="uniprot" src="https://user-images.githubusercontent.com/34743589/168455684-0cc53a92-874e-4c2e-9c2a-dcfbd36fb798.png"><img width="175" alt="string" src="https://user-images.githubusercontent.com/34743589/168455685-368d0af3-5b41-4ba8-8f02-36bfcbcd3a24.png"><img width="143" alt="goa" src="https://user-images.githubusercontent.com/34743589/168455693-246a738e-b04b-4496-a632-afbeb36d239e.png"><img width="206" alt="go" src="https://user-images.githubusercontent.com/34743589/168455695-c733fbf1-fcab-4cc5-92e2-272fd7abe88b.png"></div>

\
The data used are:
* Sequence: download from the [UniProt website](https://www.uniprot.org/).
* PPI Network: download from the [STRING website](https://string-db.org/).
* Annotation: download from the [GOA website](https://www.ebi.ac.uk/GOA/).
* Gene Ontology: download from the [GO website](http://geneontology.org/).

For a detailed description of data files, please see [here](data/README.md).

## Usage
Run the `train.py` script to train the model (e.g. for BPO):
```
python train.py --ontology bp
```
Run the `predict.py` script to make predictions about the input file (e.g. for MFO):
```
python predict.py --ontology mf -f ${input file path}
```
If no input file is specified, the test data from the corresponding ontology will be used for prediction by default.
\
\
For evaluation of model, run the `evaluate.py` script (e.g. for CCO):
```
python evaluate.py --ontology cc
```

