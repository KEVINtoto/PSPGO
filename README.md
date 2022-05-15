# PSPGO: cross-species heterogeneous network propagation for protein function prediction

This is the code repository for protein automatic function prediction model PSPGO. 

**PSPGO** is a cross-species heterogeneous network propagation model, which can **P**ropagate feature and label information on **S**equence similarity (SS) network and **P**PI network for predicting **G**ene **O**ntology terms. The model is evaluated on a large multi-species dataset split based on time, and is compared with several state-of-the-art methods. The results show that PSPGO not only has good performance on multi-species test set but also performs well in prediction for single species.

<div align=center><img width="300" alt="image" src="https://user-images.githubusercontent.com/34743589/168454793-1445c76b-cd5c-47a7-b345-08fb7dd49e54.png"></div>

# Dependencies
* The code was developed and tested using python 3.8.
* To install python dependencies run: `pip install -r requirements.txt`. Some libraries may need to be installed via conda.
* The version of CUDA is `cudatoolkit==11.3.1`

# Data
<img width="147" alt="截屏2022-05-15 11 04 00" src="https://user-images.githubusercontent.com/34743589/168455604-1747f347-e0ac-42d2-b1a2-3cbb2b5abc08.png"><img width="181" alt="截屏2022-05-15 11 03 18" src="https://user-images.githubusercontent.com/34743589/168455610-a16fe1d5-00cc-4b58-ad64-07b86b0de58e.png"><img width="152" alt="截屏2022-05-15 11 06 14" src="https://user-images.githubusercontent.com/34743589/168455612-a71213d2-3ee4-4453-aceb-7693255e8383.png"><img width="188" alt="截屏2022-05-15 11 06 35" src="https://user-images.githubusercontent.com/34743589/168455614-907665c4-b16a-4bdb-9597-4814f772be21.png">

The protein data used are:
* Sequence: download from the (UniProt website)[https://www.uniprot.org/].
* PPI Network: download from the (STRING website)[https://string-db.org/].
* Annotation: download from the (GOA website)[https://www.ebi.ac.uk/GOA/].
* Gene Ontology: download from the (GO website)[http://geneontology.org/].
For a detailed description of data files, please see [here](data/README.md).
