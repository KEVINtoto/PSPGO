# PSPGO: cross-species heterogeneous network propagation for protein function prediction

Here is the code repository for protein automatic function prediction model PSPGO. 

**PSPGO** is a cross-species heterogeneous network propagation method, which can **P**ropagate feature and label information on **S**equence similarity (SS) network and **P**PI network for predicting **G**ene **O**ntology terms. The model is evaluated on a large multi-species dataset split based on time, and is compared with several state-of-the-art methods. The results show that PSPGO has good performance on multi-species test set and also performs well in prediction for single species.

<div align=center><img width="300" alt="image" src="https://user-images.githubusercontent.com/34743589/168454793-1445c76b-cd5c-47a7-b345-08fb7dd49e54.png"></div>

# Dependencies
* The code was developed and tested using python 3.8.
* To install python dependencies run: `pip install -r requirements.txt`. Some libraries may need to be installed via conda.
* The version of CUDA is `cudatoolkit==11.3.1`
