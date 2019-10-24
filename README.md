# stvae-source
[![Build Status](https://travis-ci.org/NRshka/stvae-source.svg?branch=master)](https://travis-ci.org/NRshka/stvae-source)

This repository contains the code for training and evaluating adversarial style transfer models on gene expression data from paper "Style transfer with variational autoencoders is a promising approach to RNA-Seq data harmonization and analysis" by  N. Russkikh, D. Antonets,  D. Shtokalo, A. Makarov, A. Zakharov, E. Terentyev. The paper is accessible by link
 https://t.co/sbjHbSzocn 
 
The main script is run.py, which contains both training and testing procedures. Testing includes transferring the style of test set expression data to all possible style categories and evaluating the accuracy of prediction of style and pre-defined semantic categories by MLP trained on the raw expression data.

Training hyperparameters can be set in config.py file
 
