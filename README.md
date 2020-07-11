# XTREME with many source languages
This is a fork of the original XTREME [repository](https://github.com/googleresearch/XTREME) used for my [bachelor thesis](https://github.com/blazejdolicki/multilingual-analysis). We adjust the code to be able to use other source languages than English for three downstream tasks: UD POS, Panx (NER) and XNLI. Additionally, we make another dataset compatible with this benchmark - CLS+ (sentiment analysis).

## Setup
Clone this repo and follow installation instructions from the original repository.

## UD POS, Panx (NER) and XNLI
To train and evaluate models use the same commands as for the original repository (for example, `>> bash scripts/train.sh xlm-roberta-large udpos`). However, you can select training and testing languages by changing the `TRAIN_LANGS` and  `PRED_LANGS` variables in `train_udpos.sh`,`train_panx.sh` and `train_xnli.sh` depending on which task you want to run. By default, we use all available language for a given dataset.

## CLS+
To run experiments on CLS+ with the XLM-R (Large) model, execute:

`>> bash scripts/train_cls.sh xlm-roberta-large`
