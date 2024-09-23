# DiffusPoll: Conditional Text Diffusion Model for Poll Generation

This repository provides the official implementation of the following paper: 

> [**DiffusPoll: Conditional Text Diffusion Model for Poll Generation**.](https://aclanthology.org/2024.findings-acl.54/) (ACL 2024) <br>



## 1. Datasets
Datasets included in the ``./datasets/`` folder. We make our dataset from [WeiboPolls](https://github.com/polyusmart/Poll-Question-Generation/tree/main/data/Weibo).

`*-key, *-topic` means the ablations with different data after by the attribute extractor. 

## 2. Requirements

```bash 
pip install -r requirements.txt 
```

## 3. Training
The training script is launched in the ``scripts`` folder.
```bash
cd scripts
bash train.sh
```
Arguments explanation:
- ```--dataset```: WeiboPolls datasets, mentioned above
- ```--div_loss```:  whether use the diversity loss
- ```--mask```: whether use the task-specific mask strategy

## 4. Inference
You need to modify the path to ```model_dir```, which is obtained in the training stage.
```bash
cd scripts
bash infer.sh
```

## 5. Evaluate
You need to specify the folder of decoded texts. This folder should contain the decoded files from the same model but sampling with different random seeds where |S|=10 .
```bash
cd scripts 
python eval_seq2seq.py --folder ../{your-path-to-outputs} --tokenizer char --mbr
```

## Acknowledgement

DiffusPoll benifits from [Diffuseq](https://github.com/Shark-NLP/DiffuSeq). We are grateful to the authors for work open-source.
