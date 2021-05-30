# Exploration and Exploitation: Two Ways to Improve Chinese Spelling Correction Models

This is the official code for paper titled "Exploration and Exploitation: Two Ways to Improve Chinese Spelling Correction Models".

We propose a method, which continually identifies the weak spots of a model to generate  more  valuable  training  instances,  and  apply a task-specific pre-training strategy to enhance the model. Experimental results show that such an adversarial training method combined with the pre-training strategy can improve both the generalization and robustness of multiple CSC models across three different datasets, achieving state-of-the-art performance for CSC task.

## Requirements

For BERT and Soft-Masked BERT:

- python==
- pytorch==
- transformers==

For SpellGCN, we borrow some codes from [SpellGCN](https://github.com/ACL2020SpellGCN/SpellGCN), thus our requirements are the same with their.

- Tensorflow==1.13.1
- python==2.7
- "BERT-Base, Chinese" from [google-research](https://github.com/google-research/bert)

## How to run?

#### 1. Prepare the datasets:

- **For pre-train:**
  1. Wiki: Download the latest [zhwiki](https://dumps.wikimedia.org/zhwiki/) and  process the dump with [gensim](https://radimrehurek.com/gensim/corpora/wikicorpus.html).
  2. Weibo: Download the [Weibo](61.93.89.94/Noah_NRM_Data/) datasets.
  3. Pre-process the corpus to remove noise, such as splitting paragraphs into sentences, filtering out the inappropriate sentences (too long or too short) and so on. 
- **For train:**
  1. Download the additional 270K data samples from [here](https://github.com/wdimmy/Automatic-Corpus-Generation).
  2. Extract the training samples from the file "train.sgml".
- **Note: The data samples mentioned above are absent here due to the lack of permission.**



#### 2. Run the models:

- **For BERT and Soft-Masked BERT:**

1. Set up an virtual environment for BERT and Soft-Masked BERT(python==,torch==,transformers==) using Anaconda

   ```
   conda create -n bert python=
   conda activate bert
   pip install torch==
   pip install transformers==
   ```

2. Go to the directory "scripts", set up your private parameters(like the path of initial model and data)

   ```
   cd scripts
   vim run.sh
   ```

3. bash run.sh

   ```
   bash run.sh
   ```

- **For SpellGCN:**

  1. Set up an virtual environment for SpellGCN (python==2.7, Tensorflow==1.13.1) using Anaconda

     ```
     conda create -n spellgcn python=2.7.1
     source activate spellgcn
     pip install tensorflow==1.13.1
     ```

  2. Go to the directory "scripts", set up your private parameters (like the path of BERT and initial model)

     ```
     cd scripts
     vim run.sh
     ```

  3. bash run.sh

     ```
     bash run.sh
     ```

#### 3. Or you can download the models you need and init your models from them.

- Baidu Wangpan:

  - 链接：https://pan.baidu.com/s/1O9mLjWSiXzxcPBy0fU-_BQ 
    提取码：y25e

     


## Contact

chongli17@fudan.edu.cn and cenyuanzhang17@fudan.edu.cn



## How to cite our paper?


```
@inproceedings{li-etal-2021-2Ways,
  author    = {Chong Li and
               Cenyuan Zhang and
               Xiaoqing Zheng and
               Xuanjing Huang},
  title="Exploration and Exploitation: Two Ways to Improve Chinese Spelling Correction Models",
  booktitle="Proceedings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing",
  publisher = "Association for Computational Linguistics",
  year="2021"
}
```