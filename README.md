# Exploration and Exploitation: Two Ways to Improve Chinese Spelling Correction Models

This is the official code for paper titled "Exploration and Exploitation: Two Ways to Improve Chinese Spelling Correction Models".

We propose a method, which continually identifies the weak spots of a model to generate  more  valuable  training  instances,  and  apply a task-specific pre-training strategy to enhance the model. Experimental results show that such an adversarial training method combined with the pre-training strategy can improve both the generalization and robustness of multiple CSC models across three different datasets, achieving state-of-the-art performance for CSC task.

## Requirements

For BERT and Soft-Masked BERT:



For SpellGCN, we borrow some codes from [SpellGCN](https://github.com/ACL2020SpellGCN/SpellGCN), thus our requirements are the same with their.

## How to run?

#### 1. Prepare the data set:



#### 2. Download models you need






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