## Character-level Intra Attention Network
Implementation of the Character-level Intra Attention Network (CIAN) as a master thesis project in [Universitat Politècnica de Catalunya · BarcelonaTech](http://www.upc.edu/). 
Task description of the task can be seen at [RepEval 2017](https://repeval2017.github.io/shared/). 
Architecture of the model: 
![Visual image description of the model](https://github.com/yanghanxy/CIAN/blob/master/figure/architecture.png)

### Requirements
Code is written in python 2.7 and requires [Keras 2](https://github.com/fchollet/keras).

### Data
Dataset could be downloaded at [MultiNLI](http://www.nyu.edu/projects/bowman/multinli/) and [SNLI](http://nlp.stanford.edu/projects/snli/). 
Dataset should be be put in folder ./data and renamed as train.jsoonl, train_snli.jsonl, dev.jsoonl, test_matched.jsonl and test_mismatched.jsonl. 

### Result
![Visual image training result of the model](https://github.com/yanghanxy/CIAN/blob/master/figure/training.png)

### Visualization of Attention
PairID 192997e, label Entailment
![Visual image attention](https://github.com/yanghanxy/CIAN/blob/master/figure/3P.png)
![Visual image attention](https://github.com/yanghanxy/CIAN/blob/master/figure/3H.png)
PairID 254941e, label Entailment
![Visual image attention](https://github.com/yanghanxy/CIAN/blob/master/figure/7P.png)
![Visual image attention](https://github.com/yanghanxy/CIAN/blob/master/figure/7H.png)

### Reference
[Character-Aware Neural Language Models](https://github.com/jarfo/kchar)
[Intra Attention Mechanism](https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2)
