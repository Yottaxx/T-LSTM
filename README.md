
### 2020年毕业设计

# 基于语义理解的事件真实性预测

## Datasets/数据集
[FACTBANK](https://link.springer.com/article/10.1007/s10579-009-9089-9)
[UW](https://www.washington.edu/)
[ UDS_IH2](https://arxiv.org/pdf/1804.02472.pdf)
[MEANTIME](http://www.lrec-conf.org/proceedings/lrec2016/pdf/488_Paper.pdf)
## Unified Datasets Tool/统一数据处理工具
[工具](https://github.com/gabrielStanovsky/unified-factuality) [论文](https://www.aclweb.org/anthology/P17-2056.pdf)
需要：python2.7


## Baseline/基线模型
[https://arxiv.org/pdf/1804.02472.pdf](https://arxiv.org/pdf/1804.02472.pdf)<br>
[https://arxiv.org/pdf/1907.03227.pdf](https://arxiv.org/pdf/1907.03227.pdf)<br>

## T-LSTM/触发状态长短期记忆网络
通过拓展lstm视野域，加入通过GCN提取的trigger-state。
事件真实性预测回归架构

## Experiment/实验
### 预处理
torchtext glove-42b-300d
transformers bert-large-base
### 环境
linux 16.04
4*titan xp
python 3.6
pytorch 1.4 cuda10.1
## Hyperparameter optimization/超参数优化
### 工具：NNI
#### Tuner/调优器
Tree-structured Parzen Estimator
#### Assessor/评估器
Curve Fitting Assessor 

## 模型优点
- [✓] 线性迭代+树形结构
- [✓] 长依赖关系+语义融合
## 实验结果
- [✓] MAE+R 指标表现优良

---


