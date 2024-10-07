# Pep-CapsuleGAN
论文的实现
## 要求
本项目使用依赖如下：
```
Python 3.9.18
fair-esm 2.0.0
pandas 1.3.5
numpy 1.21.6
scikit-learn 1.0.2
torch 1.13.0+cu116
tensorflow 2.6.0
keras 2.6.0
```
如缺少其他库，可自行使用命令安装`pip install package_name==2.0.0`
## 数据集的t-SNE图
![t-SNE图](https://github.com/Joker-A7/Pep-CapsuleGAN/blob/main/image/t-SNE.png)
## 模型效果
![ROC曲线](https://github.com/Joker-A7/Pep-CapsuleGAN/blob/main/image/Pep_ROC_Ind.png)
![PR曲线](https://github.com/Joker-A7/Pep-CapsuleGAN/blob/main/image/Pep_PR_Ind.png)
## 使用流程
注意：数据集使用 0 和 1 分别表示高活性和低/无活性。  
### 提取特征
在 __iFeature_exteactor__ 文件中，提供了传统特征的提取方式，可以根据自己的需求进行修改。 __ESM and PortT5__ 文件中提供ESM-2和PortT5特征的提取方法，根据自己需要来进行提取。
### 特征处理
特征处理方法在 __extra_features__ 文件夹中，里面包含了特征融合和特征选择，可根据自己的需求来对提取的特征进行处理。
### 模型预测
在 __fusion_features__ 文件夹中，包含着多种机器学习模型和深度学习模型。如：LR、SVM、MLP、CNN、DNN等。通过上传自己的数据集便可以对模型进行训练和测试。
## 进一步的调整和修改
请随意进行个性化修改。只需向下滚动到模型架构部分并进行修改以满足您的期望。
