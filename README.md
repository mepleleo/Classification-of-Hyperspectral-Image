# Classification-of-Hyperspectral-Image
CNN分类Indian Pines高光谱图像

高光谱图像是在电磁光谱的大量波段中捕获的图像。本项目致力于开发深度神经网络用于高光谱图像的土地覆盖分类。土地覆盖分类的任务是为每个像素分配一个代表土地覆盖类型的分类标签。

现有的大部分研究和研究都是遵循传统的模式识别范式，即基于复杂手工特征的构建。然而，对于手头的问题，很少有人知道哪些功能是重要的。在对比这些方法的基础上，提出了一种基于深度学习的自动分层构造高层特征的分类方法。在本项目中，开发了卷积神经网络，对像素的光谱和空间信息进行编码，并使用多层感知器进行分类任务。
说明：

1)打开global_variables.txt文件，写入所需的窗口大小、PCA成分数量和train/test数据拆分。

2)运行CreatetheDatasets，创建Xtrain, Xtest, ytrain, ytest矩阵。矩阵以数字格式保存。

3)运行TrainTheModel，进行模型训练。正在保存模型，包括权重。

4)运行“验证+分类映射”，验证模型并创建分类图。

# Figures

| Patch Size | Overall Accuracy |
|   :---:    | :---:            |
|   5x5      | 83%              |
|7x7         | 88%              |
| 9x9        | 94%              |
|11x11       | 95%              |
English version：
Most of the existing studies and research efforts are following the conventional pattern recognition paradigm, which is based on the construction of complex handcrafted features. However, it is rarely known which features are important for the problem at hand. In contrast to these approaches, a deep learning based classification method that hierarchically constructs high-level features in an automated way, is proposed. In this project, exploitation of a Convolutional Neural Network, is taking part, to encode pixels’ spectral and spatial information and a Multi-Layer Perceptron to conduct the classification task.

This project is based on the paper "DEEP SUPERVISED LEARNING FOR HYPERSPECTRAL DATA CLASSIFICATION
THROUGH CONVOLUTIONAL NEURAL NETWORKS" by Makantasis et al. 

`Just to clarify, my code has nothing to do with the previously mentioned paper. I refer to it, because I followed the same resoning to build my code.`

Description of the repository
1) Open the global_variables.txt file and write the wanted windowsize, the number of the PCA components and the test train split.
2) Run the notebook "CreatetheDatasets", in order to create the Xtrain, Xtest, ytrain, ytest matrices. Matrices are saved in a numpy format.
3) Run the notebook "TrainTheModel", in order to train the model. The model is being saved including the weights.
4) Run the "Validation+ClassificationMaps", for validating the model and creating the clasification map.

# Figures

| Patch Size | Overall Accuracy |
|   :---:    | :---:            |
|   5x5      | 83%              |
|7x7         | 88%              |
| 9x9        | 94%              |
|11x11       | 95%              |

![CNN_Architecture](./images/CNN_Architecture.jpeg)


