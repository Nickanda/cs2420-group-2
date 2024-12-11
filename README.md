# MoIST: Mixture of Intellectuals from Student Teachers

Computer Science 2420: Computing at Scale Group 2

Group Members: [Amy Dong](mailto:amydong@college.harvard.edu), [Henry Huang](mailto:hhuang@college.harvard.edu), [Nicholas Yang](mailto:nyang@college.harvard.edu), [Samuel Goldston](mailto:sgoldston@college.harvard.edu)

## 1. Introduction & Problem Statement

In recent years, significant efforts have focused on developing deep learning models that are both computationally and memory efficient, enabling their deployment on devices with limited resources. With a computational bottleneck being reached, researchers are looking to optimize aspects of machine learning algorithms so that fewer computational resources are required while running these models.

Our approach draws upon two core ideas that have reduced computation in other machine learning models: Knowledge Distillation (KD) and Mixture of Experts (MoE). In short, we take a large, pretrained teacher model and distill its knowledge to a set of smaller student models. We then employ a routing mechanism to assign and specialize each student model to a disjoint subset of the dataset. This router then selects the appropriate student model to perform inference. Instead of having a large model perform inference on a data point, that data point is routed to a smaller, specialized student tailored to that specific field, where inference is then conducted.

## 2. Related Works

1. **Knowledge Distillation**: Knowledge distillation rests on the core foundation that a pre-trained large model can guide the training of a small model such that the small model can achieve a similar accuracy while reducing the overall computational cost required per inference [^1].

2. **Mixture of Experts**: Mixture of experts employs a number of expert models, each of which are specialized in some way over their subset of data, and uses a routing mechanism that will choose one of the experts to perform inference on [^2].

Our project proposes to combine these two ideas so that one large teacher model can train multiple specialized students, forming a mixture of experts. We can then utilize this routing mechanism to use only one of the specialized students for the inference, reducing overall computation from both the Knowledge Distillation and Mixture of Experts methodologies.

## 3. Installation

We use Python >= 3.9 to run our script. Our program uses three Python libraries: PyTorch 2.5, Transformers 4.46.3, fvcore. To install the packages, you can run:

```
pip install -r requirements.txt
```

## 4. Training & Inference

We have implemented the MoIST model in `main.py` with all of the functions necessary to run and train the model. By default, MoIST will use the teacher model weights given by the `resnet18_cifar10_tailored_epoch20.pth` file and student weights given by the `student_1.pth` file. Adjustable parameters can be found within the `Config` class at the top of the file.

For convenience, we have also attached the Jupyter notebook in `main.ipynb` that can be easily imported into Google Colab.

## 5. References

[^1]: Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. “Distilling the Knowledge in a Neural Network.” arXiv, March 9, 2015. http://arxiv.org/abs/1503.02531.
[^2]: Shazeer, Noam, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.” arXiv, January 23, 2017. http://arxiv.org/abs/1701.06538.
