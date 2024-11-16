# MoIST: Mixture of Intellectuals from Student Teachers

Computer Science 2420: Computing at Scale Group 2

Group Members: [Amy Dong](mailto:amydong@college.harvard.edu), [Henry Huang](mailto:hhuang@college.harvard.edu), [Nicholas Yang](mailto:nyang@college.harvard.edu), [Samuel Goldston](mailto:sgoldston@college.harvard.edu)

## 1. Introduction & Problem Statement

Within recent years, many efforts have been made to produce deep learning models that are both computationally efficient and memory efficient, such that these machine learning models can be employed in devices that may not have as much computational resources. With a computational bottleneck being reached in terms of computational chip research, many researchers are looking to optimize many aspects of machine learning algorithms so that less computational resources is demanded from the machine that is running these algorithms.

Our approach draws upon two core ideas that have reduced computational requirements in other models: (1) knowledge distillation, and (2) mixture of experts. In short, we take a large, pre-trained teacher model to distill its knowledge to a set of smaller student models, each of which is specialized a disjoint subset of the dataset so that each student becomes specialized in their "field." We then employ a routing mechanism, which will learn which one of the student models will be most specialized for any given inference data, and select that student model to perform inference on the data. In this way, rather than having an entire large model perform inference on a singular data point, we are able to employ smaller student models, who are specialized in the "field" that the data point is in, and run inference on that smaller model.

## 2. Related Works

1. **Knowledge Distillation**: Knowledge distillation rests on the core foundation that a pre-trained large model can guide the training of a small model such that the small model can achieve a similar accuracy while reducing the overall computational cost required per inference [^1].

2. **Mixture of Experts**: Mixture of experts employs a number of expert models, each of which are specialized in some way over their subset of data, and uses a routing mechanism that will choose one of the experts to perform inference on [^2].

Our project proposes to combine these two ideas so that one large teacher model can train multiple specialized students, forming a mixture of experts. We can then utilize this routing mechanism to use only one of the specialized students for the inference, reducing overall computation from both the Knowledge Distillation and Mixture of Experts methodologies.

## 3. Installation

## 4. Training & Inference

## x. References

[^1]: Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. “Distilling the Knowledge in a Neural Network.” arXiv, March 9, 2015. http://arxiv.org/abs/1503.02531.
[^2]: Shazeer, Noam, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.” arXiv, January 23, 2017. http://arxiv.org/abs/1701.06538.
