---
permalink: /
title: "COMS 6998 Practical Deep Learning Systems Performance Course"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Module 1: Introduction to Machine Learning (ML) and Deep Learning (DL) 
------
ML revolution and cloud;  Overview of ML algorithms, Supervised and Unsupervised Learning; ML performance concepts/techniques: bias, variance, generalization, regularization;  Performance metrics: algorithmic and system level; DL training: backpropagation, gradient descent, activation functions, data preprocessing, batch normalization, exploding and vanishing gradients, weight initialization, learning rate policies; Regularization techniques in DL Training: dropout, early stopping, data augmentation


Module 2: DL Training Architectures, Frameworks,  Hyperparameters  
------
Stochastic and mini-batch gradient descent; Gradient descent strategies: momentum-based, AdaGrad, AdaDelta, RMSProp, Adam; DL training architectures: model and data parallelism, single node training, distributed training, parameter server, all reduce;  DL training hyperparameters: batch size, learning rate, momentum, weight decay, convergence and runtime issues; DL  training frameworks: Caffe, Tensorflow, Pytorch, Keras;  Hardware acceleration: GPUs, Tensor cores, NCCL, Intra and Inter node performance; Specialized DL architectures: CNNs, RNNs, LSTMs, GANs


Module 3: Cloud Technologies and ML Platforms 
------
ML system stack on cloud; Micro-services architecture: docker, kubernetes, kubeflow; Cloud storage: file, block, object storage, performance and flexibility; Network support on cloud platforms; Cloud based ML platforms from AWS, Microsoft, Google, and IBM; System stack, capabilities and tools support in different platforms;  Monitoring, performance, availability, and  observability 


Module 4: DL Performance Evaluation Tools and Techniques 
------
Monitoring tools: GPU resources (nvprof, nvidiasmi), host system (top, iostat),  network monitoring; Time series analysis of resource usage data; Predictive performance modeling techniques: black-box vs white-box modeling, regression modeling, analytical modeling; Predictive performance models for DL: accuracy and runtime


Module 5: ML Benchmarks 
------
DAWNBench, MLperfsuite, TensorflowHPM, Kaggle,OpenML; Datasets: MNIST, CIFAR10/100, ImageNet; Performance metrics for DL jobs; Runtime, cost, response time, accuracy, time to accuracy (TTA); Study of published numbers by different cloud service providers/vendors at benchmark forums; Compare performance scaling across GPUs for different models in MLperf ; Open Neural Network Exchange (ONNX)


Module 6: DL Systems Performance Evaluation 
------
Training-logs: framework specific support, instrumentation, analysis ; Checkpointing: framework specific support, restarting from checkpoint; Job scheduling policies like FIFO, gang, earliest deadline first; Job Schedulers : Kubernetes, Gandiva,  Optimus; Job Elasticity: scaling GPUs during runtime, platform support; Scalability: learners, batch size, single node, distributed; Overview of conferences at intersection of ML and systems


Module 7:  Advanced Topics 
------
Transfer Learning: finetuning and pseudo-labeling techniques; Deep reinforcement Learning; neural network synthesis and architecture search; Hyperparameter optimization; Automated Machine Learning; Robustness and adversarial training; Bias in models and de-biasing techniques; Devops principles in machine learning; Model lifecycle management; Drift detection and re-training; Federated learning and ML on edge devices 

