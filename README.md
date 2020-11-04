<h1 align="center">EAdam Optimizier</h1>
<h3 align="center">EAdam OPtimizer: How Epsilon Impact Adam</h3>

## Introduction
* We find that simply changing the position of epsilon can obtain better performance than Adam through experiments.
Based on this finding, we propose a new variant of Adam called EAdam, which doesn't need extra hyper-parameters or computational costs.  We also discuss the relationships and differences between our method and Adam. We perform a thorough evaluation of our EAdam optimizer against popular and latest optimization methods including Adam, RAdam and Adabelief on different deep learning tasks.  We focus on these following tasks: image classification on CIFAR-10 and CIFAR-100, language modeling on Penn Treebank  and object detection on PASCAL VOC. 

## Algorithm

<center class="half">
    <img src="results/adam.jpg" width="50%"/><img src="results/eadam.jpg" width="50%"/>
</center>

* According to update formulas in Algorithms, Vt can be expressed by the gradients at all previous timesteps as follows

  * <center class="half">
        <img src="results/vt_adam.png" width="40%"/><img src="results/vt_eadam.png" width="40%"/>
    </center>

* After the bias correction step, we have

  * <center class="half">
        <img src="results/correct_vt_adam.png" width="40%"/><img src="results/correct_vt_eadam.png" width="40%"/>
    </center>

* Then， the adaptive stepsize are

  * <p align='center'>
    <img src="results/stepsize.png" width="40%"> </p>

* We firstly let  <img src="https://render.githubusercontent.com/render/math?math=\epsilon^{'}=\epsilon=10^{-8}">, then we want to analyse the differences of stepsizes when using Adam and EAdam to train deep networks. At the begin of training, the elements in <img src="https://render.githubusercontent.com/render/math?math=\G_t"> are far larger than <img src="https://render.githubusercontent.com/render/math?math=\epsilon^{'}"> and <img src="https://render.githubusercontent.com/render/math?math=\epsilon">, the stepsizes in Adam and EAdam can all approximated as <img src="https://render.githubusercontent.com/render/math?math=\alpha/\sqrt{G_t}">. In this case, the stepsize is determined by <img src="https://render.githubusercontent.com/render/math?math=\G_t">. Then, the elements in <img src="https://render.githubusercontent.com/render/math?math=\G_t"> may become small and <img src="https://render.githubusercontent.com/render/math?math=\epsilon^{'}"> or <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> can affect the elements in <img src="https://render.githubusercontent.com/render/math?math=\G_t">. In this case, the stepsize is determined by <img src="https://render.githubusercontent.com/render/math?math=\G_t"> and <img src="https://render.githubusercontent.com/render/math?math=\epsilon^{'}">(<img src="https://render.githubusercontent.com/render/math?math=\epsilon">). It easy to see that this case happens earlier in EAdam because <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> is added to <img src="https://render.githubusercontent.com/render/math?math=\G_t"> rather than<img src="https://render.githubusercontent.com/render/math?math=\sqrt{G_t}">. Finally, the elements in <img src="https://render.githubusercontent.com/render/math?math=\G_t"> may become far smaller than <img src="https://render.githubusercontent.com/render/math?math=\epsilon^{'}"> or <img src="https://render.githubusercontent.com/render/math?math=\epsilon">, and the stepsizes become

  * <p align='center'>
    <img src="results/final_stepsize.png" width="25%"> </p>
    
  * In this case, EAdam takes smaller stepsize than Adam. 

*  We can see that EAdam essentially adds a constant times of <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> to <img src="https://render.githubusercontent.com/render/math?math=\G_t"> before the square root operation. However, this operation is not equivalent to adding a fixed constant <img src="https://render.githubusercontent.com/render/math?math=\epsilon^{'}"> to <img src="https://render.githubusercontent.com/render/math?math=\sqrt{G_t}">. In other words, we can't find a fixed constant <img src="https://render.githubusercontent.com/render/math?math=\epsilon^{'}"> such that <a href="https://www.codecogs.com/eqnedit.php?latex=\sqrt{G_t}&plus;\epsilon^{'}=\sqrt{G_t&plus;\epsilon/(1-\beta_2)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sqrt{G_t}&plus;\epsilon^{'}=\sqrt{G_t&plus;\epsilon/(1-\beta_2)}" title="\sqrt{G_t}+\epsilon^{'}=\sqrt{G_t+\epsilon/(1-\beta_2)}" /></a>, where <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> is known, for the following reasons. If we let <img src="https://latex.codecogs.com/gif.latex?\sqrt{G_t}&plus;\epsilon^{'}=\sqrt{G_t&plus;\epsilon/(1-\beta_2)}" title="\sqrt{G_t}+\epsilon^{'}=\sqrt{G_t+\epsilon/(1-\beta_2)}" /> where <img src="https://render.githubusercontent.com/render/math?math=\epsilon^{'}"> is known. Then, we have

  <p align='center'>
  <img src="https://latex.codecogs.com/gif.latex?\epsilon^{'}=\sqrt{G_t&plus;\epsilon(1-\beta_2)}-\sqrt{G_t}" title="\epsilon^{'}=\sqrt{G_t+\epsilon(1-\beta_2)}-\sqrt{G_t}" /> </p>
  
* Because <img src="https://render.githubusercontent.com/render/math?math=\G_t"> is constantly updated, <img src="https://render.githubusercontent.com/render/math?math=\epsilon^{'}"> is also adjusted based on <img src="https://render.githubusercontent.com/render/math?math=\G_t"> in the iterative process. Therefore, <img src="https://render.githubusercontent.com/render/math?math=\epsilon^{'}"> is not fixed. From this interpretation, the change in EAdam can be seen as adopting an adaptive <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> rather than a constant in Adam. To sum up, we give some intuitive comparisons and explanations for EAdam in this subsection. However, analyzing the reasons why EAdam performances better in theory may be difficult and it is worthy to be further studied.

## Experiments

We did not precisely adjust the parameters and repeat the experiment, which will be supplemented in the future.

Code is base on:

* https://github.com/juntang-zhuang/Adabelief-Optimizer
* https://github.com/Luolc/AdaBound
* https://github.com/open-mmlab/mmdetection

#### CIFAR10 and CIFAR100

* Experiment is base on torch1.4.0

* Parameter Settings for all methods are shown in the following table

* | lr   | beta1 | beta2 | eps  | weight decay | batch size |
  | ---- | ----- | ----- | ---- | ------------ | ---------- |
  | 1e-3 | 0.9   | 0.999 | 1e-8 | 5e-4         | 128        |

* **Results:**

  <center class="half">    
  	<img src="results/Test Accuracy for Vgg11 on CIFAR10.png" width="30%"/><img src="results/Test Accuracy for ResNet18 on CIFAR10.png" width="30%"/><img src="results/Test Accuracy for DenseNet121 on CIFAR10.png" width="30%"/> 
  </center>

  <center class="half">    
  	<img src="results/Test Accuracy for Vgg11 on CIFAR100.png" width="30%"/><img src="results/Test Accuracy for ResNet18 on CIFAR100.png" width="30%"/><img src="results/Test Accuracy for DenseNet121 on CIFAR100.png" width="30%"/> 
  </center>

  

  <p align='center'>
  <img src="results/cifar_table.jpg" width="100%"> </p>

#### Penn Treebank

* Experiment is base on torch1.1.0

* Parameter Settings  shown in the following table

* | model        | lr                  | beta1 | beta2 | eps                                 | weight decay | batch size |
  | ------------ | ------------------- | ----- | ----- | ----------------------------------- | ------------ | ---------- |
  | 1-layer LSTM | 1e-3                | 0.9   | 0.999 | 1e-8(EAdam and AdaBelief are 1e-16) | 1.2e-6       | 20         |
  | 2-layer LSTM | 1e-2(RAdam is 1e-3) | 0.9   | 0.999 | 1e-8(EAdam and AdaBelief are 1e-16) | 1.2e-6       | 20         |
  | 2-layer LSTM | 1e-2(RAdam is 1e-3) | 0.9   | 0.999 | 1e-8(EAdam and AdaBelief are 1e-16) | 1.2e-6       | 20         |

* **Results:**

  <center class="half">    
  	<img src="results/Test_lstm_1layer.png" width="30%"/><img src="results/Test_lstm_2layer.png" width="30%"/><img src="results/Test_lstm_3layer.png" width="30%"/> 
  </center>

<p align='center'>
<img src="results/lstm_table.jpg" width="100%"> </p>

#### Pascal Voc

* Experiment is base on torch1.6.0, torchvision0.7.0 and mmcv-full1.1.6

* Parameter Settings for all methods are shown in the following table

* | lr   | beta1 | beta2 | eps  | weight decay | batch size |
  | ---- | ----- | ----- | ---- | ------------ | ---------- |
  | 1e-4 | 0.9   | 0.999 | 1e-8 | 1e-4         | 2          |

* **Results:**

  <p align='center'>
  <img src="results/voc_table.jpg" width="100%"> </p>

  <p align='center'>
  <img src="results/detection_demos.jpg" width="100%"> </p>

## Plan

* We will precisely adjust the parameters and repeat the experiment in the future. We may add extra experiments incluing image classification on ImageNet and objective detection on COCO. More experimental data will be published in this repository in the future.