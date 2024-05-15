# Introduction
These are assets that I've created for [Interview Kickstart](https://learn.interviewkickstart.com/) where I teach and create content for various Deep Learning courses.

## Create the environment
conda create -c pytorch -c nvidia -n ik pytorch torchvision torchaudio pytorch-cuda=11.8
conda activate ik
pip install -r requirements.txt

## Relevant Files

### [yolo.ipynb](yolo/yolo.ipynb 'yolo.ipynb')

This is a Jupyter notebook that shows how to fine tune the YOLO v8 model on a custom dataset. To do this I create arbitrarily sized and colored triangles and circles on a noisy background, a config file to specify triangle and circle classes and the ground truths for their bounding boxes. The intent here is to cleanly show how YOLO can be used for new object detection use cases and minimize the time it takes to learn how to do this.

### [demo.ipynb](model_demo/demo.ipynb)

This is a Jupyter notebook that shows how to create classification models using Fully Connected, Convolutional, Vision Transformer and Hybrid models. These models are trained on the MNIST, CIFAR10 and CIFAR100 datasets and graphics are generated that show the trade offs between inference time, size and accuracy. 

### [tf_cnn.py](tf_cnn/tf_cnn.py)

This script simply prints out a TensorFlow model summary which is used in the curriculum to show the layers that are produced for different lines of code.
