# fastai with onnx to caffe2 export support

this is a clone of fastai from the 12.3.2018 modified to support ONNX serialization to be used with caffe2

## installation

`git clone https://github.com/jaidmin/fastai `

`cd fastai`

`conda env update`

this sets up the fastai environment

for the onnx environement run:

`conda create -n onnx python=3 anaconda`

`source activate onnx`

`conda install -c caffe2 caffe2`

`pip install git+git://github.com/onnx/onnx.git@master`

## documentation

step by step notebook for exporting a model to onnx: courses/dl1/onnx-export-project.ipynb

step by step notebook for importing a model from onnx: courses/dl1/onnx-import-project.ipynb


## changes made

changes were made to the following files:

fastai/layers.py

    added BatchNormContract and BatchNormExpand layers (basically just reshaping to get compatibility with caffe2)

    modified the AdaptiveConcatPool2d to use normal pooling instead of AdaptivePooling , as adaptive pooling is not supported by ONNX yet.

fastai/conv_learner.py

    added setting: fastai.conv_learner.caffe2_batch_norm_compat (default: True) that enables the use of the BatchNormContract and BatchNormExpand layers.
    (if the model was trained with the original fastai library some changes are necessary to load the weights, please see courses/dl1/onnx-export-project.ipynb)

    changed the create_fc_layer() method to respect the setting



