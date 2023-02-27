## GoFaceRec

This repository uses [tfgo](https://github.com/galeone/tfgo) to perform face recognition on an image from file. After 
a lot of efforts, I came to the conclusion that if one wants to load a deep learning model in PyTorch or Jax in Go,
they are better to first convert their models to TensorFlow and then work with tfgo.


### Converting models
There are many ways to convert a non-TF model to a TF one. For that purpose, I used ONNX as an intermediary to convert 
the [QMagFace's model](https://github.com/pterhoer/QMagFace) from PyTorch to TF. 

Use the [model_converter.py](model_converter.py) script to convert the PyTorch model to ONNX first, and then the ONNX
model to TF. 

Some of the code in the [model_converter.py](model_converter.py) is taken from the official [QMagFace's implementation](https://github.com/pterhoer/QMagFace).


### Extracting layers

In order to run the model using tfgo, you should know the input and output layers' names. In order to extract such 
information, the `saved_model_cli` command could be useful. A model exported with `tf.saved_model.save()` automatically
comes with the "serve" tag because the SavedModel file format is designed for serving. This tag contains the various 
functions exported. Among these, there is always present the "serving_default" `signature_def`. This signature def
works exactly like the TF 1.x graph. Get the input tensor and the output tensor, and use them as placeholder to feed 
and output to get, respectively. 

To get info inside a SavedModel the best tool is `saved_model_cli` that comes with the TensorFlow Python package, for
example:
```
saved_model_cli show --all --dir output/keras
gives, among the others, this info:

signature_def['serving_default']:
The given SavedModel SignatureDef contains the following input(s):
  inputs['inputs_input'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 28, 28, 1)
      name: serving_default_inputs_input:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['logits'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 10)
      name: StatefulPartitionedCall:0
Method name is: tensorflow/serving/predict
```

Knowing the input and output layers' names, `serving_default_inputs_input:0` and `StatefulPartitionedCall:0`, is 
essential to run the model in tfgo.


### Run the model

