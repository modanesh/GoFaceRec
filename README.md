## GoFaceRec

This repository uses [tfgo](https://github.com/galeone/tfgo) to perform face recognition on an image from file. After 
a lot of efforts, I came to the conclusion that if one wants to load a deep learning model in PyTorch or Jax in Go, they better think twice before committing a good amount of effort into it. Instead, they are better to first convert their models to TensorFlow and then work with tfgo.

In this repo, the input image is first processed, and then its embeddings are compared against the ones already computed from our dataset. In order to compute and save embeddings from an arbitrary dataset, one can use the [QMagFace's repo](https://github.com/pterhoer/QMagFace). Once the embeddings are ready, this repo uses Go in order to do face recognition. If the distance between embeddings falls bellow a specific threshold, then the face is considered as unknown. Otherwise, the proper label will be printed.   

### Requirements

This project is tested using `Go 1.17` on Ubuntu 20.04. Except for `tfgo`, latest version of other packages have been used and installed.

For `gocv`, the version of `OpenCV` installed is `4.7`. And for `tfgo`, I installed [this version](https://github.com/galeone/tfgo) instead of the official one.

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


### Running the model

This project uses MTCNN for face detection and QMagFace for face recognition. For MTCNN, three stages (PNet, RNet, ONet) have been used in a close fashion similar to [FaceNet](https://github.com/davidsandberg/facenet). Each stage is done in its corresponding function:
- First stage (PNet): [`totalBoxes := firstStage(scales, img, pnetModel)`](https://github.com/modanesh/GoFaceRec/blob/main/main.go?plain=1#L1639)
- Second stage (RNet): [`squaredBoxes := secondStage(totalBoxes, width, height, img, rnetModel)`](https://github.com/modanesh/GoFaceRec/blob/main/main.go?plain=1#L1648)
- Third stage (ONet): [`thirdPickedBoxes, pickedPoints := thirdStage(squaredBoxes, width, height, img, onetModel)`](https://github.com/modanesh/GoFaceRec/blob/main/main.go?plain=1#L1657)

After the face detection stage, there is a face alignment. The function to perform face alignment is [`pImgs := alignFace(thirdPickedBoxes, pickedPoints, img)`](https://github.com/modanesh/GoFaceRec/blob/main/main.go?plain=1#L1666) which imitates the steps from [here](https://github.com/pterhoer/QMagFace/blob/main/preprocessing/insightface/src/face_preprocess.py#L195).

Finally, once the face is detected and aligned, the recognition phase can start. It happens at this line: [`recognizeFace(pImgs, qmfModel, regEmbeddings, bSize, regFiles)`](https://github.com/modanesh/GoFaceRec/blob/main/main.go?plain=1#L1675).

Use the bellow command to run the code:
```shell
go run main.go IMAGE.jpg
```


### Challenges

The main challenge thus far was the conversion between [`gocv.Mat`](https://github.com/hybridgroup/gocv), [`tfgo.Tensor`](https://github.com/galeone/tfgo), [`gonum`](https://github.com/gonum/gonum/), and Go's native slice. The conversion is required as some matrix transformations are only available in `gocv` and some in `tfgo`. Also, the input to the `tfgo` model should be of type `tfgo.Tensor`, so inevitably one needs to convert the image read by `gocv` to `tfgo`. Also, some matrix operations are not available in any of these packages, so I had to implement them myself from scratch. To do so, I had to use Go's native slice. So inevitable conversions between these types are frequent throughout the code.

For example, the function [`adjustInput()`](https://github.com/modanesh/GoFaceRec/blob/main/main.go?plain=1#L502) besides doing some scaling, it also converts a `gocv.Mat` to Go's `[][][][]float32`. In addition, at this line: [`inputBufTensor, _ := tf.NewTensor(inputBuf)`](https://github.com/modanesh/GoFaceRec/blob/main/main.go?plain=1#L402) a `[][][][]float32` slice is converted to a `tfgo.Tensor`.

In contrast, these type conversions are done pretty easy and fast in Python.

### ToDo
- [X] Check why recognition model takes so long for a forward pass. In Python, it takes about 0.5 milliseconds while in Go it takes about 5500 milliseconds. For the first run, in Go, the session instantiation takes a long time. For next runs, Go runs pretty fast. [Take a look at this issue](https://github.com/galeone/tfgo/issues/4).