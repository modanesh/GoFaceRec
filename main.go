package main

import (
	"fmt"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	tg "github.com/galeone/tfgo"
	"gocv.io/x/gocv"
	"image"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("How to run:\n\tfind-circles [imgfile]")
		return
	}

	filename := os.Args[1]

	// Load the image from file and run the preprocessing steps.
	// To do so, the image is first loaded from file, then the colors are
	// converted from BGR to RGB without alpha channel. And finally it is resized to 112, 112.
	img := gocv.IMRead(filename, gocv.IMReadColor)
	defer img.Close()

	cImg := gocv.NewMat()
	defer cImg.Close()
	gocv.CvtColor(img, &cImg, gocv.ColorBGRToRGB)

	rImg := gocv.NewMat()
	defer rImg.Close()
	gocv.Resize(cImg, &rImg, image.Point{112, 112}, 0, 0, gocv.InterpolationLinear)

	// A model exported with tf.saved_model.save()
	// automatically comes with the "serve" tag because the SavedModel
	// file format is designed for serving.
	// This tag contains the various functions exported. Among these, there is
	// always present the "serving_default" signature_def. This signature def
	// works exactly like the TF 1.x graph. Get the input tensor and the output tensor,
	// and use them as placeholder to feed and output to get, respectively.

	// To get info inside a SavedModel the best tool is saved_model_cli
	// that comes with the TensorFlow Python package.

	// e.g. saved_model_cli show --all --dir output/keras
	// gives, among the others, this info:

	// signature_def['serving_default']:
	// The given SavedModel SignatureDef contains the following input(s):
	//   inputs['inputs_input'] tensor_info:
	//       dtype: DT_FLOAT
	//       shape: (-1, 28, 28, 1)
	//       name: serving_default_inputs_input:0
	// The given SavedModel SignatureDef contains the following output(s):
	//   outputs['logits'] tensor_info:
	//       dtype: DT_FLOAT
	//       shape: (-1, 10)
	//       name: StatefulPartitionedCall:0
	// Method name is: tensorflow/serving/predict

	// QMagFace model
	qmfModel := tg.LoadModel("./magface_epoch_00025_pb", []string{"serve"}, nil)

	var qmfFakeData [1][3][112][112]float32
	for i := 0; i < 3; i++ {
		for j := 0; j < 112; j++ {
			for k := 0; k < 112; k++ {
				qmfFakeData[0][i][j][k] = 1.0
			}
		}
	}
	qmfFakeInput, _ := tf.NewTensor(qmfFakeData)
	//qmfFakeInput, _ := tf.NewTensor(rImg)
	fmt.Println("####################################################")
	fmt.Println(qmfFakeInput)
	root := tg.NewRoot()
	x := tg.NewTensor(root, tg.Const(root, [2][1]int64{{10}, {100}}))
	fmt.Println(x)
	fmt.Println("####################################################")

	qmfResults := qmfModel.Exec([]tf.Output{
		qmfModel.Op("PartitionedCall", 0),
	}, map[tf.Output]*tf.Tensor{
		qmfModel.Op("serving_default_input.1", 0): qmfFakeInput,
	})

	qmfPredictions := qmfResults[0]
	fmt.Println("####################### QMF output ######################")
	fmt.Println(qmfPredictions.Value())
	fmt.Println("#########################################################")

	// MTCNN model
	pnetModel := tg.LoadModel("./mtcnn_pb/pnet_pb", []string{"serve"}, nil)

	//pnetFakeInput, _ := tf.NewTensor([1][12][12][3]float32{})
	var pnetFakeData [1][12][12][3]float32
	for i := 0; i < 12; i++ {
		for j := 0; j < 12; j++ {
			for k := 0; k < 3; k++ {
				pnetFakeData[0][i][j][k] = 1.0
			}
		}
	}
	pnetFakeInput, _ := tf.NewTensor(pnetFakeData)

	pnetResults := pnetModel.Exec([]tf.Output{
		pnetModel.Op("PartitionedCall", 0),
	}, map[tf.Output]*tf.Tensor{
		pnetModel.Op("serving_default_input_1", 0): pnetFakeInput,
	})

	pnetPredictions := pnetResults[0]
	fmt.Println("####################### PNet output ######################")
	fmt.Println(pnetPredictions.Value())
	fmt.Println("#########################################################")

	rnetModel := tg.LoadModel("./mtcnn_pb/rnet_pb", []string{"serve"}, nil)

	//rnetFakeInput, _ := tf.NewTensor([1][24][24][3]float32{})
	var rnetFakeData [1][24][24][3]float32
	for i := 0; i < 24; i++ {
		for j := 0; j < 24; j++ {
			for k := 0; k < 3; k++ {
				rnetFakeData[0][i][j][k] = 1.0
			}
		}
	}
	rnetFakeInput, _ := tf.NewTensor(rnetFakeData)

	rnetResults := rnetModel.Exec([]tf.Output{
		rnetModel.Op("PartitionedCall", 0),
	}, map[tf.Output]*tf.Tensor{
		rnetModel.Op("serving_default_input_2", 0): rnetFakeInput,
	})

	rnetPredictions := rnetResults[0]
	fmt.Println("####################### RNet output ######################")
	fmt.Println(rnetPredictions.Value())
	fmt.Println("#########################################################")

	onetModel := tg.LoadModel("./mtcnn_pb/onet_pb", []string{"serve"}, nil)

	//onetFakeInput, _ := tf.NewTensor([1][48][48][3]float32{})
	var onetFakeData [1][48][48][3]float32
	for i := 0; i < 24; i++ {
		for j := 0; j < 24; j++ {
			for k := 0; k < 3; k++ {
				onetFakeData[0][i][j][k] = 1.0
			}
		}
	}
	onetFakeInput, _ := tf.NewTensor(onetFakeData)

	onetResults := onetModel.Exec([]tf.Output{
		onetModel.Op("PartitionedCall", 0),
	}, map[tf.Output]*tf.Tensor{
		onetModel.Op("serving_default_input_3", 0): onetFakeInput,
	})

	onetPredictions := onetResults[0]
	fmt.Println("####################### ONet output ######################")
	fmt.Println(onetPredictions.Value())
	fmt.Println("#########################################################")

	onetResults1 := onetModel.Exec([]tf.Output{
		onetModel.Op("PartitionedCall", 1),
	}, map[tf.Output]*tf.Tensor{
		onetModel.Op("serving_default_input_3", 0): onetFakeInput,
	})

	onetPredictions1 := onetResults1[0]
	fmt.Println("####################### ONet output ######################")
	fmt.Println(onetPredictions1.Value())
	fmt.Println("#########################################################")

	onetResults2 := onetModel.Exec([]tf.Output{
		onetModel.Op("PartitionedCall", 2),
	}, map[tf.Output]*tf.Tensor{
		onetModel.Op("serving_default_input_3", 0): onetFakeInput,
	})

	onetPredictions2 := onetResults2[0]
	fmt.Println("####################### ONet output ######################")
	fmt.Println(onetPredictions2.Value())
	fmt.Println("#########################################################")

}
