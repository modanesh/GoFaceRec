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

	c_img := gocv.NewMat()
	defer c_img.Close()
	gocv.CvtColor(img, &c_img, gocv.ColorBGRToRGB)

	r_img := gocv.NewMat()
	defer r_img.Close()
	gocv.Resize(c_img, &r_img, image.Point{112, 112}, 0, 0, gocv.InterpolationLinear)

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
	qmf_model := tg.LoadModel("./magface_epoch_00025_pb", []string{"serve"}, nil)

	var qmf_fakeData [1][3][112][112]float32
	for i := 0; i < 3; i++ {
		for j := 0; j < 112; j++ {
			for k := 0; k < 112; k++ {
				qmf_fakeData[0][i][j][k] = 1.0
			}
		}
	}
	qmf_fakeInput, _ := tf.NewTensor(qmf_fakeData)
	//qmf_fakeInput, _ := tf.NewTensor(r_img)
	fmt.Println("####################################################")
	fmt.Println(qmf_fakeInput)
	root := tg.NewRoot()
	x := tg.NewTensor(root, tg.Const(root, [2][1]int64{{10}, {100}}))
	fmt.Println(x)
	fmt.Println("####################################################")

	qmf_results := qmf_model.Exec([]tf.Output{
		qmf_model.Op("PartitionedCall", 0),
	}, map[tf.Output]*tf.Tensor{
		qmf_model.Op("serving_default_input.1", 0): qmf_fakeInput,
	})

	qmf_predictions := qmf_results[0]
	fmt.Println("####################### QMF output ######################")
	fmt.Println(qmf_predictions.Value())
	fmt.Println("#########################################################")

	// MTCNN model
	pnet_model := tg.LoadModel("./mtcnn_pb/pnet_pb", []string{"serve"}, nil)

	//pnet_fakeInput, _ := tf.NewTensor([1][12][12][3]float32{})
	var pnet_fakeData [1][12][12][3]float32
	for i := 0; i < 12; i++ {
		for j := 0; j < 12; j++ {
			for k := 0; k < 3; k++ {
				pnet_fakeData[0][i][j][k] = 1.0
			}
		}
	}
	pnet_fakeInput, _ := tf.NewTensor(pnet_fakeData)

	pnet_results := pnet_model.Exec([]tf.Output{
		pnet_model.Op("PartitionedCall", 0),
	}, map[tf.Output]*tf.Tensor{
		pnet_model.Op("serving_default_input_1", 0): pnet_fakeInput,
	})

	pnet_predictions := pnet_results[0]
	fmt.Println("####################### PNet output ######################")
	fmt.Println(pnet_predictions.Value())
	fmt.Println("#########################################################")

	rnet_model := tg.LoadModel("./mtcnn_pb/rnet_pb", []string{"serve"}, nil)

	//rnet_fakeInput, _ := tf.NewTensor([1][24][24][3]float32{})
	var rnet_fakeData [1][24][24][3]float32
	for i := 0; i < 24; i++ {
		for j := 0; j < 24; j++ {
			for k := 0; k < 3; k++ {
				rnet_fakeData[0][i][j][k] = 1.0
			}
		}
	}
	rnet_fakeInput, _ := tf.NewTensor(rnet_fakeData)

	rnet_results := rnet_model.Exec([]tf.Output{
		rnet_model.Op("PartitionedCall", 0),
	}, map[tf.Output]*tf.Tensor{
		rnet_model.Op("serving_default_input_2", 0): rnet_fakeInput,
	})

	rnet_predictions := rnet_results[0]
	fmt.Println("####################### RNet output ######################")
	fmt.Println(rnet_predictions.Value())
	fmt.Println("#########################################################")

	onet_model := tg.LoadModel("./mtcnn_pb/onet_pb", []string{"serve"}, nil)

	//onet_fakeInput, _ := tf.NewTensor([1][48][48][3]float32{})
	var onet_fakeData [1][48][48][3]float32
	for i := 0; i < 24; i++ {
		for j := 0; j < 24; j++ {
			for k := 0; k < 3; k++ {
				onet_fakeData[0][i][j][k] = 1.0
			}
		}
	}
	onet_fakeInput, _ := tf.NewTensor(onet_fakeData)

	onet_results := onet_model.Exec([]tf.Output{
		onet_model.Op("PartitionedCall", 0),
	}, map[tf.Output]*tf.Tensor{
		onet_model.Op("serving_default_input_3", 0): onet_fakeInput,
	})

	onet_predictions := onet_results[0]
	fmt.Println("####################### ONet output ######################")
	fmt.Println(onet_predictions.Value())
	fmt.Println("#########################################################")

	onet_results_1 := onet_model.Exec([]tf.Output{
		onet_model.Op("PartitionedCall", 1),
	}, map[tf.Output]*tf.Tensor{
		onet_model.Op("serving_default_input_3", 0): onet_fakeInput,
	})

	onet_predictions_1 := onet_results_1[0]
	fmt.Println("####################### ONet output ######################")
	fmt.Println(onet_predictions_1.Value())
	fmt.Println("#########################################################")

	onet_results_2 := onet_model.Exec([]tf.Output{
		onet_model.Op("PartitionedCall", 2),
	}, map[tf.Output]*tf.Tensor{
		onet_model.Op("serving_default_input_3", 0): onet_fakeInput,
	})

	onet_predictions_2 := onet_results_2[0]
	fmt.Println("####################### ONet output ######################")
	fmt.Println(onet_predictions_2.Value())
	fmt.Println("#########################################################")

}
