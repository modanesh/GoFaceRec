package main

import (
	"fmt"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"github.com/galeone/tensorflow/tensorflow/go/op"
	tg "github.com/galeone/tfgo"
	"github.com/galeone/tfgo/image"
	"os"
	"strconv"
)

func main() {
	if len(os.Args) < 6 {
		fmt.Println("How to run:\n\tfind-circles [imgfile]")
		return
	}
	filename := os.Args[1]
	imgSize := os.Args[2]
	modelPath := os.Args[3]
	inputLayer := os.Args[4]
	outputLayer := os.Args[5]
	intImgSize, _ := strconv.ParseFloat(imgSize, 32)

	root := tg.NewRoot()
	img := image.Read(root, filename, 3)
	img = img.ResizeArea(image.Size{Height: float32(intImgSize), Width: float32(intImgSize)})
	rImg := op.Reshape(root, img.Value(), tg.Const(root, [3]int32{3, int32(intImgSize), int32(intImgSize)}))
	rImg = op.ExpandDims(root, rImg, tg.Const(root, []int32{0}))
	results := tg.Exec(root, []tf.Output{rImg}, nil, &tf.SessionOptions{})
	tensor_img, _ := tf.NewTensor(results[0].Value())

	qmf_model := tg.LoadModel(modelPath, []string{"serve"}, nil)

	qmf_results := qmf_model.Exec([]tf.Output{
		qmf_model.Op(inputLayer, 0),
	}, map[tf.Output]*tf.Tensor{
		qmf_model.Op(outputLayer, 0): tensor_img,
	})

	qmf_predictions := qmf_results[0]
	fmt.Println("####################### QMF output ######################")
	fmt.Println(qmf_predictions.Value())
	fmt.Println("#########################################################")

}
