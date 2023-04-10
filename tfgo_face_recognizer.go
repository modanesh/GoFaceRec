package main

import (
	"fmt"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"github.com/galeone/tensorflow/tensorflow/go/op"
	tg "github.com/galeone/tfgo"
	tfimage "github.com/galeone/tfgo/image"
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
	img := tfimage.Read(root, filename, 3)
	img = img.ResizeArea(tfimage.Size{Height: float32(intImgSize), Width: float32(intImgSize)})
	rImg := op.Reshape(root, img.Value(), tg.Const(root, [3]int32{3, int32(intImgSize), int32(intImgSize)}))
	rImg = op.ExpandDims(root, rImg, tg.Const(root, []int32{0}))
	results := tg.Exec(root, []tf.Output{rImg}, nil, &tf.SessionOptions{})
	tensorImg, _ := tf.NewTensor(results[0].Value())

	qmfModel := tg.LoadModel(modelPath, []string{"serve"}, nil)

	qmfResults := qmfModel.Exec([]tf.Output{
		qmfModel.Op(inputLayer, 0),
	}, map[tf.Output]*tf.Tensor{
		qmfModel.Op(outputLayer, 0): tensorImg,
	})

	fmt.Println("####################### QMF input ######################")
	fmt.Println(tensorImg.Value())
	fmt.Println("####################### QMF output ######################")
	qmfPredictions := qmfResults[0]
	fmt.Println(qmfPredictions.Value())
	fmt.Println("#########################################################")

}
