package main

import (
	"fmt"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"github.com/galeone/tensorflow/tensorflow/go/op"
	tg "github.com/galeone/tfgo"
	"github.com/galeone/tfgo/image"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("How to run:\n\tfind-circles [imgfile]")
		return
	}
	filename := os.Args[1]

	root := tg.NewRoot()
	img := image.Read(root, filename, 3)
	img = img.ResizeArea(image.Size{Height: 112, Width: 112})
	r_img := op.Reshape(root, img.Value(), tg.Const(root, [3]int32{3, 112, 112}))
	r_img = op.ExpandDims(root, r_img, tg.Const(root, []int32{0}))
	results := tg.Exec(root, []tf.Output{r_img}, nil, &tf.SessionOptions{})
	tensor_img, _ := tf.NewTensor(results[0].Value())

	qmf_model := tg.LoadModel("./magface_epoch_00025_pb", []string{"serve"}, nil)

	qmf_results := qmf_model.Exec([]tf.Output{
		qmf_model.Op("PartitionedCall", 0),
	}, map[tf.Output]*tf.Tensor{
		qmf_model.Op("serving_default_input.1", 0): tensor_img,
	})

	qmf_predictions := qmf_results[0]
	fmt.Println("####################### QMF output ######################")
	fmt.Println(qmf_predictions.Value())
	fmt.Println("#########################################################")

}
