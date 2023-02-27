package main

import (
	"fmt"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"github.com/galeone/tensorflow/tensorflow/go/op"
	tg "github.com/galeone/tfgo"
	"github.com/galeone/tfgo/image"
	"os"
)

//func makeTensorFromImage(filename string) (*tf.Tensor, error) {
//	bytes, err := ioutil.ReadFile(filename)
//	if err != nil {
//		return nil, err
//	}
//	// DecodeJpeg uses a scalar String-valued tensor as input.
//	tensor, err := tf.NewTensor(string(bytes))
//	if err != nil {
//		return nil, err
//	}
//	// Construct a graph to normalize the image
//	graph, input, output, err := constructGraphToNormalizeImage()
//	if err != nil {
//		return nil, err
//	}
//	// Execute that graph to normalize this one image
//	session, err := tf.NewSession(graph, nil)
//	if err != nil {
//		return nil, err
//	}
//	defer session.Close()
//	normalized, err := session.Run(
//		map[tf.Output]*tf.Tensor{input: tensor},
//		[]tf.Output{output},
//		nil)
//	if err != nil {
//		return nil, err
//	}
//	return normalized[0], nil
//}

//func constructGraphToNormalizeImage() (graph *tf.Graph, input, output tf.Output, err error) {
//	const (
//		H, W  = 112, 112
//		Mean  = float32(117)
//		Scale = float32(1)
//	)
//	s := op.NewScope()
//	input = op.Placeholder(s, tf.String)
//	output = op.Div(s,
//		op.Sub(s,
//			op.ResizeBilinear(s,
//				op.ExpandDims(s,
//					op.Cast(s,
//						op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)), tf.Float),
//					op.Const(s.SubScope("make_batch"), int32(0))),
//				op.Const(s.SubScope("size"), []int32{H, W})),
//			op.Const(s.SubScope("mean"), Mean)),
//		op.Const(s.SubScope("scale"), Scale))
//	graph, err = s.Finalize()
//	return graph, input, output, err
//}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("How to run:\n\tfind-circles [imgfile]")
		return
	}
	filename := os.Args[1]

	root := tg.NewRoot()
	img := image.Read(root, filename, 3)
	fmt.Println("***************** 1")
	fmt.Println(img)
	img = img.ResizeArea(image.Size{Height: 112, Width: 112})
	fmt.Println("***************** 2")
	fmt.Println(img)
	r_img := op.Reshape(root, img.Value(), tg.Const(root, [3]int32{3, 112, 112}))
	fmt.Println("***************** 3")
	fmt.Println(r_img)
	r_img = op.ExpandDims(root, r_img, tg.Const(root, []int32{0}))
	fmt.Println("***************** 4")
	fmt.Println(r_img)
	results := tg.Exec(root, []tf.Output{r_img}, nil, &tf.SessionOptions{})
	tensor_img, err := tf.NewTensor(results[0].Value())
	fmt.Println("***************** 5")
	fmt.Println(tensor_img)
        fmt.Println(err)
	
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
