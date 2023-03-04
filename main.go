package main

import (
	"fmt"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	tg "github.com/galeone/tfgo"
	"gocv.io/x/gocv"
	"image"
)

func main() {
	filename := "./airplane.png"

	img := gocv.IMRead(filename, gocv.IMReadColor)
	defer img.Close()

	cImg := gocv.NewMat()
	defer cImg.Close()
	gocv.CvtColor(img, &cImg, gocv.ColorBGRToRGB)

	rImg := gocv.NewMat()
	defer rImg.Close()
	gocv.Resize(cImg, &rImg, image.Point{112, 112}, 0, 0, gocv.InterpolationLinear)

	normalizedImg := gocv.NewMat()
	defer normalizedImg.Close()
	//TODO
	gocv.Normalize(rImg, &normalizedImg, 1, 0, 1)

	toImgInt := normalizedImg.ToBytes()
	toImgFloat := make([]float32, len(toImgInt))
	for i, v := range toImgInt {
		toImgFloat[i] = float32(v)
	}
	tensorImg, _ := tf.NewTensor(toImgFloat)
	_ = tensorImg.Reshape([]int64{1, 3, 112, 112})

	qmfModel := tg.LoadModel("./magface_epoch_00025_pb", []string{"serve"}, nil)

	qmfResults := qmfModel.Exec([]tf.Output{
		qmfModel.Op("PartitionedCall", 0),
	}, map[tf.Output]*tf.Tensor{
		qmfModel.Op("serving_default_input.1", 0): tensorImg,
	})

	fmt.Println("####################### QMF input ######################")
	fmt.Println(tensorImg.Value())
	fmt.Println("####################### QMF output ######################")
	qmf_predictions := qmfResults[0]
	fmt.Println(qmf_predictions.Value())
	fmt.Println("#########################################################")

}
