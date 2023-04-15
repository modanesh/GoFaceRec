package main

import (
	"fmt"
	"image"
	"os"

	"gocv.io/x/gocv"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("How to run:\n\tfind-circles [imgfile]")
		return
	}

	filename := os.Args[1]

	img := gocv.IMRead(filename, gocv.IMReadColor)
	defer img.Close()

	cImg := gocv.NewMat()
	defer cImg.Close()
	gocv.CvtColor(img, &cImg, gocv.ColorBGRToRGB)

	rImg := gocv.NewMat()
	defer rImg.Close()
	gocv.Resize(cImg, &rImg, image.Point{112, 112}, 0, 0, gocv.InterpolationLinear)

	gocv.IMWrite("resized.png", rImg)
}
