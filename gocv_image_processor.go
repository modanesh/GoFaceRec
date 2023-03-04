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

	c_img := gocv.NewMat()
	defer c_img.Close()
	gocv.CvtColor(img, &c_img, gocv.ColorBGRToRGB)

	r_img := gocv.NewMat()
	defer r_img.Close()
	gocv.Resize(c_img, &r_img, image.Point{112, 112}, 0, 0, gocv.InterpolationLinear)

	gocv.IMWrite("resized.png", r_img)
}
