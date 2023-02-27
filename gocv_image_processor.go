// What it does:
//
// This example shows how to find circles in an image using Hough transform.
//
// How to run:
//
// 		go run ./cmd/find-circles/main.go ./images/circles.jpg
//

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

	//gocv.MedianBlur(img, &img, 5)

	c_img := gocv.NewMat()
	defer c_img.Close()
	gocv.CvtColor(img, &c_img, gocv.ColorBGRToRGB)

	r_img := gocv.NewMat()
	defer r_img.Close()
	gocv.Resize(c_img, &r_img, image.Point{112, 112}, 0, 0, gocv.InterpolationLinear)

	gocv.IMWrite("resized.png", r_img)

	//circles := gocv.NewMat()
	//defer circles.Close()
	//
	//gocv.HoughCirclesWithParams(
	//	img,
	//	&circles,
	//	gocv.HoughGradient,
	//	1,                     // dp
	//	float64(img.Rows()/8), // minDist
	//	75,                    // param1
	//	20,                    // param2
	//	10,                    // minRadius
	//	0,                     // maxRadius
	//)
	//
	//blue := color.RGBA{0, 0, 255, 0}
	//red := color.RGBA{255, 0, 0, 0}
	//
	//for i := 0; i < circles.Cols(); i++ {
	//	v := circles.GetVecfAt(0, i)
	// //	if circles are found
	//	if len(v) > 2 {
	//		x := int(v[0])
	//		y := int(v[1])
	//		r := int(v[2])
	//
	//		gocv.Circle(&cimg, image.Pt(x, y), r, blue, 2)
	//		gocv.Circle(&cimg, image.Pt(x, y), 2, red, 3)
	//	}
	//}
}
