package main

import (
	"fmt"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	tg "github.com/galeone/tfgo"
	"github.com/nfnt/resize"
	"github.com/sbinet/npyio"
	"gocv.io/x/gocv"
	"gorgonia.org/tensor"
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
)

// FloatImage represents a custom image type that satisfies the image.Image interface
type FloatImage struct {
	data [][]float32
}

func (f FloatImage) ColorModel() color.Model {
	return color.Gray16Model
}

func (f FloatImage) Bounds() image.Rectangle {
	height := len(f.data)
	width := len(f.data[0])
	return image.Rect(0, 0, width, height)
}

func (f FloatImage) At(x, y int) color.Color {
	return color.Gray16{Y: uint16(f.data[y][x])}
}

// ConvertFloatImage converts [][]float32 to FloatImage
func ConvertFloatImage(data [][]float32) *FloatImage {
	return &FloatImage{data: data}
}

func matToFloat32Slice(mat gocv.Mat) [][]float32 {
	rows, cols := mat.Rows(), mat.Cols()
	slice := make([][]float32, rows)

	for i := 0; i < rows; i++ {
		rowSlice := make([]float32, cols)
		for j := 0; j < cols; j++ {
			val := mat.GetFloatAt(i, j)
			rowSlice[j] = float32(val)
		}
		slice[i] = rowSlice
	}
	return slice
}

func sliceIndex(number int) []int {
	indexes := make([]int, number)
	for i := 0; i < number; i++ {
		indexes[i] = i
	}
	return indexes
}

func removeIndex(s []int, idx []int) []int {
	sort.Ints(idx)
	offset := 0
	for _, i := range idx {
		i -= offset
		s = append(s[:i], s[i+1:]...)
		offset++
	}
	return s
}

func nms(boxes [][]float32, overlapThreshold float64, mode string) []int {
	if len(boxes) == 0 {
		return []int{}
	}
	// if the bounding boxes integers, convert them to floats
	if reflect.TypeOf(boxes[0][0]).Kind() == reflect.Int {
		for i := range boxes {
			for j := range boxes[i] {
				boxes[i][j] = boxes[i][j]
			}
		}
	}
	// initialize the list of picked indexes
	pick := []int{}

	// grab the coordinates of the bounding boxes
	var x1, y1, x2, y2, score []float32
	for _, box := range boxes {
		x1 = append(x1, box[0])
		y1 = append(y1, box[1])
		x2 = append(x2, box[2])
		y2 = append(y2, box[3])
		score = append(score, box[4])
	}

	area := make([]float32, len(boxes))
	for i := range boxes {
		area[i] = (x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1)
	}

	idxs := make([]int, len(score))
	for i := 0; i < len(score); i++ {
		idxs[i] = i
	}

	sort.Slice(idxs, func(i, j int) bool { return score[idxs[i]] < score[idxs[j]] })

	// keep looping while some indexes still remain in the indexes list
	for len(idxs) > 0 {
		// grab the last index in the indexes list and add the index value to the list of picked indexes
		last := len(idxs) - 1
		i := idxs[last]
		pick = append(pick, i)

		xx1 := make([]float64, last)
		yy1 := make([]float64, last)
		xx2 := make([]float64, last)
		yy2 := make([]float64, last)
		for j := 0; j < last; j++ {
			xx1[j] = math.Max(float64(x1[i]), float64(x1[idxs[j]]))
			yy1[j] = math.Max(float64(y1[i]), float64(y1[idxs[j]]))
			xx2[j] = math.Min(float64(x2[i]), float64(x2[idxs[j]]))
			yy2[j] = math.Min(float64(y2[i]), float64(y2[idxs[j]]))
		}

		// compute the width and height of the bounding box
		w := make([]float64, last)
		h := make([]float64, last)
		for j := 0; j < last; j++ {
			w[j] = math.Max(0, xx2[j]-xx1[j]+1)
			h[j] = math.Max(0, yy2[j]-yy1[j]+1)
		}
		inter := make([]float64, last)
		overlap := make([]float64, last)
		for j := 0; j < last; j++ {
			inter[j] = w[j] * h[j]
			if mode == "Min" {
				overlap[j] = inter[j] / math.Min(float64(area[i]), float64(area[idxs[j]]))
			} else {
				overlap[j] = inter[j] / (float64(area[i]) + float64(area[idxs[j]]) - inter[j])
			}
		}

		// delete all indexes from the index list that have
		toDelete := []int{}
		for j := 0; j < last; j++ {
			if overlap[j] > overlapThreshold {
				toDelete = append(toDelete, j)
			}
		}
		toDelete = append(toDelete, last)

		deleteIdxs := make([]int, len(toDelete))
		for j := 0; j < len(toDelete); j++ {
			deleteIdxs[j] = idxs[toDelete[j]]
		}
		idxs = removeIndex(idxs, toDelete)
	}
	return pick
}

func reshape4DArray(originalArray [][][][]float32, d1, d2, d3, d4 int) [][][][]float32 {
	newShape := [4]int{d1, d2, d3, d4}
	newArray := make([][][][]float32, newShape[0])
	for i := range newArray {
		newArray[i] = make([][][]float32, newShape[1])
		for j := range newArray[i] {
			newArray[i][j] = make([][]float32, newShape[2])
			for k := range newArray[i][j] {
				newArray[i][j][k] = make([]float32, newShape[3])
				for l := range newArray[i][j][k] {
					newArray[i][j][k][l] = originalArray[0][l][0][j]
				}
			}
		}
	}
	return newArray
}

func generateBBox(heatmap [][]float32, reg [][][][]float32, scale float64, threshold float32) [][]float32 {
	stride := 2
	cellsize := 12

	tIndex := make([][]int, 0)

	for i, row := range heatmap {
		for j, value := range row {
			if value > threshold {
				tIndex = append(tIndex, []int{i, j})
			}
		}
	}

	if len(tIndex) == 0 {
		return [][]float32{}
	}
	dx1, dy1, dx2, dy2 := make([]float32, len(tIndex)), make([]float32, len(tIndex)), make([]float32, len(tIndex)), make([]float32, len(tIndex))
	for i, idx := range tIndex {
		dx1[i] = reg[0][0][idx[0]][idx[1]]
		dy1[i] = reg[0][1][idx[0]][idx[1]]
		dx2[i] = reg[0][2][idx[0]][idx[1]]
		dy2[i] = reg[0][3][idx[0]][idx[1]]
	}

	score := make([]float32, len(tIndex))
	for i, idx := range tIndex {
		score[i] = heatmap[idx[0]][idx[1]]
	}

	boundingBox := make([][]float32, len(tIndex))
	for i := range boundingBox {
		boundingBox[i] = make([]float32, 9)
	}

	for i, idx := range tIndex {
		boundingBox[i][0] = float32(math.Round((float64(stride*idx[1] + 1)) / scale))
		boundingBox[i][1] = float32(math.Round((float64(stride*idx[0] + 1)) / scale))
		boundingBox[i][2] = float32(math.Round((float64(stride*idx[1] + 1 + cellsize)) / scale))
		boundingBox[i][3] = float32(math.Round((float64(stride*idx[0] + 1 + cellsize)) / scale))
		boundingBox[i][4] = score[i]
		boundingBox[i][5] = dx1[i]
		boundingBox[i][6] = dy1[i]
		boundingBox[i][7] = dx2[i]
		boundingBox[i][8] = dy2[i]
	}

	return boundingBox
}

func flatten4DTo2D(data [][][][]float32) [][]float32 {
	var output [][]float32
	for i := 0; i < len(data[0][1]); i++ {
		row := make([]float32, len(data[0][1][i]))
		copy(row, data[0][1][i])
		output = append(output, row)
	}
	return output
}

func flatten4DTo3D(data [][][][]float32) [][][]float32 {
	result := make([][][]float32, len(data))

	for i := range data {
		result[i] = make([][]float32, len(data[i]))
		for j := range data[i] {
			temp := make([]float32, 0)
			for k := range data[i][j] {
				temp = append(temp, data[i][j][k]...)
			}
			result[i][j] = temp
		}
	}

	return result
}

func transpose(x [][][][]float32, order []int) [][][][]float32 {
	if len(order) != 4 {
		panic("order must have a length of 4")
	}

	dims := []int{len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])}
	newDims := []int{dims[order[0]], dims[order[1]], dims[order[2]], dims[order[3]]}

	// Create output slice with transposed dimensions
	out := make([][][][]float32, newDims[0])
	for i := range out {
		out[i] = make([][][]float32, newDims[1])
		for j := range out[i] {
			out[i][j] = make([][]float32, newDims[2])
			for k := range out[i][j] {
				out[i][j][k] = make([]float32, newDims[3])
			}
		}
	}

	// Transpose elements
	for i := 0; i < dims[0]; i++ {
		for j := 0; j < dims[1]; j++ {
			for k := 0; k < dims[2]; k++ {
				for l := 0; l < dims[3]; l++ {
					xIndex := []int{i, j, k, l}
					outIndex := []int{xIndex[order[0]], xIndex[order[1]], xIndex[order[2]], xIndex[order[3]]}
					out[outIndex[0]][outIndex[1]][outIndex[2]][outIndex[3]] = x[xIndex[0]][xIndex[1]][xIndex[2]][xIndex[3]]
				}
			}
		}
	}

	return out
}

func detectFirstStage(img gocv.Mat, net *tg.Model, scale float64, threshold float32) [][]float32 {
	height, width := img.Size()[0], img.Size()[1]
	ws := int(math.Ceil(float64(width) * scale))
	hs := int(math.Ceil(float64(height) * scale))

	imData := gocv.NewMat()
	defer imData.Close()
	gocv.Resize(img, &imData, image.Point{X: ws, Y: hs}, 0, 0, gocv.InterpolationLinear)
	//gocv.Resize(img, &imData, image.Point{X: 12, Y: 12}, 0, 0, gocv.InterpolationLinear)

	inputBuf := adjustInput(imData)
	inputBufTensor, _ := tf.NewTensor(inputBuf)
	newShape := []int64{1, int64(ws), int64(hs), 3}
	inputBufTensor.Reshape(newShape)
	netOutput := net.Exec([]tf.Output{
		net.Op("PartitionedCall", 0),
		net.Op("PartitionedCall", 1),
	}, map[tf.Output]*tf.Tensor{
		net.Op("serving_default_input_1", 0): inputBufTensor,
	})
	reg, ok := netOutput[0].Value().([][][][]float32)
	if !ok {
		fmt.Println("Failed to convert reg to [][][][]float64")
	}
	heatmap, ok := netOutput[1].Value().([][][][]float32)
	if !ok {
		fmt.Println("Failed to convert heatmap to [][]float64")
	}
	order := []int{0, 3, 2, 1}
	reg = transpose(reg, order)
	heatmap = transpose(heatmap, order)
	boxes := generateBBox(flatten4DTo2D(heatmap), reg, scale, threshold)

	if len(boxes) == 0 {
		return nil
	}

	// nms
	pick := nms(boxes, 0.5, "Union")
	var pickedBoxes [][]float32
	for _, index := range pick {
		pickedBoxes = append(pickedBoxes, boxes[index])
	}
	return pickedBoxes
}

func convertToSquare(bbox [][]float32) [][]float32 {
	squareBbox := make([][]float32, len(bbox))
	for i := 0; i < len(bbox); i++ {
		squareBbox[i] = make([]float32, len(bbox[i]))
		copy(squareBbox[i], bbox[i])
		h := bbox[i][3] - bbox[i][1] + 1
		w := bbox[i][2] - bbox[i][0] + 1
		maxSide := float32(math.Max(float64(h), float64(w)))
		squareBbox[i][0] = bbox[i][0] + w*0.5 - maxSide*0.5
		squareBbox[i][1] = bbox[i][1] + h*0.5 - maxSide*0.5
		squareBbox[i][2] = squareBbox[i][0] + maxSide - 1
		squareBbox[i][3] = squareBbox[i][1] + maxSide - 1
	}
	return squareBbox
}

func pad(bboxes [][]float32, w float32, h float32) ([]float32, []float32, []float32, []float32, []float32, []float32, []float32, []float32, []float32, []float32) {
	numBox := len(bboxes)
	tmpw := make([]float32, numBox)
	tmph := make([]float32, numBox)

	for i := range bboxes {
		tmpw[i] = bboxes[i][2] - bboxes[i][0] + 1
		tmph[i] = bboxes[i][3] - bboxes[i][1] + 1
	}

	dx := make([]float32, numBox)
	dy := make([]float32, numBox)
	edx := make([]float32, numBox)
	edy := make([]float32, numBox)

	x := make([]float32, numBox)
	y := make([]float32, numBox)
	ex := make([]float32, numBox)
	ey := make([]float32, numBox)

	for i := range bboxes {
		dx[i], dy[i] = 0, 0
		edx[i], edy[i] = tmpw[i]-1, tmph[i]-1
		x[i], y[i], ex[i], ey[i] = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]

		if ex[i] > w-1 {
			edx[i] = tmpw[i] + w - 2 - ex[i]
			ex[i] = w - 1
		}
		if ey[i] > h-1 {
			edy[i] = tmph[i] + h - 2 - ey[i]
			ey[i] = h - 1
		}
		if x[i] < 0 {
			dx[i] = 0 - x[i]
			x[i] = 0
		}
		if y[i] < 0 {
			dy[i] = 0 - y[i]
			y[i] = 0
		}
	}

	return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph
}

func adjustInput(inData gocv.Mat) [][]float32 {
	// adjust the input from (h, w, c) to (1, c, h, w) for network input
	channels := inData.Channels()
	rows, cols := inData.Rows(), inData.Cols()

	// transpose (h, w, c) to (c, h, w)
	outData := make([][]float32, channels)
	for c := 0; c < channels; c++ {
		outData[c] = make([]float32, rows*cols)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				v := inData.GetVecfAt(i, j)[c]
				outData[c][i*cols+j] = float32(v)
			}
		}
	}

	// expand dims to (1, c, h, w)
	outData = [][]float32{flatten2DTo1D(outData)}

	// normalize
	for c := 0; c < channels; c++ {
		for i := 0; i < rows*cols; i++ {
			outData[0][c*rows*cols+i] = (outData[0][c*rows*cols+i] - 127.5) * 0.0078125
		}
	}

	return outData
}

func flatten2DTo1D(arr [][]float32) []float32 {
	var res []float32
	for _, a := range arr {
		res = append(res, a...)
	}
	return res
}

func flatten4DTo1D(data [][][][]float64) []float64 {
	result := make([]float64, 0)

	for i := range data {
		for j := range data[i] {
			for k := range data[i][j] {
				for l := range data[i][j][k] {
					result = append(result, data[i][j][k][l])
				}
			}
		}
	}

	return result
}

func tensorsToFloat64Slices(tensors []*tf.Tensor) ([][]float64, error) {
	result := make([][]float64, len(tensors))

	for i, t := range tensors {
		// Get the data from the *tf.Tensor as a 1D []float32 slice
		data, ok := t.Value().([]float32)
		if !ok {
			return nil, fmt.Errorf("expected tensor to be of type []float32, but got %T", t.Value())
		}

		// Convert the []float32 to []float64
		float64Data := make([]float64, len(data))
		for j, v := range data {
			float64Data[j] = float64(v)
		}

		// Append the []float64 to the result
		result[i] = float64Data
	}

	return result, nil
}

func CalibrateBox(bbox [][]float32, reg [][]float32) [][]float32 {
	n := len(bbox)
	w := make([]float32, n)
	h := make([]float32, n)
	for i := 0; i < n; i++ {
		w[i] = bbox[i][2] - bbox[i][0] + 1
		h[i] = bbox[i][3] - bbox[i][1] + 1
	}
	regM := make([][]float32, n)
	for i := range regM {
		regM[i] = make([]float32, 4)
		regM[i][0] = w[i]
		regM[i][1] = h[i]
		regM[i][2] = w[i]
		regM[i][3] = h[i]
	}
	aug := make([][]float32, n)
	for i := range aug {
		aug[i] = make([]float32, 4)
		aug[i][0] = regM[i][0] * reg[i][0]
		aug[i][1] = regM[i][1] * reg[i][1]
		aug[i][2] = regM[i][2] * reg[i][2]
		aug[i][3] = regM[i][3] * reg[i][3]
	}
	for i := 0; i < n; i++ {
		bbox[i][0] += aug[i][0]
		bbox[i][1] += aug[i][1]
		bbox[i][2] += aug[i][2]
		bbox[i][3] += aug[i][3]
	}
	return bbox
}

func float32SliceToPoint2fSlice(float32Slice []float32) []gocv.Point2f {
	if len(float32Slice)%2 != 0 {
		panic("float32Slice length must be even.")
	}

	point2fSlice := make([]gocv.Point2f, len(float32Slice)/2)

	for i := 0; i < len(float32Slice); i += 2 {
		point2fSlice[i/2] = gocv.Point2f{X: float32Slice[i], Y: float32Slice[i+1]}
	}

	return point2fSlice
}

func preprocess(img gocv.Mat, bbox []float32, landmark []gocv.Point2f) gocv.Mat {
	var M gocv.Mat
	imageSize := []int{112, 112}

	if landmark != nil {
		src := []gocv.Point2f{
			{X: 30.2946, Y: 51.6963},
			{X: 65.5318, Y: 51.5014},
			{X: 48.0252, Y: 71.7366},
			{X: 33.5493, Y: 92.3655},
			{X: 62.7299, Y: 92.2041},
		}

		if imageSize[1] == 112 {
			for i := range src {
				src[i].X += 8.0
			}
		}

		dst := gocv.NewPoint2fVectorFromPoints(landmark)
		srcVec := gocv.NewPoint2fVectorFromPoints(src)

		M = gocv.EstimateAffine2D(dst, srcVec)
	}

	if M.Empty() {
		var det []float32
		if bbox == nil {
			det = make([]float32, 4)
			det[0] = float32(img.Cols()) * 0.0625
			det[1] = float32(img.Rows()) * 0.0625
			det[2] = float32(img.Cols()) - det[0]
			det[3] = float32(img.Rows()) - det[1]
		} else {
			det = bbox
		}
		margin := 44
		bb := make([]int, 4)
		bb[0] = int(math.Max(float64(det[0])-float64(margin/2), 0))
		bb[1] = int(math.Max(float64(det[1])-float64(margin/2), 0))
		bb[2] = int(math.Min(float64(det[2])+float64(margin/2), float64(img.Cols())))
		bb[3] = int(math.Min(float64(det[3])+float64(margin/2), float64(img.Rows())))

		ret := img.Region(image.Rect(bb[0], bb[1], bb[2], bb[3]))
		if len(imageSize) > 0 {
			gocv.Resize(ret, &ret, image.Point{X: imageSize[1], Y: imageSize[0]}, 0, 0, gocv.InterpolationLinear)
		}
		return ret
	} else {
		warped := gocv.NewMat()
		gocv.WarpAffine(img, &warped, M, image.Point{X: imageSize[1], Y: imageSize[0]})
		return warped
	}
}

func ConvertToFloats(img image.Image) [][]float32 {
	bounds := img.Bounds()
	height := bounds.Dy()
	width := bounds.Dx()

	data := make([][]float32, height)
	for y := 0; y < height; y++ {
		data[y] = make([]float32, width)
		for x := 0; x < width; x++ {
			grayColor := color.GrayModel.Convert(img.At(x, y)).(color.Gray)
			data[y][x] = float32(grayColor.Y)
		}
	}

	return data
}

func generateEmbeddings(imgs [][][]float32) *tensor.Dense {
	mean := []float32{0.0, 0.0, 0.0}
	std := []float32{1.0, 1.0, 1.0}
	trans := tensor.New(tensor.WithShape(3), tensor.WithBacking([]float32{
		1.0 / std[0], 0.0, 0.0,
		0.0, 1.0 / std[1], 0.0,
		0.0, 0.0, 1.0 / std[2],
	}))

	permutedImgs := tensor.New(tensor.WithShape(len(imgs), 3, len(imgs[0]), len(imgs[0][0])), tensor.WithBacking(make([]float32, len(imgs)*3*len(imgs[0])*len(imgs[0][0]))))
	for i, img := range imgs {
		imageImage := ConvertFloatImage(img)
		listImg := resize.Resize(224, 224, imageImage, resize.Lanczos3)
		img = ConvertToFloats(listImg)
		for y := 0; y < len(img); y++ {
			for x := 0; x < len(img[y]); x++ {
				pixel := img[y][x]
				permutedImgs.SetAt(pixel/255.0-mean[2], i, 2, y, x)
			}
		}
	}
	transformedImgs := tensor.New(tensor.WithShape(len(imgs), 3, 224, 224), tensor.WithBacking(make([]float64, len(imgs)*3*224*224)))
	tensor.Transpose(permutedImgs, 0, 3, 1, 2)
	tensor.Mul(transformedImgs, trans)
	return transformedImgs
}

func normalize(vecs [][]float64) ([][]float64, []float64) {
	r := len(vecs)
	c := len(vecs[0])
	norms := make([]float64, r)
	for i := 0; i < r; i++ {
		norm := 0.0
		for _, v := range vecs[i] {
			norm += v * v
		}
		norms[i] = math.Sqrt(norm)
	}
	normed := make([][]float64, r)
	for i := 0; i < r; i++ {
		normed[i] = make([]float64, c)
		for j := 0; j < c; j++ {
			normed[i][j] = vecs[i][j] / norms[i]
		}
	}
	return normed, norms
}

func cosineSimilarityNoPair(fAnch, fTest [][]float64, isNormed bool) []float64 {
	if !isNormed {
		fAnch, _ = normalize(fAnch)
		fTest, _ = normalize(fTest)
	}
	r1 := len(fAnch)
	r2 := len(fTest)

	result := make([]float64, r1*r2)
	counter := 0

	for i := 0; i < r1; i++ {
		for j := 0; j < r2; j++ {
			sum := 0.0
			for k := range fAnch[i] {
				sum += fAnch[i][k] * fTest[j][k]
			}
			result[counter] = sum
			counter++
		}
	}

	return result
}

func computeSQNoPair(fAnchor, fTest [][]float64) ([]float64, []float64) {
	fAnchor, qAnchor := normalize(fAnchor)
	fTest, qTest := normalize(fTest)
	s := cosineSimilarityNoPair(fAnchor, fTest, true)

	q := make([]float64, len(qAnchor)*len(qTest))
	counter := 0
	for _, i := range qAnchor {
		for _, j := range qTest {
			q[counter] = math.Min(i, j)
			counter++
		}
	}
	return s, q
}

func denseToTFTensor(dense *tensor.Dense) (*tf.Tensor, error) {
	shape := dense.Shape()
	data := dense.Data().([]float32) // Assuming the data type is float32

	// Reshape the data to a 1D slice
	reshapedData := make([]float32, 0, len(data))
	for _, v := range data {
		reshapedData = append(reshapedData, v)
	}

	// Create a *tf.Tensor from the reshaped data
	reshapedDataTensor, err := tf.NewTensor(reshapedData)
	if err != nil {
		return nil, err
	}

	// Convert tensor.Shape to []int64
	int64Shape := make([]int64, len(shape))
	for i, dim := range shape {
		int64Shape[i] = int64(dim)
	}

	// Reshape the *tf.Tensor to match the original shape
	reshapedDataTensor.Reshape(int64Shape)

	return reshapedDataTensor, nil
}

func similarityNoPair(fAnch [][]float64, fTest [][]float64) []float64 {
	s, q := computeSQNoPair(fAnch, fTest)
	alpha := 0.077428
	beta := 0.125926
	omega := make([]float64, len(s))
	for i, v := range s {
		omega[i] = beta*v - alpha
		if omega[i] >= 0 {
			omega[i] = 0
		}
	}

	result := make([]float64, len(s))
	for i := range s {
		result[i] = omega[i]*q[i] + s[i]
	}
	return result
}

func loadNpy(filePath string) ([][]float64, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var matrix [][]float64
	if err := npyio.Read(file, &matrix); err != nil {
		return nil, err
	}

	return matrix, nil
}

func getRegFiles(regDataPath string) ([]string, error) {
	var regFiles []string

	err := filepath.Walk(regDataPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if !info.IsDir() && (strings.HasSuffix(path, ".png") || strings.HasSuffix(path, ".jpg")) {
			regFiles = append(regFiles, path)
		}

		return nil
	})

	if err != nil {
		return nil, err
	}

	return regFiles, nil
}

func getShape(arr [][]float32) (int, int) {
	numRows := len(arr)
	numCols := 0
	if numRows > 0 {
		numCols = len(arr[0])
	}
	return numRows, numCols
}

func main() {
	filename := "./obama.jpg"

	img := gocv.IMRead(filename, gocv.IMReadColor)
	defer img.Close()

	cImg := gocv.NewMat()
	defer cImg.Close()
	gocv.CvtColor(img, &cImg, gocv.ColorBGRToRGB)
	height := cImg.Size()[0]
	width := cImg.Size()[1]
	minDetSize := 12
	minSize := 50
	var scales []float64
	m := float64(minDetSize) / float64(minSize)
	minL := math.Min(float64(height), float64(width)) * m
	factorCount := 0
	factor := 0.709
	for minL > float64(minDetSize) {
		scales = append(scales, m*math.Pow(factor, float64(factorCount)))
		minL *= factor
		factorCount++
	}

	//************************************************************************************
	// detect face
	//************************************************************************************

	////////////////////////////////////////////
	// first stage
	////////////////////////////////////////////
	pnetModel := tg.LoadModel("./mtcnn_pb/pnet_pb", []string{"serve"}, nil)
	slicedIndex := sliceIndex(len(scales))
	var totalBoxes [][]float32
	for _, batch := range slicedIndex {
		localBoxes := detectFirstStage(cImg, pnetModel, scales[batch], 0.6)
		totalBoxes = append(totalBoxes, localBoxes...)
	}

	// remove the Nones
	var validBoxes [][]float32
	for _, box := range totalBoxes {
		if box != nil {
			validBoxes = append(validBoxes, box)
		}
	}
	totalBoxes = validBoxes

	// merge the detection from first stage
	mergedBoxes := nms(totalBoxes, 0.7, "Union")
	var pickedBoxes [][]float32
	for _, idx := range mergedBoxes {
		pickedBoxes = append(pickedBoxes, totalBoxes[idx])
	}

	// refine the boxes
	var refinedBoxes [][]float32
	for _, box := range totalBoxes {
		bbw := box[2] - box[0] + 1
		bbh := box[3] - box[1] + 1
		refinedBox := []float32{box[0] + box[5]*bbw, box[1] + box[6]*bbh, box[2] + box[7]*bbw, box[3] + box[8]*bbh, box[4]}
		refinedBoxes = append(refinedBoxes, refinedBox)
	}
	totalBoxes = convertToSquare(totalBoxes)
	for i := range totalBoxes {
		totalBoxes[i][0] = float32(math.Round(float64(totalBoxes[i][0])))
		totalBoxes[i][1] = float32(math.Round(float64(totalBoxes[i][1])))
		totalBoxes[i][2] = float32(math.Round(float64(totalBoxes[i][2])))
		totalBoxes[i][3] = float32(math.Round(float64(totalBoxes[i][3])))
	}

	////////////////////////////////////////////
	// second stage
	////////////////////////////////////////////
	rnetModel := tg.LoadModel("./mtcnn_pb/rnet_pb", []string{"serve"}, nil)
	numBox := len(totalBoxes)

	// pad the bbox
	dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph := pad(totalBoxes, float32(width), float32(height))
	// (3, 24, 24) is the input shape for RNet
	inputBuf := make([][][][]float32, numBox)
	for i := 0; i < numBox; i++ {
		inputBuf[i] = make([][][]float32, 3)
		for j := 0; j < 3; j++ {
			inputBuf[i][j] = make([][]float32, 24)
			for k := 0; k < 24; k++ {
				inputBuf[i][j][k] = make([]float32, 24)
			}
		}
	}

	for i := 0; i < numBox; i++ {
		tmp := gocv.NewMatWithSize(int(tmph[i]), int(tmpw[i]), gocv.MatTypeCV8UC3)
		//defer tmp.Close()
		scalar := gocv.NewScalar(0, 0, 0, 0)
		tmp.SetTo(scalar)
		roi := img.Region(image.Rect(int(dx[i]), int(dy[i]), int(edx[i]+1), int(edy[i]+1)))
		//defer roi.Close()
		region := tmp.Region(image.Rect(int(x[i]), int(y[i]), int(ex[i]+1), int(ey[i]+1)))
		//defer region.Close()
		roi.CopyTo(&region)
		resizedTmp := gocv.NewMat()
		//defer resizedTmp.Close()
		gocv.Resize(tmp, &resizedTmp, image.Point{X: 24, Y: 24}, 0, 0, gocv.InterpolationDefault)
		inputBuf[0][i] = adjustInput(resizedTmp)
	}

	inputBufTensor, _ := tf.NewTensor(inputBuf)
	output := rnetModel.Exec([]tf.Output{
		rnetModel.Op("PartitionedCall", 0),
	}, map[tf.Output]*tf.Tensor{
		rnetModel.Op("serving_default_input_2", 0): inputBufTensor,
	})
	rNetOutput, ok := output[0].Value().([][][][]float32)
	if !ok {
		fmt.Println("Failed to convert rNetOutput to [][][][]float64")
	}
	flattenOutput := flatten4DTo3D(rNetOutput)

	// filter the total_boxes with threshold
	var passed []int
	for i, row := range flattenOutput[1] {
		if row[1] > 0.7 {
			passed = append(passed, i)
		}
	}
	var secondTotalBoxes [][]float32
	for _, i := range passed {
		secondTotalBoxes = append(secondTotalBoxes, totalBoxes[i])
	}

	if len(secondTotalBoxes) == 0 {
		fmt.Println("return nil")
	}

	var scores [][]float32
	var reg [][]float32
	for _, i := range passed {
		scores = append(scores, []float32{flattenOutput[1][i][1]})
		reg = append(reg, flattenOutput[0][i])
	}

	// nms
	pick := nms(scores, 0.7, "Union")
	var newPickedBoxes [][]float32
	var pickedReg [][]float32
	for _, i := range pick {
		newPickedBoxes = append(newPickedBoxes, secondTotalBoxes[i])
		pickedReg = append(pickedReg, reg[i])
	}
	calibratedBoxes := CalibrateBox(newPickedBoxes, pickedReg)
	squaredBoxes := convertToSquare(calibratedBoxes)
	for i := range squaredBoxes {
		for j := 0; j < 4; j++ {
			squaredBoxes[i][j] = float32(math.Round(float64(squaredBoxes[i][j])))
		}
	}

	////////////////////////////////////////////
	// third stage
	////////////////////////////////////////////
	onetModel := tg.LoadModel("./mtcnn_pb/onet_pb", []string{"serve"}, nil)
	numBox = len(squaredBoxes)
	// pad the bbox
	dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(squaredBoxes, float32(width), float32(height))
	// (3, 48, 48) is the input shape for ONet
	inputBuf = make([][][][]float32, numBox)
	for i := 0; i < numBox; i++ {
		inputBuf[i] = make([][][]float32, 3)
		for j := 0; j < 3; j++ {
			inputBuf[i][j] = make([][]float32, 48)
			for k := 0; k < 24; k++ {
				inputBuf[i][j][k] = make([]float32, 48)
			}
		}
	}

	for i := 0; i < numBox; i++ {
		tmp := gocv.NewMatWithSize(int(tmph[i]), int(tmpw[i]), gocv.MatTypeCV8UC3)
		//defer tmp.Close()
		scalar := gocv.NewScalar(0, 0, 0, 0)
		tmp.SetTo(scalar)
		roi := img.Region(image.Rect(int(dx[i]), int(dy[i]), int(edx[i]+1), int(edy[i]+1)))
		//defer roi.Close()
		region := tmp.Region(image.Rect(int(x[i]), int(y[i]), int(ex[i]+1), int(ey[i]+1)))
		//defer region.Close()
		roi.CopyTo(&region)
		resizedTmp := gocv.NewMat()
		//defer resizedTmp.Close()
		gocv.Resize(tmp, &resizedTmp, image.Point{X: 48, Y: 48}, 0, 0, gocv.InterpolationDefault)
		inputBuf[0][i] = adjustInput(resizedTmp)
	}

	inputBufTensorOnet, _ := tf.NewTensor(inputBuf)
	onetOutput := onetModel.Exec([]tf.Output{
		onetModel.Op("PartitionedCall", 0),
	}, map[tf.Output]*tf.Tensor{
		onetModel.Op("serving_default_input_3", 0): inputBufTensorOnet,
	})
	oNetOutput, ok := onetOutput[0].Value().([][][][]float32)
	if !ok {
		fmt.Println("Failed to convert oNetOutput to [][][][]float64")
	}
	flattenOutput = flatten4DTo3D(oNetOutput)

	// filter the total_boxes with threshold
	var thirdPassed []int
	for i, row := range flattenOutput[2] {
		if row[1] > 0.8 {
			thirdPassed = append(thirdPassed, i)
		}
	}
	var thirdFilteredBoxes [][]float32
	for _, i := range thirdPassed {
		thirdFilteredBoxes = append(thirdFilteredBoxes, squaredBoxes[i])
	}

	if len(thirdFilteredBoxes) == 0 {
		fmt.Println("return nil")
	}

	var thirdScores [][]float32
	var thirdReg [][]float32
	var points [][]float32
	for _, i := range thirdPassed {
		thirdScores = append(thirdScores, []float32{flattenOutput[2][i][1]})
		thirdReg = append(thirdReg, flattenOutput[1][i])
		points = append(points, flattenOutput[0][i])
	}
	bbw := make([]float32, len(thirdScores))
	bbh := make([]float32, len(thirdScores))
	for i, box := range thirdScores {
		bbw[i] = box[2] - box[0] + 1
		bbh[i] = box[3] - box[1] + 1
	}
	for i := range points {
		for j := 0; j < 5; j++ {
			points[i][j] = thirdScores[i][0] + bbw[i]*points[i][j]
		}
		for j := 5; j < 10; j++ {
			points[i][j] = thirdScores[i][1] + bbh[i]*points[i][j]
		}
	}

	// nms
	calibratedBoxes = CalibrateBox(thirdScores, thirdReg)
	pick = nms(calibratedBoxes, 0.7, "Min")

	var thirdPickedBoxes [][]float32
	var pickedPoints [][]float32
	for _, i := range pick {
		thirdPickedBoxes = append(thirdPickedBoxes, calibratedBoxes[i])
		pickedPoints = append(pickedPoints, points[i])
	}
	fmt.Println("return thirdPickedBoxes")

	//************************************************************************************
	// align face
	//************************************************************************************
	if len(thirdPickedBoxes) == 0 || len(pickedPoints) == 0 {
		fmt.Println("return nil")
	}

	var images [][][]float32
	for i := range points {
		p := points[i]
		var p2d [2][5]float32
		for j := 0; j < 5; j++ {
			p2d[0][j] = p[j]
			p2d[1][j] = p[j+5]
		}
		p = make([]float32, 10)
		for j := 0; j < 5; j++ {
			p[j] = p2d[0][j]
			p[j+5] = p2d[1][j]
		}
		b := thirdPickedBoxes[i]
		pPoint2f := float32SliceToPoint2fSlice(p)
		processedImg := preprocess(img, b, pPoint2f)
		sliceImg := matToFloat32Slice(processedImg)
		images = append(images, sliceImg)
	}

	//************************************************************************************
	// recognize face
	//************************************************************************************
	qmfModel := tg.LoadModel("./magface_epoch_00025_pb", []string{"serve"}, nil)
	if len(images) == 0 {
		fmt.Println("return nil")
	}
	transformedFaces := generateEmbeddings(images)
	transformedFacesTFTensor, err := denseToTFTensor(transformedFaces)

	frameEmbeddings := qmfModel.Exec([]tf.Output{
		qmfModel.Op("PartitionedCall", 0),
	}, map[tf.Output]*tf.Tensor{
		qmfModel.Op("serving_default_input.1", 0): transformedFacesTFTensor,
	})

	filePath := "./reg_embeddings.npy"
	regEmbeddings, err := loadNpy(filePath)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	frameEmbeddingsFloat64, err := tensorsToFloat64Slices(frameEmbeddings)
	qmfScores := similarityNoPair(frameEmbeddingsFloat64, regEmbeddings)
	regFiles, _ := getRegFiles("./_data/aligned_camera_data_anchor")
	bSize := len(regFiles)
	nB := int(math.Ceil(float64(len(qmfScores)) / float64(bSize)))

	classIDs := make([]string, nB)
	recScores := make([]float64, nB)
	targetTh := -0.4

	for i := 0; i < nB; i++ {
		startIndex := i * bSize
		endIndex := (i + 1) * bSize
		if endIndex > len(qmfScores) {
			endIndex = len(qmfScores)
		}
		qmfSlice := qmfScores[startIndex:endIndex]

		maxScore := qmfSlice[0]
		maxIndex := 0
		for j, score := range qmfSlice {
			if score > maxScore {
				maxScore = score
				maxIndex = j
			}
		}

		if maxScore > targetTh {
			classIDs[i] = filepath.Base(filepath.Dir(regFiles[maxIndex]))
		} else {
			classIDs[i] = "unknown"
		}
		recScores[i] = maxScore
	}

	fmt.Println(classIDs)
	fmt.Println(recScores)
}
