package main

import (
	"fmt"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	tg "github.com/galeone/tfgo"
	"github.com/sbinet/npyio"
	"gocv.io/x/gocv"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/lapack/lapack64"
	"image"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"time"
)

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

func extractData(npArray [][][]float32, index int) [][]float32 {
	// Assuming the dimensions of your slice are as follows:
	var dim1, dim2 = len(npArray), len(npArray[0])

	// Creating a 2D slice to store the extracted values
	var extractedData = make([][]float32, dim1)
	for i := 0; i < dim1; i++ {
		extractedData[i] = make([]float32, dim2)
	}

	// Extracting the desired values
	for i := 0; i < dim1; i++ {
		for j := 0; j < dim2; j++ {
			extractedData[i][j] = npArray[i][j][index]
		}
	}
	return extractedData
}

func fix(val float32) float32 {
	if val < 0 {
		return float32(math.Ceil(float64(val)))
	} else {
		return float32(math.Floor(float64(val)))
	}
}

func computeQ1(bb [][]float32, stride, scale float32) [][]float32 {
	q1 := make([][]float32, len(bb))
	for i, pair := range bb {
		q1[i] = make([]float32, len(pair))
		for j, val := range pair {
			q1[i][j] = fix((stride*val + 1) / scale)
		}
	}
	return q1
}

func computeQ2(bb [][]float32, stride, scale, cellsize float32) [][]float32 {
	q2 := make([][]float32, len(bb))
	for i, pair := range bb {
		q2[i] = make([]float32, len(pair))
		for j, val := range pair {
			q2[i][j] = fix((stride*val + cellsize) / scale)
		}
	}
	return q2
}

func generateBBox(imap [][]float32, reg [][][]float32, scale float64, t float32) [][]float32 {
	stride := 2
	cellsize := 12

	imap = transpose2D(imap)
	dx1 := transpose2D(extractData(reg, 0))
	dy1 := transpose2D(extractData(reg, 1))
	dx2 := transpose2D(extractData(reg, 2))
	dy2 := transpose2D(extractData(reg, 3))

	var ys []int
	var xs []int
	for i := range imap {
		for j := range imap[i] {
			if imap[i][j] >= t {
				ys = append(ys, i)
				xs = append(xs, j)
			}
		}
	}

	if len(ys) == 1 {
		dx1 = flip2D(dx1)
		dy1 = flip2D(dy1)
		dx2 = flip2D(dx2)
		dy2 = flip2D(dy2)
	}

	scores := make([]float32, len(ys))
	for i, y := range ys {
		scores[i] = imap[y][xs[i]]
	}
	regResult := make([][]float32, len(ys))
	for i := range regResult {
		regResult[i] = []float32{
			dx1[ys[i]][xs[i]],
			dy1[ys[i]][xs[i]],
			dx2[ys[i]][xs[i]],
			dy2[ys[i]][xs[i]],
		}
	}
	if len(regResult) == 0 {
		regResult = make([][]float32, 0)
	}

	bb := make([][]float32, len(xs))
	for i := 0; i < len(xs); i++ {
		bb[i] = []float32{float32(ys[i]), float32(xs[i])}
	}

	q1 := computeQ1(bb, float32(stride), float32(scale))
	q2 := computeQ2(bb, float32(stride), float32(scale), float32(cellsize))

	boundingbox := make([][]float32, len(ys))
	for i := range boundingbox {
		boundingbox[i] = append(append(q1[i], q2[i]...), append([]float32{scores[i]}, regResult[i]...)...)
	}

	return boundingbox
}

func transpose2D(matrix [][]float32) [][]float32 {
	rows := len(matrix)
	if rows == 0 {
		return matrix
	}

	cols := len(matrix[0])
	result := make([][]float32, cols)
	for i := range result {
		result[i] = make([]float32, rows)
	}

	for i, row := range matrix {
		for j, val := range row {
			result[j][i] = val
		}
	}

	return result
}

func transpose3D(slice [][][]float32) [][][]float32 {
	var (
		x = len(slice)
		y = len(slice[0])
		z = len(slice[0][0])
	)

	newSlice := make([][][]float32, z)
	for i := range newSlice {
		newSlice[i] = make([][]float32, x)
		for j := range newSlice[i] {
			newSlice[i][j] = make([]float32, y)
		}
	}

	for i, s := range slice {
		for j, ss := range s {
			for k, v := range ss {
				newSlice[k][i][j] = v
			}
		}
	}

	return newSlice
}

func flip2D(matrix [][]float32) [][]float32 {
	for i := 0; i < len(matrix)/2; i++ {
		matrix[i], matrix[len(matrix)-1-i] = matrix[len(matrix)-1-i], matrix[i]
	}
	return matrix
}

func flatten4DTo2D(data [][][][]float32) [][]float32 {
	var dim2, dim3 = len(data[0]), len(data[0][0])

	var extractedData = make([][]float32, dim2)
	for i := 0; i < dim2; i++ {
		extractedData[i] = make([]float32, dim3)
	}

	for i := 0; i < dim2; i++ {
		for j := 0; j < dim3; j++ {
			extractedData[i][j] = data[0][i][j][1]
		}
	}
	return extractedData
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

func initializeExtremeValue(operation string) float32 {
	switch operation {
	case "min":
		return float32(math.Inf(1)) // Initialize with positive infinity for min operation
	case "max":
		return float32(math.Inf(-1)) // Initialize with negative infinity for max operation
	default:
		return 0
	}
}

func isBetter(candidate float32, currentExtreme float32, operation string) bool {
	switch operation {
	case "min":
		return candidate < currentExtreme
	case "max":
		return candidate > currentExtreme
	default:
		return false
	}
}

func flattenSlice(slice interface{}) ([]interface{}, error) {
	value := reflect.ValueOf(slice)
	kind := value.Kind()

	// Ensure the input is a slice
	if kind != reflect.Slice {
		return nil, fmt.Errorf("input is not a slice")
	}

	// Flatten the slice recursively
	var flattened []interface{}
	for i := 0; i < value.Len(); i++ {
		element := value.Index(i)
		elementKind := element.Kind()

		if elementKind == reflect.Slice {
			subSlice, err := flattenSlice(element.Interface())
			if err != nil {
				return nil, err
			}
			flattened = append(flattened, subSlice...)
		} else {
			flattened = append(flattened, element.Interface())
		}
	}

	return flattened, nil
}

func detectFirstStage(img gocv.Mat, net *tg.Model, scale float64, threshold float32) [][]float32 {
	height, width := img.Size()[0], img.Size()[1]
	ws := int(math.Ceil(float64(width) * scale))
	hs := int(math.Ceil(float64(height) * scale))

	imData := gocv.NewMat()
	defer imData.Close()
	gocv.Resize(img, &imData, image.Point{X: ws, Y: hs}, 0, 0, gocv.InterpolationLinear)

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
	order := []int{0, 2, 1, 3}
	reg = transpose(reg, order)
	heatmap2d := flatten4DTo2D(transpose(heatmap, order))

	boxes := generateBBox(heatmap2d, reg[0], scale, threshold)
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

func adjustInput(inData gocv.Mat) [][][][]float32 {
	width := inData.Cols()
	height := inData.Rows()
	channels := inData.Channels()
	outData := make([][][][]float32, 1)
	outData[0] = make([][][]float32, width)

	// Scale the data and expand dimensions
	for w := 0; w < width; w++ {
		outData[0][w] = make([][]float32, height)
		for h := 0; h < height; h++ {
			outData[0][w][h] = make([]float32, channels)
			val := inData.GetVecbAt(h, w)
			for c := 0; c < channels; c++ {
				outData[0][w][h][c] = (float32(val[c]) - 127.5) * 0.0078125
			}
		}
	}
	return outData
}

func matToSlice(inData gocv.Mat) [][][]float32 {
	width := inData.Rows()
	height := inData.Cols()
	channels := inData.Channels()
	outData := make([][][]float32, width)

	for w := 0; w < width; w++ {
		outData[w] = make([][]float32, height)
		for h := 0; h < height; h++ {
			outData[w][h] = make([]float32, channels)
			val := inData.GetVecbAt(w, h)
			for c := 0; c < channels; c++ {
				outData[w][h][c] = float32(val[c]) / 255
			}
		}
	}
	return outData
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

func preprocessMat(img gocv.Mat, bbox []float32, landmark [][]float32) gocv.Mat {
	var M [][]float64 = nil
	var det, bb []float32 = nil, nil

	if landmark != nil {
		src := [][]float64{
			{30.2946, 51.6963},
			{65.5318, 51.5014},
			{48.0252, 71.7366},
			{33.5493, 92.3655},
			{62.7299, 92.2041},
		}

		for i := 0; i < len(src); i++ {
			src[i][0] += 8.0
		}
		dst := float32ToFloat64(landmark)
		M = umeyama(dst, src, true)
		M = M[:len(M)-1]
		fmt.Printf("Matrix M: %v\n", M)
	}

	if M == nil {
		if bbox == nil {
			det = make([]float32, 4)
			det[0] = float32(img.Cols()) * 0.0625
			det[1] = float32(img.Rows()) * 0.0625
			det[2] = float32(img.Cols()) - det[0]
			det[3] = float32(img.Rows()) - det[1]
		} else {
			det = bbox
		}

		bb = make([]float32, 4)
		bb[0] = float32(math.Max(float64(det[0])-float64(44.0/2), 0))
		bb[1] = float32(math.Max(float64(det[1])-float64(44.0/2), 0))
		bb[2] = float32(math.Min(float64(det[2])+float64(44.0/2), float64(img.Cols())))
		bb[3] = float32(math.Min(float64(det[3])+float64(44.0/2), float64(img.Rows())))

		ret := img.Region(image.Rect(int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])))
		gocv.Resize(ret, &ret, image.Point{X: 112, Y: 112}, 0, 0, gocv.InterpolationLinear)
		return ret
	} else {
		warped := gocv.NewMat()
		mMat := float64ToMat(M)
		val := img.GetVecbAt(0, 0)
		fmt.Printf("img[0,0]: %v\n", val)
		val = img.GetVecbAt(24, 10)
		fmt.Printf("img[24,10]: %v\n", val)

		gocv.WarpAffine(img, &warped, mMat, image.Point{X: 112, Y: 112})
		val = warped.GetVecbAt(0, 0)
		fmt.Printf("warped[0,0]: %v\n", val)
		val = warped.GetVecbAt(0, 1)
		fmt.Printf("warped[0,1]: %v\n", val)

		return warped
	}
}

func umeyama(src, dst [][]float64, estimateScale bool) [][]float64 {
	num := len(src)
	dim := len(src[0])

	// Compute mean of src and dst.
	srcMean := make([]float64, dim)
	dstMean := make([]float64, dim)
	for i := 0; i < num; i++ {
		for j := 0; j < dim; j++ {
			srcMean[j] += src[i][j]
			dstMean[j] += dst[i][j]
		}
	}
	for j := 0; j < dim; j++ {
		srcMean[j] /= float64(num)
		dstMean[j] /= float64(num)
	}

	// Subtract mean from src and dst.
	srcDemean := make([][]float64, num)
	dstDemean := make([][]float64, num)
	for i := range srcDemean {
		srcDemean[i] = make([]float64, dim)
		dstDemean[i] = make([]float64, dim)
	}
	for i := 0; i < num; i++ {
		for j := 0; j < dim; j++ {
			srcDemean[i][j] = src[i][j] - srcMean[j]
			dstDemean[i][j] = dst[i][j] - dstMean[j]
		}
	}
	fmt.Printf("src: %v\n", src)
	fmt.Printf("srcMean: %v\n", srcMean)
	fmt.Printf("srcDemean: %v\n", srcDemean)

	// Eq. (38).
	A := make([][]float64, dim)
	for i := range A {
		A[i] = make([]float64, dim)
	}
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			for k := 0; k < num; k++ {
				A[i][j] += dstDemean[k][i] * srcDemean[k][j]
			}
			A[i][j] /= float64(num)
		}
	}

	// Eq. (39).
	d := make([]float64, dim)
	for i := range d {
		d[i] = 1.0
	}
	if determinant(A) < 0 {
		d[dim-1] = -1
	}

	T := make([][]float64, dim+1)
	for i := range T {
		T[i] = make([]float64, dim+1)
		for j := range T[i] {
			if i == j {
				T[i][j] = 1.0
			}
		}
	}

	V, S, U := svd(A)
	fmt.Printf("Matrix A: %v\n", A)
	fmt.Printf("Matrix U: %v\n", U)
	fmt.Printf("Matrix V: %v\n", V)
	fmt.Printf("Vector d: %v\n", d)
	fmt.Printf("Matrix S: %v\n", S)

	// Eq. (40) and (43).
	rank := matrixRank(A)
	if rank == 0 {
		for i := range T {
			for j := range T[i] {
				T[i][j] = math.NaN()
			}
		}
		return T
	} else if rank == dim-1 {
		if determinant(U)*determinant(V) > 0 {
			for i := 0; i < dim; i++ {
				for j := 0; j < dim; j++ {
					T[i][j] = 0
					for k := 0; k < dim; k++ {
						T[i][j] += U[i][k] * V[k][j]
					}
				}
			}
		} else {
			s := d[dim-1]
			d[dim-1] = -1
			Temp := make([][]float64, dim)
			for i := range Temp {
				Temp[i] = make([]float64, dim)
			}
			for i := 0; i < dim; i++ {
				for j := 0; j < dim; j++ {
					if i == j {
						Temp[i][j] = d[i]
					}
				}
			}
			for i := 0; i < dim; i++ {
				for j := 0; j < dim; j++ {
					T[i][j] = 0
					for k := 0; k < dim; k++ {
						T[i][j] += U[i][k] * Temp[k][j]
					}
				}
			}
			for i := 0; i < dim; i++ {
				for j := 0; j < dim; j++ {
					Temp[i][j] = 0
					for k := 0; k < dim; k++ {
						Temp[i][j] += T[i][k] * V[k][j]
					}
				}
			}
			T = Temp
			d[dim-1] = s
		}
	} else {
		Temp := make([][]float64, dim)
		for i := range Temp {
			Temp[i] = make([]float64, dim)
		}
		for i := 0; i < dim; i++ {
			for j := 0; j < dim; j++ {
				if i == j {
					Temp[i][j] = d[i]
				}
			}
		}
		for i := 0; i < dim; i++ {
			for j := 0; j < dim; j++ {
				T[i][j] = 0
				for k := 0; k < dim; k++ {
					T[i][j] += U[i][k] * Temp[k][j]
				}
			}
		}
		for i := 0; i < dim; i++ {
			for j := 0; j < dim; j++ {
				Temp[i][j] = 0
				for k := 0; k < dim; k++ {
					Temp[i][j] += T[i][k] * V[k][j]
				}
			}
		}
		T = Temp
	}
	fmt.Printf("Matrix T 0: %v\n", T)

	var scale float64
	if estimateScale {
		// Eq. (41) and (42).
		var srcVar float64
		for i := 0; i < dim; i++ {
			var temp float64
			for j := 0; j < num; j++ {
				temp += (src[j][i] - srcMean[i]) * (src[j][i] - srcMean[i])
			}
			srcVar += temp / float64(num)
		}
		var sumSD float64
		for i := 0; i < dim; i++ {
			sumSD += S[0][i] * d[i]
		}
		scale = 1.0 / srcVar * sumSD
		fmt.Printf("srcVar: %v\n", srcVar)
		fmt.Printf("sumSD: %v\n", sumSD)
	} else {
		scale = 1.0
	}
	fmt.Printf("Scale: %v\n", scale)

	for i := 0; i < dim; i++ {
		var temp float64
		for j := 0; j < dim; j++ {
			temp += T[i][j] * srcMean[j]
		}
		T[i][dim] = dstMean[i] - scale*temp
	}
	fmt.Printf("Matrix T 1: %v\n", T)

	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			T[i][j] *= scale
		}
	}
	fmt.Printf("Matrix T 2: %v\n", T)
	return T
}

func matrixRank(A [][]float64) int {
	U, _, _ := svd(A)
	rank := 0
	for i := range U {
		if U[i][i] > 1e-10 {
			rank++
		}
	}
	return rank
}

func determinant(A [][]float64) float64 {
	n := len(A)
	if n == 2 {
		return A[0][0]*A[1][1] - A[0][1]*A[1][0]
	}

	det := 0.0
	for i := 0; i < n; i++ {
		subMatrix := make([][]float64, n-1)
		for j := range subMatrix {
			subMatrix[j] = make([]float64, n-1)
			copy(subMatrix[j], A[j+1])
			subMatrix[j] = append(subMatrix[j][:i], subMatrix[j][i+1:]...)
		}

		det += math.Pow(-1, float64(i)) * A[0][i] * determinant(subMatrix)
	}

	return det
}

func svd(A [][]float64) (U, S, Vt [][]float64) {
	m := len(A)
	n := len(A[0])

	// Convert A to column-major order.
	data := make([]float64, m*n)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			data[i*m+j] = A[j][i]
		}
	}

	U = make([][]float64, m)
	for i := range U {
		U[i] = make([]float64, m)
	}

	S = make([][]float64, m)
	for i := range S {
		S[i] = make([]float64, n)
	}

	Vt = make([][]float64, n)
	for i := range Vt {
		Vt[i] = make([]float64, n)
	}

	// Compute SVD using gonum/lapack.
	blasData := blas64.General{
		Rows:   m,
		Cols:   n,
		Stride: n,
		Data:   data,
	}
	blasU := blas64.General{
		Rows:   m,
		Cols:   m,
		Stride: m,
		Data:   make([]float64, m*m),
	}
	blasVt := blas64.General{
		Rows:   n,
		Cols:   n,
		Stride: n,
		Data:   make([]float64, n*n),
	}
	work := make([]float64, 1)
	lwork := -1

	// Compute work size.
	ok := lapack64.Gesvd(lapack.SVDAll, lapack.SVDAll, blasData, blasU, blasVt, S[0], work, lwork)

	if !ok {
		panic("SVD failed")
	}

	// Allocate work with the correct size.
	lwork = int(work[0])
	work = make([]float64, lwork)

	// Compute SVD.
	ok = lapack64.Gesvd(lapack.SVDAll, lapack.SVDAll, blasData, blasU, blasVt, S[0], work, lwork)

	if !ok {
		panic("SVD failed")
	}

	// Convert U, S, Vt back to row-major order.
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			U[i][j] = blasU.Data[j*m+i]
		}
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			S[i][j] = S[i][j]
		}
	}

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			Vt[i][j] = blasVt.Data[j*n+i]
		}
	}

	return U, S, Vt
}

func eigen(A [][]float64) (values, vectors [][]float64) {
	// Perform eigenvalue decomposition using QR algorithm with shifts.
	n := len(A)

	values = make([][]float64, n)
	vectors = make([][]float64, n)
	for i := range values {
		values[i] = make([]float64, 1)
		vectors[i] = make([]float64, n)
	}

	B := make([][]float64, n)
	for i := range B {
		B[i] = make([]float64, n)
		copy(B[i], A[i])
	}

	for iter := 0; iter < 50; iter++ {
		// Check if B is diagonal.
		sumOffDiagonal := 0.0
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if i != j {
					sumOffDiagonal += B[i][j] * B[i][j]
				}
			}
		}

		if sumOffDiagonal <= 1e-12 {
			break
		}

		// QR decomposition of B.
		Q, R := qr(B)

		// B = R * Q
		B = matmul(R, Q)
	}

	for i := 0; i < n; i++ {
		values[i][0] = B[i][i]
		for j := 0; j < n; j++ {
			vectors[i][j] = B[i][j]
		}
	}

	return values, vectors
}

func qr(A [][]float64) (Q, R [][]float64) {
	m := len(A)
	n := len(A[0])

	Q = make([][]float64, m)
	for i := range Q {
		Q[i] = make([]float64, n)
	}

	R = make([][]float64, n)
	for i := range R {
		R[i] = make([]float64, n)
	}

	for j := 0; j < n; j++ {
		v := make([]float64, m)
		for i := 0; i < m; i++ {
			v[i] = A[i][j]
		}

		for k := 0; k < j; k++ {
			R[k][j] = dotproduct(Q[k], v)
			for i := 0; i < m; i++ {
				v[i] -= R[k][j] * Q[i][k]
			}
		}

		R[j][j] = norm(v)
		for i := 0; i < m; i++ {
			Q[i][j] = v[i] / R[j][j]
		}
	}

	return Q, R
}

func dotproduct(u, v []float64) float64 {
	sum := 0.0
	for i := range u {
		sum += u[i] * v[i]
	}
	return sum
}

func norm(v []float64) float64 {
	sum := 0.0
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

func matmul(A, B [][]float64) [][]float64 {
	m := len(A)
	n := len(B[0])
	p := len(B)

	C := make([][]float64, m)
	for i := range C {
		C[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			sum := 0.0
			for k := 0; k < p; k++ {
				sum += A[i][k] * B[k][j]
			}
			C[i][j] = sum
		}
	}

	return C
}

func float32ToFloat64(data32 [][]float32) [][]float64 {
	data64 := make([][]float64, len(data32))
	for i := range data32 {
		data64[i] = make([]float64, len(data32[i]))
		for j := range data32[i] {
			data64[i][j] = float64(data32[i][j])
		}
	}
	return data64
}

func float64ToMat(data [][]float64) gocv.Mat {
	rows := len(data)
	cols := len(data[0])

	sizes := []int{rows, cols}
	mat := gocv.NewMatWithSizes(sizes, gocv.MatTypeCV32F)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			mat.SetFloatAt(i, j, float32(data[i][j]))
		}
	}
	return mat
}

func generateEmbeddings(imgs [][][]float32) *tf.Tensor {
	transposedImgs := transpose3D(imgs)
	permutedImgs, _ := tf.NewTensor(transposedImgs)
	permutedImgs.Reshape([]int64{1, 3, 112, 112})
	return permutedImgs
}

func normalize(vecs [][]float32) ([][]float32, []float32) {
	r := len(vecs)
	c := len(vecs[0])
	norms := make([]float32, r)
	for i := 0; i < r; i++ {
		norm := 0.0
		for _, v := range vecs[i] {
			norm += float64(v * v)
		}
		norms[i] = float32(math.Sqrt(norm))
	}
	normed := make([][]float32, r)
	for i := 0; i < r; i++ {
		normed[i] = make([]float32, c)
		for j := 0; j < c; j++ {
			normed[i][j] = vecs[i][j] / norms[i]
		}
	}
	return normed, norms
}

func cosineSimilarityNoPair(fAnch, fTest [][]float32, isNormed bool) []float32 {
	if !isNormed {
		fAnch, _ = normalize(fAnch)
		fTest, _ = normalize(fTest)
	}
	r1 := len(fAnch)
	r2 := len(fTest)

	result := make([]float32, r1*r2)
	counter := 0

	for i := 0; i < r1; i++ {
		for j := 0; j < r2; j++ {
			sum := float32(0.0)
			for k := range fAnch[i] {
				sum += fAnch[i][k] * fTest[j][k]
			}
			result[counter] = sum
			counter++
		}
	}

	return result
}

func computeSQNoPair(fAnchor, fTest [][]float32) ([]float32, []float32) {
	fAnchor, qAnchor := normalize(fAnchor)
	fTest, qTest := normalize(fTest)
	s := cosineSimilarityNoPair(fAnchor, fTest, true)

	q := make([]float32, len(qAnchor)*len(qTest))
	counter := 0
	for _, i := range qAnchor {
		for _, j := range qTest {
			q[counter] = float32(math.Min(float64(i), float64(j)))
			counter++
		}
	}
	return s, q
}

func similarityNoPair(fAnch, fTest [][]float32) []float32 {
	s, q := computeSQNoPair(fAnch, fTest)
	alpha := float32(0.077428)
	beta := float32(0.125926)
	omega := make([]float32, len(s))
	for i, v := range s {
		omega[i] = beta*v - alpha
		if omega[i] >= 0 {
			omega[i] = 0
		}
	}

	result := make([]float32, len(s))
	for i := range s {
		result[i] = omega[i]*q[i] + s[i]
	}
	return result
}

func loadNpy(filePath string) ([][]float32, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var flattened []float32
	if err := npyio.Read(file, &flattened); err != nil {
		return nil, err
	}

	// Assuming the second dimension is 512
	const dim2 = 512
	dim1 := len(flattened) / dim2

	matrix := make([][]float32, dim1)
	for i := range matrix {
		matrix[i] = make([]float32, dim2)
		for j := range matrix[i] {
			matrix[i][j] = flattened[i*dim2+j]
		}
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

func refineBoxes(totalBoxes [][]float32) [][]float32 {
	bbW := make([]float32, len(totalBoxes))
	bbH := make([]float32, len(totalBoxes))

	for i := range totalBoxes {
		bbW[i] = totalBoxes[i][2] - totalBoxes[i][0] + 1
		bbH[i] = totalBoxes[i][3] - totalBoxes[i][1] + 1
	}

	refined := make([][]float32, len(totalBoxes))
	for i := range totalBoxes {
		refined[i] = make([]float32, 5)
		refined[i][0] = totalBoxes[i][0] + totalBoxes[i][5]*bbW[i]
		refined[i][1] = totalBoxes[i][1] + totalBoxes[i][6]*bbH[i]
		refined[i][2] = totalBoxes[i][2] + totalBoxes[i][7]*bbW[i]
		refined[i][3] = totalBoxes[i][3] + totalBoxes[i][8]*bbH[i]
		refined[i][4] = totalBoxes[i][4]
	}

	return refined
}

func firstStage(scales []float64, img gocv.Mat, pnetModel *tg.Model) [][]float32 {
	slicedIndex := sliceIndex(len(scales))
	var totalBoxes [][]float32
	for _, batch := range slicedIndex {
		localBoxes := detectFirstStage(img, pnetModel, scales[batch], 0.6)
		totalBoxes = append(totalBoxes, localBoxes...)
	}

	// merge the detection from first stage
	fiveTotalBoxes := make([][]float32, len(totalBoxes))
	for i := range fiveTotalBoxes {
		fiveTotalBoxes[i] = totalBoxes[i][:5] // Using slice notation a[i:j]
	}
	mergedBoxes := nms(fiveTotalBoxes, 0.7, "Union")

	var pickedBoxes [][]float32
	for _, idx := range mergedBoxes {
		pickedBoxes = append(pickedBoxes, totalBoxes[idx])
	}

	// refine the boxes
	refinedBoxes := refineBoxes(pickedBoxes)

	totalBoxes = convertToSquare(refinedBoxes)
	for i := range totalBoxes {
		totalBoxes[i][0] = float32(math.Round(float64(totalBoxes[i][0])))
		totalBoxes[i][1] = float32(math.Round(float64(totalBoxes[i][1])))
		totalBoxes[i][2] = float32(math.Round(float64(totalBoxes[i][2])))
		totalBoxes[i][3] = float32(math.Round(float64(totalBoxes[i][3])))
	}
	return totalBoxes
}

func secondStage(totalBoxes [][]float32, width, height int, img gocv.Mat, rnetModel *tg.Model) [][]float32 {
	numBox := len(totalBoxes)

	// pad the bbox
	dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph := pad(totalBoxes, float32(width), float32(height))
	inputBuf := make([][][][]float32, numBox)
	for i := 0; i < numBox; i++ {
		tmp := gocv.NewMatWithSize(int(tmph[i]), int(tmpw[i]), gocv.MatTypeCV8UC3)
		defer tmp.Close()

		imgRoi := img.Region(image.Rect(int(x[i]), int(y[i]), int(ex[i])+1, int(ey[i])+1))
		defer imgRoi.Close()

		tmpRoi := tmp.Region(image.Rect(int(dx[i]), int(dy[i]), int(edx[i])+1, int(edy[i])+1))
		defer tmpRoi.Close()

		imgRoi.CopyTo(&tmpRoi)

		resized := gocv.NewMat()
		defer resized.Close()
		gocv.Resize(tmp, &resized, image.Pt(24, 24), 0, 0, gocv.InterpolationLinear)

		inputBuf[i] = adjustInput(resized)[0]
	}

	inputBufTensor, _ := tf.NewTensor(inputBuf)
	output := rnetModel.Exec([]tf.Output{
		rnetModel.Op("PartitionedCall", 0),
		rnetModel.Op("PartitionedCall", 1),
	}, map[tf.Output]*tf.Tensor{
		rnetModel.Op("serving_default_input_2", 0): inputBufTensor,
	})
	rNetOutput0, ok := output[0].Value().([][]float32)
	if !ok {
		fmt.Println("Failed to convert rNetOutput to [][]float64")
	}
	rNetOutput1, ok := output[1].Value().([][]float32)
	if !ok {
		fmt.Println("Failed to convert rNetOutput to [][]float64")
	}

	score := make([]float32, len(rNetOutput1))
	for i, v := range rNetOutput1 {
		score[i] = v[1]
	}

	passed := make([]int, 0)
	for i, v := range score {
		if v > 0.7 {
			passed = append(passed, i)
		}
	}

	totalBoxesNew := make([][]float32, 0)
	for _, i := range passed {
		totalBoxesNew = append(totalBoxesNew, totalBoxes[i])
	}
	totalBoxes = totalBoxesNew
	if len(totalBoxes) == 0 {
		fmt.Println("No face detected! Stop!")
	}

	for i, idx := range passed {
		totalBoxes[i][4] = rNetOutput1[idx][1]
	}

	reg := make([][]float32, len(passed))
	for i, idx := range passed {
		reg[i] = rNetOutput0[idx]
	}

	// nms
	pick := nms(totalBoxes, 0.7, "Union")
	var newPickedBoxes [][]float32
	var pickedReg [][]float32
	for _, i := range pick {
		newPickedBoxes = append(newPickedBoxes, totalBoxes[i])
		pickedReg = append(pickedReg, reg[i])
	}

	calibratedBoxes := CalibrateBox(newPickedBoxes, pickedReg)

	squaredBoxes := convertToSquare(calibratedBoxes)
	for i := range squaredBoxes {
		for j := 0; j < 4; j++ {
			squaredBoxes[i][j] = float32(math.Round(float64(squaredBoxes[i][j])))
		}
	}
	return squaredBoxes
}

func thirdStage(squaredBoxes [][]float32, width, height int, img gocv.Mat, onetModel *tg.Model) ([][]float32, [][]float32) {
	numBox := len(squaredBoxes)
	totalBoxes := squaredBoxes
	// pad the bbox
	dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph := pad(totalBoxes, float32(width), float32(height))

	// (3, 48, 48) is the input shape for ONet
	inputBuf := make([][][][]float32, numBox)
	for i := 0; i < numBox; i++ {
		tmp := gocv.NewMatWithSize(int(tmph[i]), int(tmpw[i]), gocv.MatTypeCV8UC3)
		defer tmp.Close()

		imgRoi := img.Region(image.Rect(int(x[i]), int(y[i]), int(ex[i])+1, int(ey[i])+1))
		defer imgRoi.Close()

		tmpRoi := tmp.Region(image.Rect(int(dx[i]), int(dy[i]), int(edx[i])+1, int(edy[i])+1))
		defer tmpRoi.Close()

		imgRoi.CopyTo(&tmpRoi)

		resized := gocv.NewMat()
		defer resized.Close()
		gocv.Resize(tmp, &resized, image.Pt(48, 48), 0, 0, gocv.InterpolationLinear)

		inputBuf[i] = adjustInput(resized)[0]
	}

	inputBufTensor, _ := tf.NewTensor(inputBuf)
	output := onetModel.Exec([]tf.Output{
		onetModel.Op("PartitionedCall", 0),
		onetModel.Op("PartitionedCall", 1),
		onetModel.Op("PartitionedCall", 2),
	}, map[tf.Output]*tf.Tensor{
		onetModel.Op("serving_default_input_3", 0): inputBufTensor,
	})
	oNetOutput0, ok := output[1].Value().([][]float32)
	if !ok {
		fmt.Println("Failed to convert rNetOutput to [][]float64")
	}
	oNetOutput1, ok := output[0].Value().([][]float32)
	if !ok {
		fmt.Println("Failed to convert rNetOutput to [][]float64")
	}
	oNetOutput2, ok := output[2].Value().([][]float32)
	if !ok {
		fmt.Println("Failed to convert rNetOutput to [][]float64")
	}

	score := make([]float32, len(oNetOutput2))
	for i, v := range oNetOutput2 {
		score[i] = v[1]
	}

	passed := make([]int, 0)
	for i, v := range score {
		if v > 0.8 {
			passed = append(passed, i)
		}
	}

	totalBoxesNew := make([][]float32, 0)
	for _, i := range passed {
		totalBoxesNew = append(totalBoxesNew, totalBoxes[i])
	}
	totalBoxes = totalBoxesNew
	if len(totalBoxes) == 0 {
		fmt.Println("No face detected! Stop!")
	}

	for i, idx := range passed {
		totalBoxes[i][4] = oNetOutput2[idx][1]
	}

	reg := make([][]float32, len(passed))
	for i, idx := range passed {
		reg[i] = oNetOutput1[idx]
	}

	points := make([][]float32, len(passed))
	for i, idx := range passed {
		points[i] = oNetOutput0[idx]
	}

	bbW := make([]float32, len(totalBoxes))
	bbH := make([]float32, len(totalBoxes))

	for i := range totalBoxes {
		bbW[i] = totalBoxes[i][2] - totalBoxes[i][0] + 1
		bbH[i] = totalBoxes[i][3] - totalBoxes[i][1] + 1
	}

	for i := 0; i < len(points); i++ {
		for j := 0; j < 5; j++ {
			points[i][j] = totalBoxes[i][0] + bbW[i]*points[i][j]
			points[i][j+5] = totalBoxes[i][1] + bbH[i]*points[i][j+5]
		}
	}

	// nms
	calibratedBoxes := CalibrateBox(totalBoxes, reg)
	pick := nms(calibratedBoxes, 0.7, "Min")

	var thirdPickedBoxes [][]float32
	var pickedPoints [][]float32
	for _, i := range pick {
		thirdPickedBoxes = append(thirdPickedBoxes, calibratedBoxes[i])
		pickedPoints = append(pickedPoints, points[i])
	}
	return thirdPickedBoxes, pickedPoints
}

func alignFace(thirdPickedBoxes, pickedPoints [][]float32, img gocv.Mat) [][][][]float32 {
	if len(thirdPickedBoxes) == 0 || len(pickedPoints) == 0 {
		fmt.Println("return nil")
	}
	var pImgs [][][][]float32
	for i := 0; i < len(pickedPoints); i++ {
		p := make([][]float32, 5)
		for j := 0; j < 5; j++ {
			p[j] = make([]float32, 2)
			p[j][0] = pickedPoints[i][j]
			p[j][1] = pickedPoints[i][j+5]
		}
		b := thirdPickedBoxes[i]
		matFace := preprocessMat(img, b, p)

		val := matFace.GetVecbAt(0, 0)
		fmt.Printf("matFace[0,0]: %v\n", val)
		val = matFace.GetVecbAt(12, 23)
		fmt.Printf("matFace[12,23]: %v\n", val)

		sliceFace := matToSlice(matFace)
		pImgs = append(pImgs, sliceFace)
	}
	return pImgs
}

func recognizeFace(pImgs [][][][]float32, qmfModel *tg.Model, regEmbeddings [][]float32, bSize int, regFiles []string) {
	if len(pImgs) == 0 {
		fmt.Println("return nil")
	}

	for _, pImg := range pImgs {
		transformedFaces := generateEmbeddings(pImg)
		recognitionStart_1 := time.Now()
		frameEmbeddings := qmfModel.Exec([]tf.Output{
			qmfModel.Op("PartitionedCall", 0),
		}, map[tf.Output]*tf.Tensor{
			qmfModel.Op("serving_default_input.1", 0): transformedFaces,
		})
		elapsed := time.Since(recognitionStart_1)
		elapsedMilliseconds := elapsed.Milliseconds()
		fmt.Printf("----------------> ForwardPass execution time: %d milliseconds\n", elapsedMilliseconds)

		frameEmbeddingsFloat32, ok := frameEmbeddings[0].Value().([][]float32)
		if !ok {
			fmt.Println("Failed to convert rNetOutput to [][]float32")
		}

		qmfScores := similarityNoPair(frameEmbeddingsFloat32, regEmbeddings)
		nB := int(math.Ceil(float64(len(qmfScores)) / float64(bSize)))

		classIDs := make([]string, nB)
		recScores := make([]float32, nB)
		targetTh := float32(-0.4)

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

		fmt.Println("----------------- classIDs ------->", classIDs)
		fmt.Println("----------------- recScores ------->", recScores)
	}
}

func main() {

	//************************************************************************************
	// preprocessing and loading
	//************************************************************************************
	filename := os.Args[1]

	img := gocv.IMRead(filename, gocv.IMReadColor)
	defer img.Close()

	height := img.Size()[0]
	width := img.Size()[1]
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
	pnetModel := tg.LoadModel("./models/mtcnn_pb/pnet_pb", []string{"serve"}, nil)
	rnetModel := tg.LoadModel("./models/mtcnn_pb/rnet_pb", []string{"serve"}, nil)
	onetModel := tg.LoadModel("./models/mtcnn_pb/onet_pb", []string{"serve"}, nil)
	qmfModel := tg.LoadModel("./models/magface_epoch_00025_pb", []string{"serve"}, nil)

	filePath := "./_data/aligned_camera_data_anchor/embeddings.npy"
	regEmbeddings, err := loadNpy(filePath)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	regFiles, _ := getRegFiles("./_data/aligned_camera_data_anchor")
	bSize := len(regFiles)

	//************************************************************************************
	// detect face
	//************************************************************************************

	////////////////////////////////////////////
	// first stage
	////////////////////////////////////////////
	start := time.Now()
	totalBoxes := firstStage(scales, img, pnetModel)
	elapsed := time.Since(start)
	elapsedMilliseconds := elapsed.Milliseconds()
	fmt.Printf("----------------> First detection execution time: %d milliseconds\n", elapsedMilliseconds)

	////////////////////////////////////////////
	// second stage
	////////////////////////////////////////////
	detection2Start := time.Now()
	squaredBoxes := secondStage(totalBoxes, width, height, img, rnetModel)
	elapsed = time.Since(detection2Start)
	elapsedMilliseconds = elapsed.Milliseconds()
	fmt.Printf("----------------> Second detection execution time: %d milliseconds\n", elapsedMilliseconds)

	//////////////////////////////////////////////
	//// third stage
	//////////////////////////////////////////////
	detection3Start := time.Now()
	thirdPickedBoxes, pickedPoints := thirdStage(squaredBoxes, width, height, img, onetModel)
	elapsed = time.Since(detection3Start)
	elapsedMilliseconds = elapsed.Milliseconds()
	fmt.Printf("----------------> Third detection execution time: %d milliseconds\n", elapsedMilliseconds)

	//************************************************************************************
	// align face
	//************************************************************************************
	alignStart := time.Now()
	pImgs := alignFace(thirdPickedBoxes, pickedPoints, img)
	elapsed = time.Since(alignStart)
	elapsedMilliseconds = elapsed.Milliseconds()
	fmt.Printf("----------------> Face alignment execution time: %d milliseconds\n", elapsedMilliseconds)

	////************************************************************************************
	//// recognize face
	////************************************************************************************
	recognitionStart := time.Now()
	recognizeFace(pImgs, qmfModel, regEmbeddings, bSize, regFiles)
	elapsed = time.Since(recognitionStart)
	elapsedMilliseconds = elapsed.Milliseconds()
	fmt.Printf("----------------> Face recognition execution time: %d milliseconds\n", elapsedMilliseconds)
	elapsed = time.Since(start)
	elapsedMilliseconds = elapsed.Milliseconds()
	fmt.Printf("----------------> Total execution time: %d milliseconds\n", elapsedMilliseconds)
}
