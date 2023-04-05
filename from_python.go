package main

import (
	"fmt"
	"github.com/nfnt/resize"
	"gocv.io/x/gocv"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gorgonia.org/tensor"
	"image"
	"image/color"
	"math"
	"reflect"
	"sort"
)

// FloatImage represents a custom image type that satisfies the image.Image interface
type FloatImage struct {
	data [][]float64
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

// ConvertFloatImage converts [][]float64 to FloatImage
func ConvertFloatImage(data [][]float64) *FloatImage {
	return &FloatImage{data: data}
}

func matToFloat64Slice(mat gocv.Mat) [][]float64 {
	rows, cols := mat.Rows(), mat.Cols()
	slice := make([][]float64, rows)

	for i := 0; i < rows; i++ {
		rowSlice := make([]float64, cols)
		for j := 0; j < cols; j++ {
			val := mat.GetFloatAt(i, j)
			rowSlice[j] = float64(val)
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

func nms(boxes [][]float64, overlapThreshold float64, mode string) []int {
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
	var x1, y1, x2, y2, score []float64
	for _, box := range boxes {
		x1 = append(x1, box[0])
		y1 = append(y1, box[1])
		x2 = append(x2, box[2])
		y2 = append(y2, box[3])
		score = append(score, box[4])
	}

	area := make([]float64, len(boxes))
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
			xx1[j] = math.Max(x1[i], x1[idxs[j]])
			yy1[j] = math.Max(y1[i], y1[idxs[j]])
			xx2[j] = math.Min(x2[i], x2[idxs[j]])
			yy2[j] = math.Min(y2[i], y2[idxs[j]])
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
				overlap[j] = inter[j] / math.Min(area[i], area[idxs[j]])
			} else {
				overlap[j] = inter[j] / (area[i] + area[idxs[j]] - inter[j])
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

func detectFirstStage(img gocv.Mat, net gocv.Net, scale float64, threshold float64) [][]float64 {
	height, width := img.Size()[0], img.Size()[1]
	ws := int(math.Ceil(float64(width) * scale))
	hs := int(math.Ceil(float64(height) * scale))

	imData := gocv.NewMat()
	defer imData.Close()
	gocv.Resize(img, &imData, image.Point{X: ws, Y: hs}, 0, 0, gocv.InterpolationLinear)

	inputBuf := adjustInput(imData)
	netOutput := net.predict(inputBuf)
	boxes := generateBbox(netOutput[1], netOutput[0], scale, threshold)

	if len(boxes) == 0 {
		return nil
	}

	// nms
	pick := nms(boxes, 0.5, "Union")
	var pickedBoxes [][]float64
	for _, index := range pick {
		pickedBoxes = append(pickedBoxes, boxes[index])
	}
	return pickedBoxes
}

func convertToSquare(bbox [][]float64) [][]float64 {
	squareBbox := make([][]float64, len(bbox))
	for i := 0; i < len(bbox); i++ {
		squareBbox[i] = make([]float64, len(bbox[i]))
		copy(squareBbox[i], bbox[i])
		h := bbox[i][3] - bbox[i][1] + 1
		w := bbox[i][2] - bbox[i][0] + 1
		maxSide := math.Max(h, w)
		squareBbox[i][0] = bbox[i][0] + w*0.5 - maxSide*0.5
		squareBbox[i][1] = bbox[i][1] + h*0.5 - maxSide*0.5
		squareBbox[i][2] = squareBbox[i][0] + maxSide - 1
		squareBbox[i][3] = squareBbox[i][1] + maxSide - 1
	}
	return squareBbox
}

func pad(bboxes [][]float64, w float64, h float64) ([]float64, []float64, []float64, []float64, []float64, []float64, []float64, []float64, []float64, []float64) {
	numBox := len(bboxes)
	tmpw := make([]float64, numBox)
	tmph := make([]float64, numBox)
	for i, box := range bboxes {
		tmpw[i] = box[2] - box[0] + 1
		tmph[i] = box[3] - box[1] + 1
	}

	dx := make([]float64, numBox)
	dy := make([]float64, numBox)
	edx := make([]float64, numBox)
	edy := make([]float64, numBox)

	x := make([]float64, numBox)
	y := make([]float64, numBox)
	ex := make([]float64, numBox)
	ey := make([]float64, numBox)

	copy(x, bboxes[0])
	copy(y, bboxes[1])
	copy(ex, bboxes[2])
	copy(ey, bboxes[3])

	for i := range bboxes {
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

	tmpw64 := make([]float64, numBox)
	tmph64 := make([]float64, numBox)
	for i := range tmpw {
		tmpw64[i] = tmpw[i]
		tmph64[i] = tmph[i]
	}

	return dy, edy, dx, edx, y, ey, x, ex, tmpw64, tmph64
}

func adjustInput(inData gocv.Mat) [][]float64 {
	// adjust the input from (h, w, c) to (1, c, h, w) for network input
	channels := inData.Channels()
	rows, cols := inData.Rows(), inData.Cols()

	// transpose (h, w, c) to (c, h, w)
	outData := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		outData[c] = make([]float64, rows*cols)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				v := inData.GetVecfAt(i, j)[c]
				outData[c][i*cols+j] = float64(v)
			}
		}
	}

	// expand dims to (1, c, h, w)
	outData = [][]float64{flatten(outData)}

	// normalize
	for c := 0; c < channels; c++ {
		for i := 0; i < rows*cols; i++ {
			outData[0][c*rows*cols+i] = (outData[0][c*rows*cols+i] - 127.5) * 0.0078125
		}
	}

	return outData
}

func flatten(arr [][]float64) []float64 {
	var res []float64
	for _, a := range arr {
		res = append(res, a...)
	}
	return res
}

func CalibrateBox(bbox [][]float64, reg [][]float64) [][]float64 {
	n := len(bbox)
	w := make([]float64, n)
	h := make([]float64, n)
	for i := 0; i < n; i++ {
		w[i] = bbox[i][2] - bbox[i][0] + 1
		h[i] = bbox[i][3] - bbox[i][1] + 1
	}
	regM := make([][]float64, n)
	for i := range regM {
		regM[i] = make([]float64, 4)
		regM[i][0] = w[i]
		regM[i][1] = h[i]
		regM[i][2] = w[i]
		regM[i][3] = h[i]
	}
	aug := make([][]float64, n)
	for i := range aug {
		aug[i] = make([]float64, 4)
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

func preprocess(img gocv.Mat, bbox []float64, landmark []float64) gocv.Mat {
	imageSize := [2]int{112, 112}
	src := [][]float64{
		{30.2946, 51.6963},
		{65.5318, 51.5014},
		{48.0252, 71.7366},
		{33.5493, 92.3655},
		{62.7299, 92.2041},
	}
	if imageSize[1] == 112 {
		for i := range src {
			src[i][0] += 8.0
		}
	}
	var dst [][]float64
	for i := 0; i < len(landmark); i += 2 {
		dst = append(dst, []float64{landmark[i], landmark[i+1]})
	}

	M := gocv.EstimateAffinePartial2D(dst, src)

	if M.Empty() {
		var det []float64
		if bbox == nil {
			det = []float64{float64(img.Cols()) * 0.0625, float64(img.Rows()) * 0.0625, float64(img.Cols()) * 0.9375, float64(img.Rows()) * 0.9375}
		} else {
			det = bbox
		}
		margin := 44
		bb := []int{int(math.Max(det[0]-float64(margin/2), 0)), int(math.Max(det[1]-float64(margin/2), 0)), int(math.Min(det[2]+float64(margin/2), float64(img.Cols()))), int(math.Min(det[3]+float64(margin/2), float64(img.Rows())))}
		ret := img.Region(image.Rect(bb[0], bb[1], bb[2], bb[3]))
		retSize := image.Pt(imageSize[1], imageSize[0])
		gocv.Resize(ret, &ret, retSize, 0, 0, gocv.InterpolationLinear)
		return ret
	} else {
		warped := gocv.NewMat()
		imageSizePoint := image.Point{X: imageSize[0], Y: imageSize[1]}
		gocv.WarpAffine(img, &warped, M, imageSizePoint)
		return warped
	}
}

func ConvertToFloats(img image.Image) [][]float64 {
	bounds := img.Bounds()
	height := bounds.Dy()
	width := bounds.Dx()

	data := make([][]float64, height)
	for y := 0; y < height; y++ {
		data[y] = make([]float64, width)
		for x := 0; x < width; x++ {
			grayColor := color.GrayModel.Convert(img.At(x, y)).(color.Gray)
			data[y][x] = float64(grayColor.Y)
		}
	}

	return data
}

func generateEmbeddings(imgs [][][]float64) *tensor.Dense {
	mean := []float64{0.0, 0.0, 0.0}
	std := []float64{1.0, 1.0, 1.0}
	trans := tensor.New(tensor.WithShape(3), tensor.WithBacking([]float64{
		(1.0 / std[0]), 0.0, 0.0,
		0.0, (1.0 / std[1]), 0.0,
		0.0, 0.0, (1.0 / std[2]),
	}))

	permutedImgs := tensor.New(tensor.WithShape(len(imgs), 3, len(imgs[0]), len(imgs[0][0])), tensor.WithBacking(make([]float64, len(imgs)*3*len(imgs[0])*len(imgs[0][0]))))
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

func similarityNoPair(fAnch, fTest *mat.Dense, isNormed bool) []float64 {
	n1, d1 := fAnch.Dims()
	n2, _ := fTest.Dims()

	if !isNormed {
		fAnch = norm(fAnch)
		fTest = norm(fTest)
	}

	fMul := mat.NewDense(n1*n2, d1, nil)
	counter := 0
	for i := 0; i < n1; i++ {
		for j := 0; j < n2; j++ {
			fMul.SetRow(counter, elementwiseMul(fAnch.RowView(i), fTest.RowView(j)))
			counter++
		}
	}

	sum := make([]float64, n1*n2)
	for i := 0; i < n1*n2; i++ {
		row := fMul.RowView(i)
		for j := 0; j < d1; j++ {
			sum[i] += row.At(j, 0)
		}
	}

	return sum
}

func norm(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	normed := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		row := m.RawRowView(i)
		norm := stat.StdDev(row, nil)
		for j := 0; j < c; j++ {
			normed.Set(i, j, row[j]/norm)
		}
	}
	return normed
}

func elementwiseMul(v1, v2 mat.Vector) []float64 {
	n := v1.Len()
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		result[i] = v1.At(i, 0) * v2.At(i, 0)
	}
	return result
}

func computeSQNoPair(fAnchor, fTest *mat.Dense) ([]float64, []float64) {
	fAnchorNorm := norm(fAnchor)
	fTestNorm := norm(fTest)

	s := similarityNoPair(fAnchorNorm, fTestNorm, true)

	n1, _ := qAnchor.Dims()
	n2, _ := qTest.Dims()
	q := make([]float64, n1*n2)
	counter := 0
	for i := 0; i < n1; i++ {
		for j := 0; j < n2; j++ {
			q[counter] = math.Min(qAnchor.At(i, 0), qTest.At(j, 0))
			counter++
		}
	}

	return s, q
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
	slicedIndex := sliceIndex(len(scales))
	var totalBoxes [][]float64
	for _, batch := range slicedIndex {
		localBoxes := detectFirstStage(cImg, pnetModel, scales[batch], 0.6)
		totalBoxes = append(totalBoxes, localBoxes...)
	}

	// remove the Nones
	var validBoxes [][]float64
	for _, box := range totalBoxes {
		if box != nil {
			validBoxes = append(validBoxes, box)
		}
	}
	totalBoxes = validBoxes

	// merge the detection from first stage
	mergedBoxes := nms(totalBoxes, 0.7, "Union")
	var pickedBoxes [][]float64
	for _, idx := range mergedBoxes {
		pickedBoxes = append(pickedBoxes, totalBoxes[idx])
	}

	// refine the boxes
	var refinedBoxes [][]float64
	for _, box := range totalBoxes {
		bbw := box[2] - box[0] + 1
		bbh := box[3] - box[1] + 1
		refinedBox := []float64{box[0] + box[5]*bbw, box[1] + box[6]*bbh, box[2] + box[7]*bbw, box[3] + box[8]*bbh, box[4]}
		refinedBoxes = append(refinedBoxes, refinedBox)
	}
	totalBoxes = convertToSquare(totalBoxes)
	for i := range totalBoxes {
		totalBoxes[i][0] = math.Round(totalBoxes[i][0])
		totalBoxes[i][1] = math.Round(totalBoxes[i][1])
		totalBoxes[i][2] = math.Round(totalBoxes[i][2])
		totalBoxes[i][3] = math.Round(totalBoxes[i][3])
	}

	////////////////////////////////////////////
	// second stage
	////////////////////////////////////////////
	numBox := len(totalBoxes)

	// pad the bbox
	dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph := pad(totalBoxes, float64(width), float64(height))
	// (3, 24, 24) is the input shape for RNet
	inputBuf := make([][][][]float64, numBox)
	for i := 0; i < numBox; i++ {
		inputBuf[i] = make([][][]float64, 3)
		for j := 0; j < 3; j++ {
			inputBuf[i][j] = make([][]float64, 24)
			for k := 0; k < 24; k++ {
				inputBuf[i][j][k] = make([]float64, 24)
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

	output := rnet.Predict(inputBuf)

	// filter the total_boxes with threshold
	var passed []int
	for i, row := range output[1] {
		if row[1] > 0.7 {
			passed = append(passed, i)
		}
	}
	var secondTotalBoxes [][]float64
	for _, i := range passed {
		secondTotalBoxes = append(secondTotalBoxes, totalBoxes[i])
	}

	if len(secondTotalBoxes) == 0 {
		fmt.Println("return nil")
	}

	var scores [][]float64
	var reg [][]float64
	for _, i := range passed {
		scores = append(scores, []float64{output[1][i][1]})
		reg = append(reg, output[0][i])
	}

	// nms
	pick := nms(scores, 0.7, "Union")
	var newPickedBoxes [][]float64
	var pickedReg [][]float64
	for _, i := range pick {
		newPickedBoxes = append(newPickedBoxes, secondTotalBoxes[i])
		pickedReg = append(pickedReg, reg[i])
	}
	calibratedBoxes := CalibrateBox(newPickedBoxes, pickedReg)
	squaredBoxes := convertToSquare(calibratedBoxes)
	for i := range squaredBoxes {
		for j := 0; j < 4; j++ {
			squaredBoxes[i][j] = math.Round(squaredBoxes[i][j])
		}
	}

	////////////////////////////////////////////
	// third stage
	////////////////////////////////////////////
	numBox = len(squaredBoxes)
	// pad the bbox
	dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(squaredBoxes, float64(width), float64(height))
	// (3, 48, 48) is the input shape for ONet
	inputBuf = make([][][][]float64, numBox)
	for i := 0; i < numBox; i++ {
		inputBuf[i] = make([][][]float64, 3)
		for j := 0; j < 3; j++ {
			inputBuf[i][j] = make([][]float64, 48)
			for k := 0; k < 24; k++ {
				inputBuf[i][j][k] = make([]float64, 48)
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

	output = onet.Predict(inputBuf)

	// filter the total_boxes with threshold
	var thirdPassed []int
	for i, row := range output[2] {
		if row[1] > 0.8 {
			thirdPassed = append(thirdPassed, i)
		}
	}
	var thirdFilteredBoxes [][]float64
	for _, i := range thirdPassed {
		thirdFilteredBoxes = append(thirdFilteredBoxes, squaredBoxes[i])
	}

	if len(thirdFilteredBoxes) == 0 {
		fmt.Println("return nil")
	}

	var thirdScores [][]float64
	var thirdReg [][]float64
	var points [][]float64
	for _, i := range thirdPassed {
		thirdScores = append(thirdScores, []float64{output[2][i][1]})
		thirdReg = append(thirdReg, output[1][i])
		points = append(points, output[0][i])
	}
	bbw := make([]float64, len(thirdScores))
	bbh := make([]float64, len(thirdScores))
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

	var thirdPickedBoxes [][]float64
	var pickedPoints [][]float64
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

	var images [][][]float64
	for i := range points {
		p := points[i]
		var p2d [2][5]float64
		for j := 0; j < 5; j++ {
			p2d[0][j] = p[j]
			p2d[1][j] = p[j+5]
		}
		p = make([]float64, 10)
		for j := 0; j < 5; j++ {
			p[j] = p2d[0][j]
			p[j+5] = p2d[1][j]
		}
		b := thirdPickedBoxes[i]
		processedImg := preprocess(img, b, p)
		sliceImg := matToFloat64Slice(processedImg)
		images = append(images, sliceImg)
	}

	//************************************************************************************
	// recognize face
	//************************************************************************************
	if len(images) == 0 {
		fmt.Println("return nil")
	}
	transformedFaces := generateEmbeddings(images)
	embeddings := magface(transformedFaces)

	// until qmagface.py line 24:
	// s, q = self._compute_s_q_no_pair(f_anch, f_test)

}
