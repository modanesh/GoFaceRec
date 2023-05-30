package main

import (
	"errors"
	"fmt"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"io/ioutil"
	"math"
	"os"
	"reflect"
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

func findExtremeValue(slice interface{}, operation string) float32 {
	switch reflect.TypeOf(slice).Kind() {
	case reflect.Slice:
		val := reflect.ValueOf(slice)
		if val.Len() == 0 {
			return 0 // Return a default value when the slice is empty
		}
		extremeVal := initializeExtremeValue(operation) // Initialize extreme value based on operation
		for i := 0; i < val.Len(); i++ {
			elem := val.Index(i).Interface()
			extreme := findExtremeValue(elem, operation)
			if isBetter(extreme, extremeVal, operation) {
				extremeVal = extreme
			}
		}
		return extremeVal
	default:
		// Handle non-slice types (e.g., single element)
		switch slice := slice.(type) {
		case float32:
			return slice
		default:
			// Return a default value when the element is not a float32
			return 0
		}
	}
}

func storeSliceToFile(slice interface{}, filename string) error {
	// Get the value and kind of the input slice
	value := reflect.ValueOf(slice)
	kind := value.Kind()

	// Ensure the input is a slice
	if kind != reflect.Slice {
		return fmt.Errorf("input is not a slice")
	}

	// Flatten the slice
	flattened, err := flattenSlice(slice)
	if err != nil {
		return err
	}

	// Convert the flattened slice to a string representation
	dataStr := strings.Trim(strings.Join(strings.Fields(fmt.Sprint(flattened)), " "), "[]")

	// Write the string data to a text file
	if _, err := os.Stat(filename); err == nil {
		fmt.Println("File already exists!")
	} else if errors.Is(err, os.ErrNotExist) {
		_ = ioutil.WriteFile(filename, []byte(dataStr), 0644)
	} else {
		fmt.Println("Schrodinger: file may or may not exist. See err for details.", err)
	}
	return nil
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

// Reshape a 1D slice into a 2D slice of the given dimensions
func reshape2D(data []float64, rows, cols int) [][]float64 {
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = data[i*cols : (i+1)*cols]
	}
	return result
}

func svdGolubReinsch(A, U, S, Vt [][]float64) {
	m := len(A)
	n := len(A[0])

	// Use A^T * A to compute eigenvectors and eigenvalues.
	AtA := make([][]float64, n)
	for i := range AtA {
		AtA[i] = make([]float64, n)
		for j := range AtA[i] {
			sum := 0.0
			for k := 0; k < m; k++ {
				sum += A[k][i] * A[k][j]
			}
			AtA[i][j] = sum
		}
	}

	// Compute eigenvalues and eigenvectors of A^T * A.
	eigenValues, eigenVectors := eigen(AtA)

	// Compute singular values and singular vectors.
	for i := 0; i < n; i++ {
		S[i][i] = math.Sqrt(eigenValues[i][0])
		for j := 0; j < n; j++ {
			Vt[i][j] = eigenVectors[i][j]
		}
	}

	// Compute U from A * V.
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += A[i][k] * Vt[k][j]
			}
			U[i][j] = sum / S[j][j]
		}
	}
}

func scaleVector(v []float64, s float64) []float64 {
	result := make([]float64, len(v))
	for i := range v {
		result[i] = v[i] * s
	}
	return result
}

func float32ToMat(data [][][]float32) gocv.Mat {
	height := len(data)
	width := len(data[0])
	channels := len(data[0][0])

	// Create a new Mat from the flat data.
	sizes := []int{height, width, channels}
	mat := gocv.NewMatWithSizes(sizes, gocv.MatTypeCV32FC3)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			for k := 0; k < channels; k++ {
				mat.SetFloatAt3(i, j, k, data[i][j][k])
			}
		}
	}
	return mat
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

func reshape(slice []float32) [][]float32 {
	var reshaped [][]float32
	for i := 0; i < len(slice); i += 9 {
		end := i + 9
		if end > len(slice) {
			end = len(slice)
		}
		reshaped = append(reshaped, slice[i:end])
	}
	return reshaped
}

func getShape(slice interface{}) []int {
	var shape []int
	val := reflect.ValueOf(slice)
	for val.Kind() == reflect.Slice {
		shape = append(shape, val.Len())
		val = val.Index(0)
	}
	return shape
}
