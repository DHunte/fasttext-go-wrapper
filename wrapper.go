package main

// #cgo LDFLAGS: -L${SRCDIR}/fastText/lib -lfasttext-wrapper -lstdc++ -lm -pthread
// #include <stdlib.h>
// int ft_load_model(char *path);
// int ft_predict(char *query, float *prob, char *buf, int buf_size);
// int ft_get_vector_dimension();
// int ft_get_sentence_vector(char* query_in, float* vector, int vector_size);
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

const (
	_ = iota

	// LabelA is an example prediction value label
	LabelA

	// LabelB is an example prediction value label
	LabelB

	// NoLabel is an example prediction value label
	NoLabel
)

// Model uses FastText for it's prediction
type Model struct {
	isInitialized bool
}

/*func Train(parameters []string) string {
	parameters = append([]string{"placeholder"}, parameters...)
	argv := make([]*C.char, len(parameters))
	for i, s := range parameters {
		cs := C.CString(s)
		defer C.free(unsafe.Pointer(cs))
		argv[i] = cs
	}
	C.ft_run(C.int(len(argv)), &argv[0])
	return ""
}*/

// New should be used to instantiate the model.
// FastTest needs some initialization for the model binary located on `file`.
func NewModel(file string) (*Model, error) {

	status := C.ft_load_model(C.CString(file))

	if status != 0 {
		return nil, fmt.Errorf("Cannot initialize model on `%s`", file)
	}

	return &Model{
		isInitialized: true,
	}, nil
}

// Predict the `keyword`
func (m *Model) Predict(keyword string) (error, string, float64) {

	if !m.isInitialized {
		return errors.New("The FastText model needs to be initialized first. It's should be done inside the `New()` function"), "", 0.0
	}

	resultSize := 32
	result := (*C.char)(C.malloc(C.ulong(resultSize)))

	var cprob C.float

	status := C.ft_predict(
		C.CString(keyword),
		&cprob,
		result,
		C.int(resultSize),
	)
	if status != 0 {
		return fmt.Errorf("Exception when predicting `%s`", keyword), "", 0.0
	}

	// Here's the result from C
	label := C.GoString(result)
	prob := float64(cprob)
	fmt.Println(label, prob)

	C.free(unsafe.Pointer(result))

	return nil, label, prob
}

// GetSentenceVector the `keyword`
func (m *Model) GetSentenceVector(keyword string) ([]float64, error) {

	if !m.isInitialized {
		return nil, errors.New("The FastText model needs to be initialized first. It's should be done inside the `New()` function")
	}

	vecDim := C.ft_get_vector_dimension()
	var cfloat C.float
	result := (*C.float)(C.malloc(C.ulong(vecDim) * C.ulong(unsafe.Sizeof(cfloat))))

	status := C.ft_get_sentence_vector(
		C.CString(keyword),
		result,
		vecDim,
	)

	if status != 0 {
		return nil, fmt.Errorf("Exception when predicting `%s`", keyword)
	}
	p2 := (*[1 << 30]C.float)(unsafe.Pointer(result))
	ret := make([]float64, int(vecDim))
	for i := 0; i < int(vecDim); i++ {
		ret[i] = float64(p2[i])
	}

	C.free(unsafe.Pointer(result))

	return ret, nil
}
