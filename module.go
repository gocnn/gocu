package gocu

// #include <cuda.h>
import "C"
import "unsafe"

// CUjitInputType represents the type of input for JIT compilation
type CUjitInputType C.CUjitInputType

const (
	JitInputTypeCubin   CUjitInputType = C.CU_JIT_INPUT_CUBIN     // Compiled device-class-specific device code
	JitInputTypePtx     CUjitInputType = C.CU_JIT_INPUT_PTX       // PTX source code
	JitInputTypeFatBin  CUjitInputType = C.CU_JIT_INPUT_FATBINARY // Fat binary
	JitInputTypeObject  CUjitInputType = C.CU_JIT_INPUT_OBJECT    // Host object with embedded device code
	JitInputTypeLibrary CUjitInputType = C.CU_JIT_INPUT_LIBRARY   // Archive of host objects with embedded device code
	JitInputTypeNvvm    CUjitInputType = C.CU_JIT_INPUT_NVVM      // NVVM intermediate representation
)

// CUjitOption represents JIT compilation options
type CUjitOption C.CUjit_option

const (
	JitMaxRegisters            CUjitOption = C.CU_JIT_MAX_REGISTERS
	JitThreadsPerBlock         CUjitOption = C.CU_JIT_THREADS_PER_BLOCK
	JitWallTime                CUjitOption = C.CU_JIT_WALL_TIME
	JitInfoLogBuffer           CUjitOption = C.CU_JIT_INFO_LOG_BUFFER
	JitInfoLogBufferSizeBytes  CUjitOption = C.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
	JitErrorLogBuffer          CUjitOption = C.CU_JIT_ERROR_LOG_BUFFER
	JitErrorLogBufferSizeBytes CUjitOption = C.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
	JitOptimizationLevel       CUjitOption = C.CU_JIT_OPTIMIZATION_LEVEL
	JitTargetFromContext       CUjitOption = C.CU_JIT_TARGET_FROM_CUCONTEXT
	JitTarget                  CUjitOption = C.CU_JIT_TARGET
	JitFallbackStrategy        CUjitOption = C.CU_JIT_FALLBACK_STRATEGY
	JitGenerateDebugInfo       CUjitOption = C.CU_JIT_GENERATE_DEBUG_INFO
	JitLogVerbose              CUjitOption = C.CU_JIT_LOG_VERBOSE
	JitGenerateLineInfo        CUjitOption = C.CU_JIT_GENERATE_LINE_INFO
	JitCacheMode               CUjitOption = C.CU_JIT_CACHE_MODE
)

// CUmoduleLoadingMode represents module loading mode
type CUmoduleLoadingMode C.CUmoduleLoadingMode

const (
	ModuleEagerLoading CUmoduleLoadingMode = C.CU_MODULE_EAGER_LOADING
	ModuleLazyLoading  CUmoduleLoadingMode = C.CU_MODULE_LAZY_LOADING
)

// Module is a CUDA module
type Module struct{ module C.CUmodule }

// C returns the Module as its C version
func (m Module) c() C.CUmodule { return m.module }

// Function is a CUDA function
type Function struct{ function C.CUfunction }

// C returns the Function as its C version
func (f Function) c() C.CUfunction { return f.function }

// LinkState is a CUDA linker state
type LinkState struct{ state C.CUlinkState }

// C returns the LinkState as its C version
func (l LinkState) c() C.CUlinkState { return l.state }

// Module Loading Functions

// ModuleLoad loads a compute module from file
func ModuleLoad(fname string) (Module, error) {
	var module C.CUmodule
	cfname := C.CString(fname)
	defer C.free(unsafe.Pointer(cfname))
	result := Check(C.cuModuleLoad(&module, cfname))
	return Module{module: module}, result
}

// ModuleLoadData loads a module's data from memory
func ModuleLoadData(image []byte) (Module, error) {
	var module C.CUmodule
	result := Check(C.cuModuleLoadData(&module, unsafe.Pointer(&image[0])))
	return Module{module: module}, result
}

// ModuleLoadDataEx loads a module's data from memory with options
func ModuleLoadDataEx(image []byte, options []CUjitOption, optionValues []unsafe.Pointer) (Module, error) {
	var module C.CUmodule
	var optionsPtr *C.CUjit_option
	var valuesPtr *unsafe.Pointer

	if len(options) > 0 {
		optionsPtr = (*C.CUjit_option)(unsafe.Pointer(&options[0]))
	}
	if len(optionValues) > 0 {
		valuesPtr = &optionValues[0]
	}

	result := Check(C.cuModuleLoadDataEx(&module, unsafe.Pointer(&image[0]),
		C.uint(len(options)), optionsPtr, (*unsafe.Pointer)(unsafe.Pointer(valuesPtr))))
	return Module{module: module}, result
}

// ModuleLoadFatBinary loads a module from a fat binary
func ModuleLoadFatBinary(fatCubin []byte) (Module, error) {
	var module C.CUmodule
	result := Check(C.cuModuleLoadFatBinary(&module, unsafe.Pointer(&fatCubin[0])))
	return Module{module: module}, result
}

// ModuleUnload unloads a module
func ModuleUnload(module Module) error {
	return Check(C.cuModuleUnload(module.c()))
}

// Unload unloads the module (method version)
func (m Module) Unload() error {
	return ModuleUnload(m)
}

// Module Function Management

// ModuleGetFunction returns a function handle from a module
func ModuleGetFunction(module Module, name string) (Function, error) {
	var function C.CUfunction
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	result := Check(C.cuModuleGetFunction(&function, module.c(), cname))
	return Function{function: function}, result
}

// GetFunction returns a function handle from the module (method version)
func (m Module) GetFunction(name string) (Function, error) {
	return ModuleGetFunction(m, name)
}

// ModuleGetFunctionCount returns the number of functions within a module
func ModuleGetFunctionCount(module Module) (uint32, error) {
	var count C.uint
	result := Check(C.cuModuleGetFunctionCount(&count, module.c()))
	return uint32(count), result
}

// GetFunctionCount returns the number of functions within the module (method version)
func (m Module) GetFunctionCount() (uint32, error) {
	return ModuleGetFunctionCount(m)
}

// ModuleEnumerateFunctions returns the function handles within a module
func ModuleEnumerateFunctions(module Module, numFunctions uint32) ([]Function, error) {
	functions := make([]C.CUfunction, numFunctions)
	var functionsPtr *C.CUfunction
	if numFunctions > 0 {
		functionsPtr = &functions[0]
	}

	result := Check(C.cuModuleEnumerateFunctions(functionsPtr, C.uint(numFunctions), module.c()))
	if result != nil {
		return nil, result
	}

	goFunctions := make([]Function, numFunctions)
	for i, f := range functions {
		goFunctions[i] = Function{function: f}
	}
	return goFunctions, nil
}

// EnumerateFunctions returns the function handles within the module (method version)
func (m Module) EnumerateFunctions(numFunctions uint32) ([]Function, error) {
	return ModuleEnumerateFunctions(m, numFunctions)
}

// Module Global Variable Management

// ModuleGetGlobal returns a global pointer from a module
func ModuleGetGlobal(module Module, name string) (DevicePtr, uint64, error) {
	var dptr C.CUdeviceptr
	var bytes C.size_t
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	result := Check(C.cuModuleGetGlobal(&dptr, &bytes, module.c(), cname))
	return DevicePtr(dptr), uint64(bytes), result
}

// GetGlobal returns a global pointer from the module (method version)
func (m Module) GetGlobal(name string) (DevicePtr, uint64, error) {
	return ModuleGetGlobal(m, name)
}

// Module Loading Mode

// ModuleGetLoadingMode queries the lazy loading mode
func ModuleGetLoadingMode() (CUmoduleLoadingMode, error) {
	var mode C.CUmoduleLoadingMode
	result := Check(C.cuModuleGetLoadingMode(&mode))
	return CUmoduleLoadingMode(mode), result
}

// JIT Linking Functions

// LinkCreate creates a pending JIT linker invocation
func LinkCreate(options []CUjitOption, optionValues []unsafe.Pointer) (LinkState, error) {
	var state C.CUlinkState
	var optionsPtr *C.CUjit_option
	var valuesPtr *unsafe.Pointer

	if len(options) > 0 {
		optionsPtr = (*C.CUjit_option)(unsafe.Pointer(&options[0]))
	}
	if len(optionValues) > 0 {
		valuesPtr = &optionValues[0]
	}

	result := Check(C.cuLinkCreate(C.uint(len(options)), optionsPtr,
		(*unsafe.Pointer)(unsafe.Pointer(valuesPtr)), &state))
	return LinkState{state: state}, result
}

// LinkDestroy destroys state for a JIT linker invocation
func LinkDestroy(state LinkState) error {
	return Check(C.cuLinkDestroy(state.c()))
}

// Destroy destroys the linker state (method version)
func (l LinkState) Destroy() error {
	return LinkDestroy(l)
}

// LinkAddData adds an input to a pending linker invocation
func LinkAddData(state LinkState, inputType CUjitInputType, data []byte, name string, options []CUjitOption, optionValues []unsafe.Pointer) error {
	var cname *C.char
	if name != "" {
		cname = C.CString(name)
		defer C.free(unsafe.Pointer(cname))
	}

	var optionsPtr *C.CUjit_option
	var valuesPtr *unsafe.Pointer

	if len(options) > 0 {
		optionsPtr = (*C.CUjit_option)(unsafe.Pointer(&options[0]))
	}
	if len(optionValues) > 0 {
		valuesPtr = &optionValues[0]
	}

	return Check(C.cuLinkAddData(state.c(), C.CUjitInputType(inputType),
		unsafe.Pointer(&data[0]), C.size_t(len(data)), cname,
		C.uint(len(options)), optionsPtr, (*unsafe.Pointer)(unsafe.Pointer(valuesPtr))))
}

// AddData adds an input to the pending linker invocation (method version)
func (l LinkState) AddData(inputType CUjitInputType, data []byte, name string, options []CUjitOption, optionValues []unsafe.Pointer) error {
	return LinkAddData(l, inputType, data, name, options, optionValues)
}

// LinkAddFile adds a file input to a pending linker invocation
func LinkAddFile(state LinkState, inputType CUjitInputType, path string, options []CUjitOption, optionValues []unsafe.Pointer) error {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	var optionsPtr *C.CUjit_option
	var valuesPtr *unsafe.Pointer

	if len(options) > 0 {
		optionsPtr = (*C.CUjit_option)(unsafe.Pointer(&options[0]))
	}
	if len(optionValues) > 0 {
		valuesPtr = &optionValues[0]
	}

	return Check(C.cuLinkAddFile(state.c(), C.CUjitInputType(inputType), cpath,
		C.uint(len(options)), optionsPtr, (*unsafe.Pointer)(unsafe.Pointer(valuesPtr))))
}

// AddFile adds a file input to the pending linker invocation (method version)
func (l LinkState) AddFile(inputType CUjitInputType, path string, options []CUjitOption, optionValues []unsafe.Pointer) error {
	return LinkAddFile(l, inputType, path, options, optionValues)
}

// LinkComplete completes a pending linker invocation
func LinkComplete(state LinkState) ([]byte, error) {
	var cubinOut unsafe.Pointer
	var sizeOut C.size_t

	result := Check(C.cuLinkComplete(state.c(), &cubinOut, &sizeOut))
	if result != nil {
		return nil, result
	}

	// Convert C memory to Go slice
	size := int(sizeOut)
	cubin := C.GoBytes(cubinOut, C.int(size))
	return cubin, nil
}

// Complete completes the pending linker invocation (method version)
func (l LinkState) Complete() ([]byte, error) {
	return LinkComplete(l)
}
