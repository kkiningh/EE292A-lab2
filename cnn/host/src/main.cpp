#define NOMINMAX // so that windows.h does not define min/max macros

#include <algorithm>
#include <iostream>
#include <fstream>
// #include <time.h>
// #include <sys/time.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "../../shared/defines.h"
#include "../../shared/utils.h"

// TODO: If you want to define constants, you can do it here
#define CONV1_WEIGHTS (5 * 5 * 32 * 1)
#define CONV1_BIAS (32)
#define CONV2_WEIGHTS (5 * 5 * 64 * 32)
#define CONV2_BIAS (64)
#define DENSE1_WEIGHTS (7 * 7 * 64 * 256)
#define DENSE1_BIAS (256)
#define DENSE2_WEIGHTS (256 * 10)
#define DENSE2_BIAS (10)

using namespace aocl_utils;

// OpenCL Global Variables.
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_kernel kernel;
cl_program program;

cl_uchar *input_images = NULL, *output_guesses = NULL, *reference_guesses = NULL;
cl_float *input_weights = NULL;
cl_mem input_images_buffer, output_guesses_buffer;

// TODO: add buffers for your weights
cl_float* conv1_weights, *conv1_bias;
cl_float* conv2_weights, *conv2_bias;
cl_float* dense1_weights, *dense1_bias;
cl_float* dense2_weights, *dense2_bias;

cl_mem conv1_weights_buffer, conv1_bias_buffer;
cl_mem conv2_weights_buffer, conv2_bias_buffer;
cl_mem dense1_weights_buffer, dense1_bias_buffer;
cl_mem dense2_weights_buffer, dense2_bias_buffer;

// Global variables.
std::string imagesFilename;
std::string labelsFilename;
std::string aocxFilename;
std::string deviceInfo;
unsigned int n_items;
bool use_fixed_point;
bool use_single_workitem;

// Function prototypes.
void classify();
void initCL();
void cleanup();
void teardown(int exit_status = 1);

bool read_weights(const char *filename, float *weights, size_t count) {
  FILE *f = fopen(filename, "rb");
  if (f == NULL){
    printf("ERROR: could not open %s\n",filename);
    return false;
  }
  int read_elements = fread(weights, sizeof(float), count, f);
  fclose(f);
  
  if (read_elements != count){
    printf("ERROR: read incorrect number of weights from %s\n", filename);
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  // Parsing command line arguments.
  Options options(argc, argv);

  if(options.has("images")) {
    imagesFilename = options.get<std::string>("images");
  } else {
    imagesFilename = "t10k-images.idx3-ubyte";
    printf("Defaulting to images file \"%s\"\n", imagesFilename.c_str());
  }
  
  if(options.has("labels")) {
    labelsFilename = options.get<std::string>("labels");  
  } else {
    labelsFilename = "t10k-labels.idx1-ubyte";
    printf("Defaulting to labels file \"%s\"\n", labelsFilename.c_str());
  }

  // Relative path to aocx filename option.
  if(options.has("aocx")) {
    aocxFilename = options.get<std::string>("aocx");  
  } else {
    aocxFilename = "linear_classifier_fp";
    printf("Defaulting to aocx file \"%s\"\n", aocxFilename.c_str());
  }
  
  // Read in the images and labels
  n_items = parse_MNIST_images(imagesFilename.c_str(), &input_images);
  if (n_items <= 0){
    printf("ERROR: Failed to parse images file.\n");
    return -1;
  }
  if (n_items != parse_MNIST_labels(labelsFilename.c_str(), &reference_guesses)){
    printf("ERROR: Number of labels does not match number of images\n");
    return -1;
  }

  // TODO: Uncomment this to verify on a smaller set of examples
  n_items = 1;
  
  // Initializing OpenCL and the kernels.
  output_guesses = (cl_uchar*)alignedMalloc(sizeof(cl_uchar) * n_items);

  conv1_weights = (cl_float*)alignedMalloc(sizeof(cl_float) * CONV1_WEIGHTS);
  conv1_bias = (cl_float*)alignedMalloc(sizeof(cl_float) * CONV1_BIAS);
  conv2_weights = (cl_float*)alignedMalloc(sizeof(cl_float) * CONV2_WEIGHTS);
  conv2_bias = (cl_float*)alignedMalloc(sizeof(cl_float) * CONV2_BIAS);
  dense1_weights = (cl_float*)alignedMalloc(sizeof(cl_float) * DENSE1_WEIGHTS);
  dense1_bias = (cl_float*)alignedMalloc(sizeof(cl_float) * DENSE1_BIAS);
  dense2_weights = (cl_float*)alignedMalloc(sizeof(cl_float) * DENSE2_WEIGHTS);
  dense2_bias = (cl_float*)alignedMalloc(sizeof(cl_float) * DENSE2_BIAS);
  
  // TODO: Read in weights from weights files
  read_weights("weights/conv1_weights", conv1_weights, CONV1_WEIGHTS);
  read_weights("weights/conv1_bias", conv1_bias, CONV1_BIAS);
  read_weights("weights/conv2_weights", conv2_weights, CONV2_WEIGHTS);
  read_weights("weights/conv2_bias", conv2_bias, CONV2_BIAS);
  read_weights("weights/dense1_weights", dense1_weights, DENSE1_WEIGHTS);
  read_weights("weights/dense1_bias", dense1_bias, DENSE1_BIAS);
  read_weights("weights/dense2_weights", dense2_weights, DENSE2_WEIGHTS);
  read_weights("weights/dense2_bias", dense2_bias, DENSE2_BIAS);
  
  initCL();

  // Start measuring time
  double start = get_wall_time();
  
  // Call the classifier.
  classify();
  
  // Stop measuring time.
  double end = get_wall_time();
  printf("TIME ELAPSED: %.2f ms\n", end - start);
   
  int correct = 0;
  for (unsigned i = 0; i < n_items; i++){
    if (output_guesses[i] == reference_guesses[i]) correct++;
  }
  printf("Classifier accuracy: %.2f%%\n", (float)correct*100/n_items);
  
  // Teardown OpenCL.
  teardown(0);
}

void classify() {
  size_t size = 1;
  cl_int status;
  cl_event event;
  const size_t global_work_size = n_items;
  
  // Create kernel input and output buffers.
  input_images_buffer = clCreateBuffer(
    context, CL_MEM_READ_ONLY, sizeof(unsigned char) * FEATURE_COUNT * n_items, NULL, &status);
  checkError(status, "Error: could not create input image buffer");
  output_guesses_buffer = clCreateBuffer(
    context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * n_items, NULL, &status);
  checkError(status, "Error: could not create output guesses buffer");

  // Create buffers for layer weights
  conv1_weights_buffer = clCreateBuffer(
    context, CL_MEM_WRITE_ONLY, sizeof(float) * CONV1_WEIGHTS, NULL, &status);
  checkError(status, "Error: could not create conv1 buffer");

  conv1_bias_buffer = clCreateBuffer(
    context, CL_MEM_WRITE_ONLY, sizeof(float) * CONV1_BIAS, NULL, &status);
  checkError(status, "Error: could not create conv1 bias buffer");

  conv2_weights_buffer = clCreateBuffer(
    context, CL_MEM_WRITE_ONLY, sizeof(float) * CONV2_WEIGHTS, NULL, &status);
  checkError(status, "Error: could not create conv2 weights buffer");

  conv2_bias_buffer = clCreateBuffer(
    context, CL_MEM_WRITE_ONLY, sizeof(float) * CONV2_BIAS, NULL, &status);
  checkError(status, "Error: could not create conv2 bias buffer");

  dense1_weights_buffer = clCreateBuffer(
    context, CL_MEM_WRITE_ONLY, sizeof(float) * DENSE1_WEIGHTS, NULL, &status);
  checkError(status, "Error: could not create dense1 weights buffer");

  dense1_bias_buffer = clCreateBuffer(
    context, CL_MEM_WRITE_ONLY, sizeof(float) * DENSE1_BIAS, NULL, &status);
  checkError(status, "Error: could not create dense1 bias buffer");

  dense2_weights_buffer = clCreateBuffer(
    context, CL_MEM_WRITE_ONLY, sizeof(float) * DENSE2_WEIGHTS, NULL, &status);
  checkError(status, "Error: could not create dense2 weights buffer");

        dense2_bias_buffer = clCreateBuffer(
    context, CL_MEM_WRITE_ONLY, sizeof(float) * DENSE2_BIAS, NULL, &status);
  checkError(status, "Error: could not create dense2 bias buffer");
  
  // Copy data to kernel input buffer.
  // TODO: Copy weights for your layers as well
  status = clEnqueueWriteBuffer(queue, input_images_buffer, CL_TRUE, 0,
    sizeof(unsigned char) * FEATURE_COUNT * n_items, input_images, 0, NULL, NULL);
  checkError(status, "Error: could not copy data into device");
  
  status = clEnqueueWriteBuffer(queue, conv1_weights_buffer, CL_TRUE, 0,
    sizeof(float) * CONV1_WEIGHTS, conv1_weights, 0, NULL, NULL);
  checkError(status, "Error: could not copy data into device");

  status = clEnqueueWriteBuffer(queue, conv1_bias_buffer, CL_TRUE, 0,
    sizeof(float) * CONV1_BIAS, conv1_bias, 0, NULL, NULL);
  checkError(status, "Error: could not copy data into device");

  status = clEnqueueWriteBuffer(queue, conv2_weights_buffer, CL_TRUE, 0,
    sizeof(float) * CONV2_WEIGHTS, conv2_weights, 0, NULL, NULL);
  checkError(status, "Error: could not copy data into device");

  status = clEnqueueWriteBuffer(queue, conv2_bias_buffer, CL_TRUE, 0,
    sizeof(float) * CONV2_BIAS, conv2_bias, 0, NULL, NULL);
  checkError(status, "Error: could not copy data into device");
  
  status = clEnqueueWriteBuffer(queue, dense1_weights_buffer, CL_TRUE, 0,
    sizeof(float) * DENSE1_WEIGHTS, dense1_weights, 0, NULL, NULL);
  checkError(status, "Error: could not copy data into device");

  status = clEnqueueWriteBuffer(queue, dense1_bias_buffer, CL_TRUE, 0,
    sizeof(float) * DENSE1_BIAS, dense1_bias, 0, NULL, NULL);
  checkError(status, "Error: could not copy data into device");

  status = clEnqueueWriteBuffer(queue, dense2_weights_buffer, CL_TRUE, 0,
    sizeof(float) * DENSE2_WEIGHTS, dense2_weights, 0, NULL, NULL);
  checkError(status, "Error: could not copy data into device");

  status = clEnqueueWriteBuffer(queue, dense2_bias_buffer, CL_TRUE, 0,
    sizeof(float) * DENSE2_BIAS, dense2_bias, 0, NULL, NULL);
  checkError(status, "Error: could not copy data into device");

  // Set the arguments for data_in, data_out and sobel kernels.
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input_images_buffer);
  checkError(status, "Error: could not set argument 0");

  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&conv1_weights_buffer);
  checkError(status, "Error: could not set argument 1");
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&conv1_bias_buffer);
  checkError(status, "Error: could not set argument 2");

  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&conv2_weights_buffer);
  checkError(status, "Error: could not set argument 3");
  status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&conv2_bias_buffer);
  checkError(status, "Error: could not set argument 4");

  status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&dense1_weights_buffer);
  checkError(status, "Error: could not set argument 5");
  status = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&dense1_bias_buffer);
  checkError(status, "Error: could not set argument 6");

  status = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&dense2_weights_buffer);
  checkError(status, "Error: could not set argument 7");
  status = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&dense2_bias_buffer);
  checkError(status, "Error: could not set argument 8");

  status = clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&output_guesses_buffer);
  checkError(status, "Error: could not set argument 9");

  
  // Enqueue the kernel. //
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size,
      NULL, 0, NULL, &event);
  checkError(status, "Error: failed to launch data_in");
  
  // Wait for command queue to complete pending events.
  printf("Waiting for kernel to finish..?\n");
  status = clFinish(queue);
  printf("Kernel has finished\n");
  checkError(status, "Kernel failed to finish");

  clReleaseEvent(event);
  
  // Read output buffer from kernel.
  status = clEnqueueReadBuffer(queue, output_guesses_buffer, CL_TRUE, 0,
      sizeof(unsigned char) * n_items, output_guesses, 0, NULL, NULL);
  checkError(status, "Error: could not copy data from device");
}

void initCL() {
  cl_int status;

  // Start everything at NULL to help identify errors.
  kernel = NULL;
  queue = NULL;
  
  // Locate files via. relative paths.
  if(!setCwdToExeDir()) {
    teardown();
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA");
  if (platform == NULL) {
    teardown();
  }

  // Get the first device.
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  checkError (status, "Error: could not query devices");

  char info[256];
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(info), info, NULL);
  deviceInfo = info;

  // Create the context.
  context = clCreateContext(0, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Error: could not create OpenCL context");

  // Create the command queues for the kernels.
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  // Create the program.
  std::string binary_file = getBoardBinaryFile(aocxFilename.c_str(), device);
  std::cout << "Using AOCX: " << binary_file << "\n";
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.
  status = clBuildProgram(program, 1, &device, "", NULL, NULL);
  checkError(status, "Error: could not build program");
  
  // Create the kernel - name passed in here must match kernel name in the original CL file.
  kernel = clCreateKernel(program, "linear_classifier", &status);
  checkError(status, "Failed to create kernel");
}

void cleanup() {
  // Called from aocl_utils::check_error, so there's an error.
  teardown(-1);
}

void teardown(int exit_status) {
  if(kernel) clReleaseKernel(kernel);
  if(queue) clReleaseCommandQueue(queue);
  if(input_images) alignedFree(input_images);
  if(input_weights) alignedFree(input_weights);
  if(reference_guesses) alignedFree(reference_guesses);
  if(output_guesses) alignedFree(output_guesses);
  if(input_images_buffer) clReleaseMemObject(input_images_buffer);
  if(output_guesses_buffer) clReleaseMemObject(output_guesses_buffer);
  if(program) clReleaseProgram(program);
  if(context) clReleaseContext(context);
  
  exit(exit_status);
}
