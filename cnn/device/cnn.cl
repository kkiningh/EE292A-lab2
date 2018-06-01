// image is a 28x28xN array (N images) of bytes (each pixel is 8 bit grayscale)
#define IMG_WIDTH 28
#define IMG_HEIGHT 28
#define IMG_CHANNELS 1
#define IMG_SIZE (IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS)

#define IMAGE_INPUT (28 * 28 * 1)

#define CONV1_INPUT_WIDTH 32
#define CONV1_INPUT_HEIGHT 32
#define CONV1_INPUT_CHANNELS 1
#define CONV1_INPUT (32 * 32 * 1)

#define CONV1_FILTER_WIDTH 5
#define CONV1_FILTER_HEIGHT 5
#define CONV1_FILTER_CHANNELS 32

#define MAXPOOL1_INPUT_WIDTH 28
#define MAXPOOL1_INPUT_HEIGHT 28
#define MAXPOOL1_INPUT_CHANNELS 32
#define MAXPOOL1_INPUT (28 * 28 * 32)

#define CONV2_INPUT_WIDTH 18
#define CONV2_INPUT_HEIGHT 18
#define CONV2_INPUT_CHANNELS 32
#define CONV2_INPUT (18 * 18 * 32)

#define CONV2_FILTER_WIDTH 5
#define CONV2_FILTER_HEIGHT 5
#define CONV2_FILTER_CHANNELS 64

#define MAXPOOL2_INPUT_WIDTH 14
#define MAXPOOL2_INPUT_HEIGHT 14
#define MAXPOOL2_INPUT_CHANNELS 64
#define MAXPOOL2_INPUT (14 * 14 * 64)

#define DENSE1_INPUT_WIDTH 7
#define DENSE1_INPUT_HEIGHT 7
#define DENSE1_INPUT_CHANNELS 64
#define DENSE1_INPUT (7 * 7 * 64)

#define DENSE2_INPUT_WIDTH 1
#define DENSE2_INPUT_HEIGHT 1
#define DENSE2_INPUT_CHANNELS 256
#define DENSE2_INPUT (256 * 1 * 1)

#define SOFTMAX_INPUT 10

float relu(float x) {
  return (x < 0.) ? 0. : x;
}

void maxpool_layer(
    local const float * restrict inputs,
    local float * restrict outputs,
    const int input_width,
    const int input_height,
    const int input_channels,
    const int pool_width,
    const int pool_height,
    const int stride_height,
    const int stride_width
) {
  const int output_width = input_width / stride_width;
  const int output_height = input_height / stride_height;
  const int output_channels = input_channels;

  for (int w = 0; w < input_width; w += stride_width) {
    for (int h = 0; h < input_height; h += stride_height) {
      for (int c = 0; c < input_channels; c++) {
        float maxima = -INFINITY;
        for (int ww = 0; ww < pool_width; ww++) {
          for (int hh = 0; hh < pool_height; hh++) {
            const int idx = (w + ww) * input_channels * input_height +
              (h + hh) * input_channels + c;
            float pixel = inputs[idx];
            if(maxima < pixel) {
              maxima = pixel;
            }
          }
        }

        const int o_idx = (w / stride_width) * output_channels * output_height + 
          (h / stride_height) * output_channels + c;
        outputs[o_idx] = maxima;
      }
    }  
  }
}
	
void conv_layer(
    constant float * restrict weights,
    constant float * restrict bias,
    local const float * restrict inputs,
    local float * restrict outputs,
    const int input_width,
    const int input_height,
    const int input_channels,
    const int filter_width,
    const int filter_height,
    const int filter_channels
) {
  const int OW = input_width - filter_width + 1;
  const int OH = input_height - filter_height + 1;
  const int OC = filter_channels;

  for (int f = 0; f < filter_channels; f++) {
    for (int h = 0; h < OH; h++) {
      for (int w = 0; w < OW; w++) {
        /* Compute the output of a single filter */
        float sum = 0;
        for (int c = 0; c < input_channels; c++) {
          for (int ww = 0; ww < filter_width; ww++) {
            for (int hh = 0; hh < filter_height; hh++) {
              float pix = inputs[
                (w + ww) * input_height * input_channels + 
                (h + hh) * input_channels + c];
              float wgt = weights[
                ww * filter_height * input_channels * filter_channels +
                hh * input_channels * filter_channels +
                c * filter_channels + f];
              sum += pix * wgt;
            }
          }
        }
        outputs[w * OC * OH + h * OC + f] = relu(sum + bias[f]);
      }
    }  
  }
}

void dense_layer(
    constant float * restrict weights,
    constant float * restrict bias,
    local const float * restrict inputs,
    local float * restrict outputs,
    const int input_width,
    const int input_height,
    const int input_channels,
    const int output_size,
    bool use_relu
) {
  const int input_size = input_width * input_height;

  for (int x = 0; x < output_size; x++) {
    float sum = 0;
    for (int w = 0; w < input_width; w++) {
      for (int h = 0; h < input_height; h++) {
        for (int c = 0; c < input_channels; c++) {
          float pix = inputs[
            w * input_height * input_channels +
            h * input_channels + c];
          float wgt = weights[
            w * input_height * input_channels * output_size + 
            h * input_channels * output_size + 
            c * output_size + x];
          sum += pix * wgt;
        } 
      }
    }
    float activation = sum + bias[x];
    if (use_relu) {
      outputs[x] = relu(activation);
    } else {
      outputs[x] = activation;
    }
  }
}

void input_pad_layer(
    global const unsigned char * inputs,
    local float * restrict outputs,
    const int input_width,
    const int input_height,
    const int input_channels,
    const int pad_width,
    const int pad_height
) {
  const int OW = input_width + pad_width * 2;
  const int OH = input_height + pad_height * 2;
  const int OC = input_channels;

  for (int w = 0; w < OW; w++) {
    for (int h = 0; h < OH; h++) {
      for (int c = 0; c < input_channels; c++) {
        const int o_idx = w * OC * OH + h * OC + c;
        if (w >= pad_width && h >= pad_height
            && w < input_width + pad_width 
            && h < input_height + pad_height) {
          const int ih = h - pad_height;
          const int iw = w - pad_width;
          const int i_idx = iw * input_height * input_channels + ih
            * input_channels + c;
          outputs[o_idx] =  inputs[i_idx];
        } else {
          outputs[o_idx] = 0;
        }
      }
    }
  }
}

void pad_layer(
    local float * restrict inputs,
    local float * restrict outputs,
    const int input_width,
    const int input_height,
    const int input_channels,
    const int pad_width,
    const int pad_height
) {
  const int OW = input_width + pad_width * 2;
  const int OH = input_height + pad_height * 2;
  const int OC = input_channels;

  for (int w = 0; w < OW; w++) {
    for (int h = 0; h < OH; h++) {
      for (int c = 0; c < input_channels; c++) {
        const int o_idx = w * OC * OH + h * OC + c;
        if (w >= pad_width && h >= pad_height
            && w < input_width + pad_width 
            && h < input_height + pad_height) {
          const int ih = h - pad_height;
          const int iw = w - pad_width;
          const int i_idx = iw * input_height * input_channels + ih
            * input_channels + c;
          outputs[o_idx] =  inputs[i_idx];
        } else {
          outputs[o_idx] = 0;
        }
      }
    }
  }
}

void print_buffer(const char* fmt, local const float * restrict buffer,
    const int width, const int height, const int channels) {
  for (int w = 0; w < width; w++) {
    for (int h = 0; h < height; h++) {
      for (int c = 0; c < 1; c++) {
        printf(fmt, buffer[w * height * channels + h * channels + c]);
      }
    }
    printf("\n");
  }
}

// first argument
__attribute__((reqd_work_group_size(100,1,1))) // change this to change workgroup size
__kernel void linear_classifier(global const unsigned char * restrict images, 
        constant float * restrict conv1_weights,
        constant float * restrict conv1_bias,
        constant float * restrict conv2_weights,
        constant float * restrict conv2_bias,
        constant float * restrict dense1_weights,
        constant float * restrict dense1_bias,
        constant float * restrict dense2_weights,
        constant float * restrict dense2_bias,
        global unsigned char * restrict guesses)
{
  global const unsigned char * image = &images[get_global_id(0) * IMG_SIZE];

  /* input image pad */
  local float conv1_input[CONV1_INPUT];
  input_pad_layer(image, conv1_input,
      IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS,
      2, 2);

  /* CONV LAYER 1 */
  local float maxpool1_input[MAXPOOL1_INPUT];
  conv_layer(
      conv1_weights, conv1_bias, conv1_input, maxpool1_input, 
      CONV1_INPUT_WIDTH, CONV1_INPUT_HEIGHT, CONV1_INPUT_CHANNELS,
      CONV1_FILTER_WIDTH, CONV1_FILTER_HEIGHT, CONV1_FILTER_CHANNELS);

  /* MAXPOOL LAYER */
  local float maxpool1_output[14 * 14 * 32];
  maxpool_layer(
      maxpool1_input, maxpool1_output,
      MAXPOOL1_INPUT_WIDTH, MAXPOOL1_INPUT_HEIGHT, MAXPOOL1_INPUT_CHANNELS,
      2, 2, 2, 2);

  /* pad */
  local float conv2_input[CONV2_INPUT];
  pad_layer(maxpool1_output, conv2_input,
      14, 14, 32, 2, 2);

  //print_buffer("%09.6f, ", conv2_input, 18, 18, 32);
  
  /* CONV LAYER 2 */
  local float maxpool2_input[MAXPOOL2_INPUT];
  conv_layer(
      conv2_weights, conv2_bias, conv2_input, maxpool2_input, 
      18, 18, 32,
      5, 5, 64);

  //print_buffer("%06.2f, ", maxpool2_input, 14, 14, 64);
  
  /* MAXPOOL LAYER 2 */
  local float dense1_input[DENSE1_INPUT];
  maxpool_layer(
      maxpool2_input, dense1_input,
      MAXPOOL2_INPUT_WIDTH, MAXPOOL2_INPUT_HEIGHT, MAXPOOL2_INPUT_CHANNELS,
      2, 2, 2, 2);
  
  //print_buffer("%06.2f, ", maxpool2_input, 7, 7, 64);

  /* DENSE LAYER 1 */
  local float dense2_input[DENSE2_INPUT];
  dense_layer(
      dense1_weights, dense1_bias, dense1_input, dense2_input,
      DENSE1_INPUT_WIDTH, DENSE1_INPUT_HEIGHT, DENSE1_INPUT_CHANNELS,
      DENSE2_INPUT, true);
  
  /* DENSE LAYER 2 */
  local float softmax_input[SOFTMAX_INPUT];
  dense_layer(
      dense2_weights, dense2_bias, dense2_input, softmax_input,
      DENSE2_INPUT_WIDTH, DENSE2_INPUT_HEIGHT, DENSE2_INPUT_CHANNELS,
      SOFTMAX_INPUT, false);

  /* FINAL GUESS */
  float maximum = -INFINITY;
  int guess = -1;
  for (int i = 0; i < SOFTMAX_INPUT; i++) {
    float pix = softmax_input[i];
    if (maximum < pix) {
      maximum = pix;
      guess = i;
    }
  }

  guesses[get_global_id(0)] = guess;
}
