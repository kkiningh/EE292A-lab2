// TODO: Define any constants you'll need
// image is a 28x28xN array (N images) of bytes (each pixel is 8 bit grayscale)
#define IMG_SIZE 28 // Image height and width
#define INPUT_CHANNELS 1 // Image channels
// TODO: If you decide you'd like to write helper functions, you can define them here
	

// TODO: Build a CNN!
__attribute__((reqd_work_group_size(10000,1,1))) // change this to change workgroup size
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
  // The input image
  global const unsigned char *image = &images[get_global_id(0) * FEATURE_COUNTS];
  int img_width = IMG_SIZE;
  int img_height = IMG_SIZE;
  int img_channels = INPUT_CHANNELS;

  int filter_width = 5;
  int filter_height = 5;
  int num_filters = 32;
  
  // TODO redefine image and output
  // TODO add padding code

  /* CONV LAYER 1 */
  for (int w = 0; w < img_width; w++) {
    for (int h = 0; h < img_height; h++) {
      for (int c = 0; c < img_channels; c++) {
        float sum = 0;
        // TODO
    	for (int f = 0; f < num_filters; f++) {
          for (int ww = 0; ww < filter_width; ww++) {
            for (int hh = 0; hh < filter_width; hh++) {
              sum += image[c * img_width * img_height + w * img_width + h + ww
                * filter_width + hh] * conv1_weights[f * c * filter_width 
		  		* filter_height + c * filter_height * filter_width + ww 
				* filter_width + hh];
            }
          }
		  sum += conv1_bias[f * c * filter_width * filter_height + c
			  * filter_height * filter_width];
        }
        output[img_width * img_height * f + img_width * w + h] = sum;
      }
    }  
  }

  // TODO redefine image and output

  int filter_width = 2;
  int filter_height = 2;
  int num_filters = 32;

  /* MAXPOOL LAYER 1 */
  for (int w = 0; w < img_width; w+=2) {
    for (int h = 0; h < img_height; h+=2) {
      for (int c = 0; c < num_filters; c++) {
        float maxima = image[c * img_width * img_height + w * img_width + h];
        // TODO
        for (int ww = 0; ww < filter_width; ww++) {
          for (int hh = 0; hh < filter_width; hh++) {
            if(maxima < image[c * img_width * img_height + w * img_width + h + ww
              * filter_width + hh]){
		      maxima = image[c * img_width * img_height + w * img_width + h + ww 
			      * filter_width + hh];
			}
          }
        }
        output[img_width * img_height * c + img_width * w + h] = maxima;
      }
    }  
  }
  
  // TODO redefine image and output
  // TODO add padding code

  int img_width = IMG_SIZE/2;
  int img_height = IMG_SIZE/2;
  int img_channels = num_filters;

  int filter_width = 5;
  int filter_height = 5;
  int num_filters = 64;

  /* CONV LAYER 2 */
  for (int w = 0; w < img_width; w++) {
    for (int h = 0; h < img_height; h++) {
      for (int c = 0; c < img_channels; c++) {
        float sum = 0;
        // TODO
    	for (int f = 0; f < num_filters; f++) {
          for (int ww = 0; ww < filter_width; ww++) {
            for (int hh = 0; hh < filter_width; hh++) {
              sum += image[c * img_width * img_height + w * img_width + h + ww
                * filter_width + hh] * conv2_weights[f * c * filter_width 
		  		* filter_height + c * filter_height * filter_width + ww 
				* filter_width + hh];
            }
          }
		  sum += conv2_bias[f * c * filter_width * filter_height + c
			  * filter_height * filter_width];
        }
        output[img_width * img_height * f + img_width * w + h] = sum;
      }
    }  
  }

  // TODO redefine image and output
  int filter_width = 2;
  int filter_height = 2;
  int num_filters = 64;

  /* MAXPOOL LAYER 2 */
  for (int w = 0; w < img_width; w+=2) {
    for (int h = 0; h < img_height; h+=2) {
      for (int c = 0; c < num_filters; c++) {
        float maxima = image[c * img_width * img_height + w * img_width + h];
        // TODO
        for (int ww = 0; ww < filter_width; ww++) {
          for (int hh = 0; hh < filter_width; hh++) {
            if(maxima < image[c * img_width * img_height + w * img_width + h + ww
              * filter_width + hh]){
		      maxima = image[c * img_width * img_height + w * img_width + h + ww 
			      * filter_width + hh];
			}
          }
        }
        output[img_width * img_height * c + img_width * w + h] = maxima;
      }
    }  
  }
  
  // TODO redefine image and output
  int img_width = IMG_SIZE/4;
  int img_height = IMG_SIZE/4;
  int img_channels = num_filters;

  int num_outputs = 256;

  /* DENSE LAYER */
  for (int x = 0; x < num_outputs; x++) {
    float sum = 0;
    for (int w = 0; w < img_width; w++) {
      for (int h = 0; h < img_height; h++) {
        for (int c = 0; c < num_filters; c++) {
          // TODO
		  sum += image[c * img_width * img_height + w * img_width + h]
			  * dense1_wieghts[x * c * img_width * img_height
			  + c * img_width * img_height + w * img_width + h];
		} 
      }
    }  
    sum += dense1_bias[x];
	output[x] = sum;
  }

  int num_inputs = 256;
  int num_outputs = 10;
  int maxima = 0;

  /* DENSE 2 */
  for (int x = 0; x < num_outputs; x++) {
    float sum = 0;
    for (int y = 0; y < num_inputs; y++) {
      // TODO
	  sum += image[y] * dense2_wieghts[y];
    }  
    sum += dense1_bias[x];
	output[x] = sum;
	if(x == 0){
      maxima = output[x];
    }
	if(output[x] > maxima){
	  maxima = output[x];
	}
  }

  /* FINAL GUESS */
  guesses[get_global_id(0)] = 0;
}
