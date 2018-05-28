// TODO: Define any constants you'll need
// image is a 28x28xN array (N images) of bytes (each pixel is 8 bit grayscale)

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

  /* CONV LAYER 1 */
  for (int w = 0; w < img_width; w++) {
    for (int h = 0; h < img_height; h++) {
        for (int c = 0; c < img_channels; c++) {
        float sum = 0;
        // TODO
        for (int ww = 0; ww < filter_width; ww++) {
          for (int hh = 0; hh < filter_width; hh++) {
            for (int f = 0; f < num_filters; f++) {
              sum += image[c * img_width * img_height + w * img_width + h + ww
                * filter_width + hh] * filter_weight[];
            }
          }
        }

        output[filter_size * f + img_width * w + h] = sum;
      }
    }  
  }

  /* MAXPOOL LAYER 1 */

  /* CONV LAYER 2 */

  /* MAXPOOL LAYER 2 */

  /* DENSE LAYER */

  /* DENSE 2 */
  
  /* FINAL GUESS */
  guesses[get_global_id(0)] = 0;
}
