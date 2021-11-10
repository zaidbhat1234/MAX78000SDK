#include "utils_faceid.h"
#include "cnn.h"

// #define STB_IMAGE_IMPLEMENTATION
// #include "stb_image/stb_image.h"
// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "stb_image/stb_image_write.h"

uint8_t rxBuffer[DATA_SIZE_IN];


int uart_write(uint8_t* data, unsigned int len)
{
  unsigned int bytes_tx_total = 0;
  unsigned int bytes_tx;

  while(bytes_tx_total < len){
    bytes_tx = MXC_UART_WriteTXFIFO(MXC_UARTn, data + bytes_tx_total, len - bytes_tx_total);
    bytes_tx_total += bytes_tx;
  }

  return 1;
}

int uart_read(uint8_t* buffer, unsigned int len)
{
  unsigned int bytes_rx_total = 0;
  unsigned int bytes_rx;

  while (bytes_rx_total < len){
    bytes_rx = MXC_UART_ReadRXFIFO(MXC_UARTn, buffer + bytes_rx_total, len - bytes_rx_total);
    bytes_rx_total += bytes_rx;
  }

  return 1;
}

int wait_for_feedback(){
  volatile uint8_t read_byte = 0;

  while (1){
    if (MXC_UART_GetRXFIFOAvailable(MXC_UARTn) > 0) {
      read_byte = MXC_UART_ReadCharacter(MXC_UARTn);
      if (read_byte == 100 )
        break;
      else if (read_byte == 200 )
        return 0;
    }          
  }

  return 1;
}

void load_input(int8_t mode)
{
 uint32_t i;
 i = 0;
  const uint32_t *in0 = input_0;
 uint32_t number;
 while (i < IMG_SIZE * IMG_SIZE) {
   while (((*((volatile uint32_t *) 0x50000004) & 1)) != 0); // Wait for FIFO 0
   number = ((uint32_t)rxBuffer[i]<<16) | ((uint32_t)rxBuffer[i+1]<<8) | ((uint32_t)rxBuffer[i+2]);

   if (mode > 0){
       uart_write((uint8_t*) &number, 4);
       uart_write((uint8_t*) (in0++), 4);
   }

   *((volatile uint32_t *) 0x50000008) = *in0++; // Write FIFO 0

   i += 1;
 }

  // 3-channel 56x56 data input (9408 bytes total / 3136 bytes per channel):
  // HWC 56x56, channels 0 to 2
  memcpy32((uint32_t *) 0x50400000, in0, IMG_SIZE*IMG_SIZE);
  //  int i;
  //  for (i = 0; i < 50176; i++) {
  //      while (((*((volatile uint32_t *) 0x50000004) & 1)) != 0); // Wait for FIFO 0
  //      *((volatile uint32_t *) 0x50000008) = *in0++; // Write FIFO 0
  //  }

}

// void inline_softmax_q17p14_q15(q31_t * vec_in, const uint16_t start, const uint16_t end)
// {
//     q31_t     sum;
//     int16_t   i;
//     uint8_t   shift;
//     q31_t     base;
//     base = -1 * 0x80000000;

//     for (i = start; i < end; i++)
//     {
//         if (vec_in[i] > base)
//         {
//             base = vec_in[i];
//         }
//     }

//     /* we ignore really small values
//      * anyway, they will be 0 after shrinking
//      * to q15_t
//      */

//     base = base - (16<<14);

//     sum = 0;

//     for (i = start; i < end; i++)
//     {
//         if (vec_in[i] > base)
//         {
//             shift = (uint8_t)((8192 + vec_in[i] - base) >> 14);
//             sum += (0x1 << shift);
//         }
//     }


//     /* This is effectively (0x1 << 32) / sum */
//     int64_t div_base = 0x100000000LL;
//     int32_t output_base = (int32_t)(div_base / sum);
//     int32_t out;

//     /* Final confidence will be output_base >> ( 17 - (vec_in[i] - base)>>14 )
//      * so 32768 (0x1<<15) -> 100% confidence when sum = 0x1 << 16, output_base = 0x1 << 16
//      * and vec_in[i]-base = 16
//      */

//     for (i = start; i < end; i++)
//     {
//         if (vec_in[i] > base)
//         {
//             /* Here minimum value of 17+base-vec[i] will be 1 */
//             shift = (uint8_t)(17+((8191 + base - vec_in[i]) >> 14));

//             out = (output_base >> shift);

//             if (out > 32767)
//                 out = 32767;

//             vec_in[i] = out;


//         } else
//         {
//             vec_in[i] = 0;
//         }
//     }

// }

// uint16_t argmax_softmax(q31_t * vec_in, const uint16_t start)
// {
//     q31_t cls_score = 0;
//     uint16_t idx = 0;
//     uint16_t i;
//     for (i = start; i < start + NUM_CLASSES; ++i) {
//         if (vec_in[i] > cls_score) {
//             idx = i;
//             cls_score = vec_in[i];
//         }
//         vec_in[i] = sigmoid(vec_in[i]);
//     }
//     inline_softmax_q17p14_q15(vec_in, start, start + NUM_CLASSES);
//     return idx;
// }

// void NMS_max(q31_t * vec_in, const uint16_t dim_vec, q31_t* max_box)
// {
//     // x1, y1, x2, y2, box_score, cls_score, cls
//     // max_box[7] = {0};
//     q31_t confident_threshold = 9011;  // 0.55

//     uint16_t cls_idx, max_i = 0;
//     uint16_t i, b;
//     uint16_t m, n;

//     q31_t gridX, gridY;
//     q31_t centerX, centerY, width, height;
//     q31_t tmp;
//     max_box[4] = confident_threshold;

//     for (i = 0; i < dim_vec; i += NUM_CHANNELS) {
//         for (b = 0; b < NUM_BOXES; ++b) {
//             tmp = sigmoid(vec_in[i + 5 * b + 4]);
//             if (tmp > max_box[4])
//             {
//                 max_i = i;
//                 max_box[0] = vec_in[i + 5 * b];
//                 max_box[1] = vec_in[i + 5 * b + 1];
//                 max_box[2] = vec_in[i + 5 * b + 2];
//                 max_box[3] = vec_in[i + 5 * b + 3];
//                 max_box[4] = tmp;
//             }
//         }
//     }
//     for (int j = 0; j < NUM_CLASSES; ++j) {
//         max_box[5 + j] = vec_in[max_i + BOX_DIMENSION + j];
//     }
//     cls_idx = argmax_softmax(max_box, 5);
//     max_box[5] = max_box[cls_idx];
//     max_box[6] = cls_idx - 5;

//     m = max_i / (NUM_GRIDS * NUM_CHANNELS);
//     n = max_i / NUM_CHANNELS % NUM_GRIDS;

//     gridX = GRID_SIZE * m;
//     gridY = GRID_SIZE * n;

//     centerX = gridX + q_mul(sigmoid(max_box[0]), GRID_SIZE);
//     centerY = gridY + q_mul(sigmoid(max_box[1]), GRID_SIZE);
//     width = q_mul(sigmoid(max_box[2]), IMG_SIZE);
//     height = q_mul(sigmoid(max_box[3]), IMG_SIZE);

//     max_box[0] = max(0, (centerX - (width >> 1)));
//     max_box[1] = max(0, (centerY - (height >> 1)));
//     max_box[2] = min(IMG_SIZE - 1, (centerX + (width >> 1)));
//     max_box[3] = min(IMG_SIZE - 1, (centerY + (height >> 1)));
// }