/*******************************************************************************
* Copyright (C) Maxim Integrated Products, Inc., All rights Reserved.
*
* This software is protected by copyright laws of the United States and
* of foreign countries. This material may also be protected by patent laws
* and technology transfer regulations of the United States and of foreign
* countries. This software is furnished under a license agreement and/or a
* nondisclosure agreement and may only be used or reproduced in accordance
* with the terms of those agreements. Dissemination of this information to
* any party or parties not specified in the license agreement and/or
* nondisclosure agreement is expressly prohibited.
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of Maxim Integrated
* Products, Inc. shall not be used except as stated in the Maxim Integrated
* Products, Inc. Branding Policy.
*
* The mere transfer of this software does not imply any licenses
* of trade secrets, proprietary technology, copyrights, patents,
* trademarks, maskwork rights, or any other form of intellectual
* property whatsoever. Maxim Integrated Products, Inc. retains all
* ownership rights.
*******************************************************************************/

// yolov1
// Created using ./ai8xize.py --verbose --log --overwrite --fifo --test-dir YOLO --prefix yolov1 --checkpoint-file /home/afshinniktash/ai/forkAfs-pt/ai8x-synthesis/YOLO/Yolov1_checkpoint-q.pth.tar --config-file YOLO/yolo-224-hwc-ai85.yaml --device MAX78000 --compact-data --mexpress --timer 0 --display-checkpoint

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"

volatile uint32_t cnn_time; // Stopwatch

//#define SERIAL_INPUT


#ifdef SERIAL_INPUT
#define CON_BAUD 115200*8   //UART baudrate used for sending data to PC, use max 921600 for serial stream
#define NUMBER_INPUT_WORDS 50176

int ConsoleUARTInit(uint32_t baud)
{

	mxc_uart_regs_t* ConsoleUart = MXC_UART_GET_UART(CONSOLE_UART);
	int err;

    NVIC_ClearPendingIRQ(MXC_UART_GET_IRQ(CONSOLE_UART));
    NVIC_DisableIRQ(MXC_UART_GET_IRQ(CONSOLE_UART));
	NVIC_SetPriority(MXC_UART_GET_IRQ(CONSOLE_UART), 1);
    NVIC_EnableIRQ(MXC_UART_GET_IRQ(CONSOLE_UART));

	if ((err = MXC_UART_Init(ConsoleUart, baud, MXC_UART_IBRO_CLK)) != E_NO_ERROR) {
		return err;
	}
	return 0;
}

uint8_t gencrc(const void* vptr, int len) {
	const uint8_t *data = vptr;
	unsigned crc = 0;
	int i, j;
	for (j = len; j; j--, data++) {
		crc ^= (*data << 8);
		for(i = 8; i; i--) {
			if (crc & 0x8000)
				crc ^= (0x1070 << 3);
			crc <<= 1;
		}
	}
	return (uint8_t)(crc >> 8);
}

void load_input_serial(void)
{
	uint8_t rxdata[4];
	uint32_t tmp;
	uint8_t crc, crc_result;
	uint32_t index = 0;
	LED_Off(LED2);

	printf("READY\n");

	for (int i = 0; i < NUMBER_INPUT_WORDS; i++) {

		index ++;

		tmp = 0;

		for (int j=0; j<4; j++)
		{
			rxdata[j] = MXC_UART_ReadCharacter(MXC_UART_GET_UART(CONSOLE_UART));

			tmp = tmp | (rxdata[j] << 8*(3-j));
		}

		//read crc
		crc = MXC_UART_ReadCharacter(MXC_UART_GET_UART(CONSOLE_UART));
		crc_result = gencrc(rxdata, 4);
		if ( crc != crc_result ) {
			printf("E %d",index);
			LED_On(LED2);
			while(1);
		}

		// load data to cnn
		while (((*((volatile uint32_t *) 0x50000004) & 1)) != 0); // Wait for FIFO 0
		*((volatile uint32_t *) 0x50000008) = tmp; // Write FIFO 0

		//printf("%x \n",tmp);
		LED_Toggle(LED1);

	}
}
#endif

void fail(void)
{
  printf("\n*** FAIL ***\n\n");
  while (1);
}

// Data input: HWC 3x224x224 (150528 bytes total / 50176 bytes per channel):
//static const uint32_t input_0[] = SAMPLE_INPUT_0;
void load_input(void)
{
  // This function loads the sample data input -- replace with actual data

  int i;
  //const uint32_t *in0 = input_0;
  uint32_t number = 0x121110;
  
  for (i = 0; i < 50176; i++) {
    while (((*((volatile uint32_t *) 0x50000004) & 1)) != 0); // Wait for FIFO 0
    *((volatile uint32_t *) 0x50000008) = number; // Write FIFO 0
	
	 //*((volatile uint32_t *) 0x50000008) = *in0++; // Write FIFO 0
  }
}

// Expected output of layer 23 for yolov1 given the sample input (known-answer test)
// Delete this function for production code
int check_output(void)
{
  int i;
  uint32_t mask, len;
  volatile uint32_t *addr;
  const uint32_t sample_output[] = SAMPLE_OUTPUT;
  const uint32_t *ptr = sample_output;

  while ((addr = (volatile uint32_t *) *ptr++) != 0) {
    mask = *ptr++;
    len = *ptr++;
    for (i = 0; i < len; i++)
      if ((*addr++ & mask) != *ptr++) return CNN_FAIL;
  }

  return CNN_OK;
}

static int32_t ml_data[CNN_NUM_OUTPUTS];

int main(void)
{
  MXC_ICC_Enable(MXC_ICC0); // Enable cache

  // Switch to 100 MHz clock
  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
  SystemCoreClockUpdate();

#ifdef SERIAL_INPUT
  // initialize UART
  ConsoleUARTInit(CON_BAUD);
#endif
  printf("Waiting...\n");


  // DO NOT DELETE THIS LINE:
  MXC_Delay(SEC(2)); // Let debugger interrupt if needed

  // Enable peripheral, enable CNN interrupt, turn on CNN clock
  // CNN clock: 50 MHz div 1
  cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);

  printf("\n*** CNN Inference Test ***\n");

  cnn_init(); // Bring state machine into consistent state
  cnn_load_weights(); // Load kernels
  cnn_load_bias();
  cnn_configure(); // Configure state machine
  cnn_start(); // Start CNN processing
#ifndef SERIAL_INPUT
  load_input(); // Load data input via FIFO
#else
  load_input_serial(); // Load data input via FIFO
#endif

  SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk; // SLEEPDEEP=0
  while (cnn_time == 0)
    __WFI(); // Wait for CNN

  if (check_output() != CNN_OK) fail();
  cnn_unload((uint32_t *) ml_data);

  printf("\n*** PASS ***\n\n");

  printf("dump CNN result: \n");
  
  for (int j=0; j<CNN_NUM_OUTPUTS; j++)
	printf("%d- %08X\n", j, ml_data[j]);
#ifdef CNN_INFERENCE_TIMER
  printf("Approximate data loading and inference time: %u us\n\n", cnn_time);
#endif

  cnn_disable(); // Shut down CNN clock, disable peripheral


  return 0;
}

/*
  SUMMARY OF OPS
  Hardware: 428,989,904 ops (420,950,768 macc; 8,039,136 comp; 0 add; 0 mul; 0 bitwise)
    Layer 0: 89,915,392 ops (86,704,128 macc; 3,211,264 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 176,920,576 ops (173,408,256 macc; 3,512,320 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 11,189,248 ops (10,838,016 macc; 351,232 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 14,551,040 ops (14,450,688 macc; 100,352 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 3,311,616 ops (3,211,264 macc; 100,352 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 58,003,456 ops (57,802,752 macc; 200,704 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6: 1,831,424 ops (1,605,632 macc; 225,792 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7: 14,500,864 ops (14,450,688 macc; 50,176 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8: 1,630,720 ops (1,605,632 macc; 25,088 comp; 0 add; 0 mul; 0 bitwise)
    Layer 9: 14,500,864 ops (14,450,688 macc; 50,176 comp; 0 add; 0 mul; 0 bitwise)
    Layer 10: 1,630,720 ops (1,605,632 macc; 25,088 comp; 0 add; 0 mul; 0 bitwise)
    Layer 11: 14,500,864 ops (14,450,688 macc; 50,176 comp; 0 add; 0 mul; 0 bitwise)
    Layer 12: 457,856 ops (401,408 macc; 56,448 comp; 0 add; 0 mul; 0 bitwise)
    Layer 13: 3,625,216 ops (3,612,672 macc; 12,544 comp; 0 add; 0 mul; 0 bitwise)
    Layer 14: 407,680 ops (401,408 macc; 6,272 comp; 0 add; 0 mul; 0 bitwise)
    Layer 15: 3,625,216 ops (3,612,672 macc; 12,544 comp; 0 add; 0 mul; 0 bitwise)
    Layer 16: 7,237,888 ops (7,225,344 macc; 12,544 comp; 0 add; 0 mul; 0 bitwise)
    Layer 17: 7,237,888 ops (7,225,344 macc; 12,544 comp; 0 add; 0 mul; 0 bitwise)
    Layer 18: 1,822,016 ops (1,806,336 macc; 15,680 comp; 0 add; 0 mul; 0 bitwise)
    Layer 19: 1,809,472 ops (1,806,336 macc; 3,136 comp; 0 add; 0 mul; 0 bitwise)
    Layer 20: 203,840 ops (200,704 macc; 3,136 comp; 0 add; 0 mul; 0 bitwise)
    Layer 21: 50,960 ops (50,176 macc; 784 comp; 0 add; 0 mul; 0 bitwise)
    Layer 22: 13,328 ops (12,544 macc; 784 comp; 0 add; 0 mul; 0 bitwise)
    Layer 23: 11,760 ops (11,760 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 298,544 bytes out of 442,368 bytes total (67%)
  Bias memory:   975 bytes out of 2,048 bytes total (48%)
*/

