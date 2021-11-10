/*******************************************************************************
* Copyright (C) Maxim Integrated Products, Inc., All Rights Reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
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
*
******************************************************************************/

/**
 * @file    main.c
 * @brief   FaceID EvKit Demo
 *
 * @details
 *
 */

#define S_MODULE_NAME	"main"

/***** Includes *****/
#include <stdio.h>
#include <stdint.h>
#include "board.h"
#include "mxc.h"
#include "mxc_delay.h"
#include "camera.h"
#include "state.h"
#include "icc.h"
#include "rtc.h"
#include "cnn.h"
#ifdef BOARD_FTHR_REVA
#include "tft_fthr.h"
#endif
#ifdef BOARD_EVKIT_V1
#include "tft.h"
#include "bitmap.h"
#endif
#include "MAXCAM_Debug.h"
#include "faceID.h"
#include "utils_faceid.h"
#include "utils.h"
#include "weights.h"
#include "embedding_process.h"
#include "keypad.h"
#define CAMERA_FREQ (10 * 1000 * 1000)

void fail(void)
{
  printf("\n*** FAIL ***\n\n");
  while (1);
}

void cnn_wait(void)
{
//  while ((*((volatile uint32_t *) 0x50100000) & (1<<12)) != 1<<12) ;
  while (cnn_time == 0)
    __WFI(); // Wait for CNN
//  CNN_COMPLETE; // Signal that processing is complete
//  cnn_time = MXC_TMR_SW_Stop(MXC_TMR0);
}

int cnn_load(int8_t mode)
{
  cnn_init(); // Bring state machine into consistent state

  cnn_load_weights(); // Load kernels
  cnn_load_bias();

  cnn_configure(); // Configure state machine

  load_input(mode); // Load data input
  cnn_start(); // Start CNN processing

  return 1;
}

// Expected output of layer 7 for yolov1 given the sample input
int cnn_check(void)
{
    if ((*((volatile uint32_t *) 0x50400000)) != 0x00000000) return CNN_FAIL; // 0,0,0
    if ((*((volatile uint32_t *) 0x50400004)) != 0x00000000) return CNN_FAIL; // 0,0,1
    if ((*((volatile uint32_t *) 0x50400008)) != 0x00000000) return CNN_FAIL; // 0,0,2
    if ((*((volatile uint32_t *) 0x5040000c)) != 0x00000000) return CNN_FAIL; // 0,0,3
    if ((*((volatile uint32_t *) 0x50408000)) != 0x00000000) return CNN_FAIL; // 0,0,4
    if ((*((volatile uint32_t *) 0x50408004)) != 0x00000000) return CNN_FAIL; // 0,0,5
    if ((*((volatile uint32_t *) 0x50408008)) != 0x00000000) return CNN_FAIL; // 0,0,6
    if ((*((volatile uint32_t *) 0x5040800c)) != 0x00000000) return CNN_FAIL; // 0,0,7
    if ((*((volatile uint32_t *) 0x50410000)) != 0x00000000) return CNN_FAIL; // 0,0,8
    if ((*((volatile uint32_t *) 0x50410004)) != 0x00000000) return CNN_FAIL; // 0,0,9
    if ((*((volatile uint32_t *) 0x50410008)) != 0x00000000) return CNN_FAIL; // 0,0,10
    if ((*((volatile uint32_t *) 0x5041000c)) != 0x00000000) return CNN_FAIL; // 0,0,11
    if ((*((volatile uint32_t *) 0x50418000)) != 0x00000000) return CNN_FAIL; // 0,0,12
    if ((*((volatile uint32_t *) 0x50418004)) != 0x00000000) return CNN_FAIL; // 0,0,13
    if ((*((volatile uint32_t *) 0x50418008)) != 0x00000000) return CNN_FAIL; // 0,0,14
    if ((*((volatile uint32_t *) 0x5041800c)) != 0x00000000) return CNN_FAIL; // 0,0,15
    if ((*((volatile uint32_t *) 0x50800000)) != 0x00000000) return CNN_FAIL; // 0,0,16
    if ((*((volatile uint32_t *) 0x50800004)) != 0x00000000) return CNN_FAIL; // 0,0,17
    if ((*((volatile uint32_t *) 0x50800008)) != 0x00000000) return CNN_FAIL; // 0,0,18
    if ((*((volatile uint32_t *) 0x5080000c)) != 0x00000000) return CNN_FAIL; // 0,0,19
    if ((*((volatile uint32_t *) 0x50808000)) != 0x00000000) return CNN_FAIL; // 0,0,20
    if ((*((volatile uint32_t *) 0x50808004)) != 0x00000000) return CNN_FAIL; // 0,0,21
    if ((*((volatile uint32_t *) 0x50808008)) != 0x00000000) return CNN_FAIL; // 0,0,22
    if ((*((volatile uint32_t *) 0x5080800c)) != 0x00000000) return CNN_FAIL; // 0,0,23
    if ((*((volatile uint32_t *) 0x50810000)) != 0x00000000) return CNN_FAIL; // 0,0,24
    if ((*((volatile uint32_t *) 0x50810004)) != 0x00000000) return CNN_FAIL; // 0,0,25
    if ((*((volatile uint32_t *) 0x50810008)) != 0x00000000) return CNN_FAIL; // 0,0,26
    if ((*((volatile uint32_t *) 0x5081000c)) != 0x00000000) return CNN_FAIL; // 0,0,27
    if ((*((volatile uint32_t *) 0x50818000)) != 0x00000000) return CNN_FAIL; // 0,0,28
    if ((*((volatile uint32_t *) 0x50818004)) != 0x00000000) return CNN_FAIL; // 0,0,29
    if ((*((volatile uint32_t *) 0x50400010)) != 0x00000000) return CNN_FAIL; // 0,1,0
    if ((*((volatile uint32_t *) 0x50400014)) != 0x00000000) return CNN_FAIL; // 0,1,1
    if ((*((volatile uint32_t *) 0x50400018)) != 0x00000000) return CNN_FAIL; // 0,1,2
    if ((*((volatile uint32_t *) 0x5040001c)) != 0x00000000) return CNN_FAIL; // 0,1,3
    if ((*((volatile uint32_t *) 0x50408010)) != 0x00000000) return CNN_FAIL; // 0,1,4
    if ((*((volatile uint32_t *) 0x50408014)) != 0x00000000) return CNN_FAIL; // 0,1,5
    if ((*((volatile uint32_t *) 0x50408018)) != 0x00000000) return CNN_FAIL; // 0,1,6
    if ((*((volatile uint32_t *) 0x5040801c)) != 0x00000000) return CNN_FAIL; // 0,1,7
    if ((*((volatile uint32_t *) 0x50410010)) != 0x00000000) return CNN_FAIL; // 0,1,8
    if ((*((volatile uint32_t *) 0x50410014)) != 0x00000000) return CNN_FAIL; // 0,1,9
    if ((*((volatile uint32_t *) 0x50410018)) != 0x00000000) return CNN_FAIL; // 0,1,10
    if ((*((volatile uint32_t *) 0x5041001c)) != 0x00000000) return CNN_FAIL; // 0,1,11
    if ((*((volatile uint32_t *) 0x50418010)) != 0x00000000) return CNN_FAIL; // 0,1,12
    if ((*((volatile uint32_t *) 0x50418014)) != 0x00000000) return CNN_FAIL; // 0,1,13
    if ((*((volatile uint32_t *) 0x50418018)) != 0x00000000) return CNN_FAIL; // 0,1,14
    if ((*((volatile uint32_t *) 0x5041801c)) != 0x00000000) return CNN_FAIL; // 0,1,15
    if ((*((volatile uint32_t *) 0x50800010)) != 0x00000000) return CNN_FAIL; // 0,1,16
    if ((*((volatile uint32_t *) 0x50800014)) != 0x00000000) return CNN_FAIL; // 0,1,17
    if ((*((volatile uint32_t *) 0x50800018)) != 0x00000000) return CNN_FAIL; // 0,1,18
    if ((*((volatile uint32_t *) 0x5080001c)) != 0x00000000) return CNN_FAIL; // 0,1,19
    if ((*((volatile uint32_t *) 0x50808010)) != 0x00000000) return CNN_FAIL; // 0,1,20
    if ((*((volatile uint32_t *) 0x50808014)) != 0x00000000) return CNN_FAIL; // 0,1,21
    if ((*((volatile uint32_t *) 0x50808018)) != 0x00000000) return CNN_FAIL; // 0,1,22
    if ((*((volatile uint32_t *) 0x5080801c)) != 0x00000000) return CNN_FAIL; // 0,1,23
    if ((*((volatile uint32_t *) 0x50810010)) != 0x00000000) return CNN_FAIL; // 0,1,24
    if ((*((volatile uint32_t *) 0x50810014)) != 0x00000000) return CNN_FAIL; // 0,1,25
    if ((*((volatile uint32_t *) 0x50810018)) != 0x00000000) return CNN_FAIL; // 0,1,26
    if ((*((volatile uint32_t *) 0x5081001c)) != 0x00000000) return CNN_FAIL; // 0,1,27
    if ((*((volatile uint32_t *) 0x50818010)) != 0x00000000) return CNN_FAIL; // 0,1,28
    if ((*((volatile uint32_t *) 0x50818014)) != 0x00000000) return CNN_FAIL; // 0,1,29
    if ((*((volatile uint32_t *) 0x50400020)) != 0x00000000) return CNN_FAIL; // 0,2,0
    if ((*((volatile uint32_t *) 0x50400024)) != 0x00000000) return CNN_FAIL; // 0,2,1
    if ((*((volatile uint32_t *) 0x50400028)) != 0x00000000) return CNN_FAIL; // 0,2,2
    if ((*((volatile uint32_t *) 0x5040002c)) != 0x00000000) return CNN_FAIL; // 0,2,3
    if ((*((volatile uint32_t *) 0x50408020)) != 0x00000000) return CNN_FAIL; // 0,2,4
    if ((*((volatile uint32_t *) 0x50408024)) != 0x00000000) return CNN_FAIL; // 0,2,5
    if ((*((volatile uint32_t *) 0x50408028)) != 0x00000000) return CNN_FAIL; // 0,2,6
    if ((*((volatile uint32_t *) 0x5040802c)) != 0x00000000) return CNN_FAIL; // 0,2,7
    if ((*((volatile uint32_t *) 0x50410020)) != 0x00000000) return CNN_FAIL; // 0,2,8
    if ((*((volatile uint32_t *) 0x50410024)) != 0x00000000) return CNN_FAIL; // 0,2,9
    if ((*((volatile uint32_t *) 0x50410028)) != 0x00000000) return CNN_FAIL; // 0,2,10
    if ((*((volatile uint32_t *) 0x5041002c)) != 0x00000000) return CNN_FAIL; // 0,2,11
    if ((*((volatile uint32_t *) 0x50418020)) != 0x00000000) return CNN_FAIL; // 0,2,12
    if ((*((volatile uint32_t *) 0x50418024)) != 0x00000000) return CNN_FAIL; // 0,2,13
    if ((*((volatile uint32_t *) 0x50418028)) != 0x00000000) return CNN_FAIL; // 0,2,14
    if ((*((volatile uint32_t *) 0x5041802c)) != 0x00000000) return CNN_FAIL; // 0,2,15
    if ((*((volatile uint32_t *) 0x50800020)) != 0x00000000) return CNN_FAIL; // 0,2,16
    if ((*((volatile uint32_t *) 0x50800024)) != 0x00000000) return CNN_FAIL; // 0,2,17
    if ((*((volatile uint32_t *) 0x50800028)) != 0x00000000) return CNN_FAIL; // 0,2,18
    if ((*((volatile uint32_t *) 0x5080002c)) != 0x00000000) return CNN_FAIL; // 0,2,19
    if ((*((volatile uint32_t *) 0x50808020)) != 0x00000000) return CNN_FAIL; // 0,2,20
    if ((*((volatile uint32_t *) 0x50808024)) != 0x00000000) return CNN_FAIL; // 0,2,21
    if ((*((volatile uint32_t *) 0x50808028)) != 0x00000000) return CNN_FAIL; // 0,2,22
    if ((*((volatile uint32_t *) 0x5080802c)) != 0x00000000) return CNN_FAIL; // 0,2,23
    if ((*((volatile uint32_t *) 0x50810020)) != 0x00000000) return CNN_FAIL; // 0,2,24
    if ((*((volatile uint32_t *) 0x50810024)) != 0x00000000) return CNN_FAIL; // 0,2,25
    if ((*((volatile uint32_t *) 0x50810028)) != 0x00000000) return CNN_FAIL; // 0,2,26
    if ((*((volatile uint32_t *) 0x5081002c)) != 0x00000000) return CNN_FAIL; // 0,2,27
    if ((*((volatile uint32_t *) 0x50818020)) != 0x00000000) return CNN_FAIL; // 0,2,28
    if ((*((volatile uint32_t *) 0x50818024)) != 0x00000000) return CNN_FAIL; // 0,2,29
    if ((*((volatile uint32_t *) 0x50400030)) != 0x00000000) return CNN_FAIL; // 0,3,0
    if ((*((volatile uint32_t *) 0x50400034)) != 0x00000000) return CNN_FAIL; // 0,3,1
    if ((*((volatile uint32_t *) 0x50400038)) != 0x00000000) return CNN_FAIL; // 0,3,2
    if ((*((volatile uint32_t *) 0x5040003c)) != 0x00000000) return CNN_FAIL; // 0,3,3
    if ((*((volatile uint32_t *) 0x50408030)) != 0x00000000) return CNN_FAIL; // 0,3,4
    if ((*((volatile uint32_t *) 0x50408034)) != 0x00000000) return CNN_FAIL; // 0,3,5
    if ((*((volatile uint32_t *) 0x50408038)) != 0x00000000) return CNN_FAIL; // 0,3,6
    if ((*((volatile uint32_t *) 0x5040803c)) != 0x00000000) return CNN_FAIL; // 0,3,7
    if ((*((volatile uint32_t *) 0x50410030)) != 0x00000000) return CNN_FAIL; // 0,3,8
    if ((*((volatile uint32_t *) 0x50410034)) != 0x00000000) return CNN_FAIL; // 0,3,9
    if ((*((volatile uint32_t *) 0x50410038)) != 0x00000000) return CNN_FAIL; // 0,3,10
    if ((*((volatile uint32_t *) 0x5041003c)) != 0x00000000) return CNN_FAIL; // 0,3,11
    if ((*((volatile uint32_t *) 0x50418030)) != 0x00000000) return CNN_FAIL; // 0,3,12
    if ((*((volatile uint32_t *) 0x50418034)) != 0x00000000) return CNN_FAIL; // 0,3,13
    if ((*((volatile uint32_t *) 0x50418038)) != 0x00000000) return CNN_FAIL; // 0,3,14
    if ((*((volatile uint32_t *) 0x5041803c)) != 0x00000000) return CNN_FAIL; // 0,3,15
    if ((*((volatile uint32_t *) 0x50800030)) != 0x00000000) return CNN_FAIL; // 0,3,16
    if ((*((volatile uint32_t *) 0x50800034)) != 0x00000000) return CNN_FAIL; // 0,3,17
    if ((*((volatile uint32_t *) 0x50800038)) != 0x00000000) return CNN_FAIL; // 0,3,18
    if ((*((volatile uint32_t *) 0x5080003c)) != 0x00000000) return CNN_FAIL; // 0,3,19
    if ((*((volatile uint32_t *) 0x50808030)) != 0x00000000) return CNN_FAIL; // 0,3,20
    if ((*((volatile uint32_t *) 0x50808034)) != 0x00000000) return CNN_FAIL; // 0,3,21
    if ((*((volatile uint32_t *) 0x50808038)) != 0x00000000) return CNN_FAIL; // 0,3,22
    if ((*((volatile uint32_t *) 0x5080803c)) != 0x00000000) return CNN_FAIL; // 0,3,23
    if ((*((volatile uint32_t *) 0x50810030)) != 0x00000000) return CNN_FAIL; // 0,3,24
    if ((*((volatile uint32_t *) 0x50810034)) != 0x00000000) return CNN_FAIL; // 0,3,25
    if ((*((volatile uint32_t *) 0x50810038)) != 0x00000000) return CNN_FAIL; // 0,3,26
    if ((*((volatile uint32_t *) 0x5081003c)) != 0x00000000) return CNN_FAIL; // 0,3,27
    if ((*((volatile uint32_t *) 0x50818030)) != 0x00000000) return CNN_FAIL; // 0,3,28
    if ((*((volatile uint32_t *) 0x50818034)) != 0x00000000) return CNN_FAIL; // 0,3,29
    if ((*((volatile uint32_t *) 0x50400040)) != 0x00000000) return CNN_FAIL; // 0,4,0
    if ((*((volatile uint32_t *) 0x50400044)) != 0x00000000) return CNN_FAIL; // 0,4,1
    if ((*((volatile uint32_t *) 0x50400048)) != 0x00000000) return CNN_FAIL; // 0,4,2
    if ((*((volatile uint32_t *) 0x5040004c)) != 0x00000000) return CNN_FAIL; // 0,4,3
    if ((*((volatile uint32_t *) 0x50408040)) != 0x00000000) return CNN_FAIL; // 0,4,4
    if ((*((volatile uint32_t *) 0x50408044)) != 0x00000000) return CNN_FAIL; // 0,4,5
    if ((*((volatile uint32_t *) 0x50408048)) != 0x00000000) return CNN_FAIL; // 0,4,6
    if ((*((volatile uint32_t *) 0x5040804c)) != 0x00000000) return CNN_FAIL; // 0,4,7
    if ((*((volatile uint32_t *) 0x50410040)) != 0x00000000) return CNN_FAIL; // 0,4,8
    if ((*((volatile uint32_t *) 0x50410044)) != 0x00000000) return CNN_FAIL; // 0,4,9
    if ((*((volatile uint32_t *) 0x50410048)) != 0x00000000) return CNN_FAIL; // 0,4,10
    if ((*((volatile uint32_t *) 0x5041004c)) != 0x00000000) return CNN_FAIL; // 0,4,11
    if ((*((volatile uint32_t *) 0x50418040)) != 0x00000000) return CNN_FAIL; // 0,4,12
    if ((*((volatile uint32_t *) 0x50418044)) != 0x00000000) return CNN_FAIL; // 0,4,13
    if ((*((volatile uint32_t *) 0x50418048)) != 0x00000000) return CNN_FAIL; // 0,4,14
    if ((*((volatile uint32_t *) 0x5041804c)) != 0x00000000) return CNN_FAIL; // 0,4,15
    if ((*((volatile uint32_t *) 0x50800040)) != 0x00000000) return CNN_FAIL; // 0,4,16
    if ((*((volatile uint32_t *) 0x50800044)) != 0x00000000) return CNN_FAIL; // 0,4,17
    if ((*((volatile uint32_t *) 0x50800048)) != 0x00000000) return CNN_FAIL; // 0,4,18
    if ((*((volatile uint32_t *) 0x5080004c)) != 0x00000000) return CNN_FAIL; // 0,4,19
    if ((*((volatile uint32_t *) 0x50808040)) != 0x00000000) return CNN_FAIL; // 0,4,20
    if ((*((volatile uint32_t *) 0x50808044)) != 0x00000000) return CNN_FAIL; // 0,4,21
    if ((*((volatile uint32_t *) 0x50808048)) != 0x00000000) return CNN_FAIL; // 0,4,22
    if ((*((volatile uint32_t *) 0x5080804c)) != 0x00000000) return CNN_FAIL; // 0,4,23
    if ((*((volatile uint32_t *) 0x50810040)) != 0x00000000) return CNN_FAIL; // 0,4,24
    if ((*((volatile uint32_t *) 0x50810044)) != 0x00000000) return CNN_FAIL; // 0,4,25
    if ((*((volatile uint32_t *) 0x50810048)) != 0x00000000) return CNN_FAIL; // 0,4,26
    if ((*((volatile uint32_t *) 0x5081004c)) != 0x00000000) return CNN_FAIL; // 0,4,27
    if ((*((volatile uint32_t *) 0x50818040)) != 0x00000000) return CNN_FAIL; // 0,4,28
    if ((*((volatile uint32_t *) 0x50818044)) != 0x00000000) return CNN_FAIL; // 0,4,29
    if ((*((volatile uint32_t *) 0x50400050)) != 0x00000000) return CNN_FAIL; // 0,5,0
    if ((*((volatile uint32_t *) 0x50400054)) != 0x00000000) return CNN_FAIL; // 0,5,1
    if ((*((volatile uint32_t *) 0x50400058)) != 0x00000000) return CNN_FAIL; // 0,5,2
    if ((*((volatile uint32_t *) 0x5040005c)) != 0x00000000) return CNN_FAIL; // 0,5,3
    if ((*((volatile uint32_t *) 0x50408050)) != 0x00000000) return CNN_FAIL; // 0,5,4
    if ((*((volatile uint32_t *) 0x50408054)) != 0x00000000) return CNN_FAIL; // 0,5,5
    if ((*((volatile uint32_t *) 0x50408058)) != 0x00000000) return CNN_FAIL; // 0,5,6
    if ((*((volatile uint32_t *) 0x5040805c)) != 0x00000000) return CNN_FAIL; // 0,5,7
    if ((*((volatile uint32_t *) 0x50410050)) != 0x00000000) return CNN_FAIL; // 0,5,8
    if ((*((volatile uint32_t *) 0x50410054)) != 0x00000000) return CNN_FAIL; // 0,5,9
    if ((*((volatile uint32_t *) 0x50410058)) != 0x00000000) return CNN_FAIL; // 0,5,10
    if ((*((volatile uint32_t *) 0x5041005c)) != 0x00000000) return CNN_FAIL; // 0,5,11
    if ((*((volatile uint32_t *) 0x50418050)) != 0x00000000) return CNN_FAIL; // 0,5,12
    if ((*((volatile uint32_t *) 0x50418054)) != 0x00000000) return CNN_FAIL; // 0,5,13
    if ((*((volatile uint32_t *) 0x50418058)) != 0x00000000) return CNN_FAIL; // 0,5,14
    if ((*((volatile uint32_t *) 0x5041805c)) != 0x00000000) return CNN_FAIL; // 0,5,15
    if ((*((volatile uint32_t *) 0x50800050)) != 0x00000000) return CNN_FAIL; // 0,5,16
    if ((*((volatile uint32_t *) 0x50800054)) != 0x00000000) return CNN_FAIL; // 0,5,17
    if ((*((volatile uint32_t *) 0x50800058)) != 0x00000000) return CNN_FAIL; // 0,5,18
    if ((*((volatile uint32_t *) 0x5080005c)) != 0x00000000) return CNN_FAIL; // 0,5,19
    if ((*((volatile uint32_t *) 0x50808050)) != 0x00000000) return CNN_FAIL; // 0,5,20
    if ((*((volatile uint32_t *) 0x50808054)) != 0x00000000) return CNN_FAIL; // 0,5,21
    if ((*((volatile uint32_t *) 0x50808058)) != 0x00000000) return CNN_FAIL; // 0,5,22
    if ((*((volatile uint32_t *) 0x5080805c)) != 0x00000000) return CNN_FAIL; // 0,5,23
    if ((*((volatile uint32_t *) 0x50810050)) != 0x00000000) return CNN_FAIL; // 0,5,24
    if ((*((volatile uint32_t *) 0x50810054)) != 0x00000000) return CNN_FAIL; // 0,5,25
    if ((*((volatile uint32_t *) 0x50810058)) != 0x00000000) return CNN_FAIL; // 0,5,26
    if ((*((volatile uint32_t *) 0x5081005c)) != 0x00000000) return CNN_FAIL; // 0,5,27
    if ((*((volatile uint32_t *) 0x50818050)) != 0x00000000) return CNN_FAIL; // 0,5,28
    if ((*((volatile uint32_t *) 0x50818054)) != 0x00000000) return CNN_FAIL; // 0,5,29
    if ((*((volatile uint32_t *) 0x50400060)) != 0x00000000) return CNN_FAIL; // 0,6,0
    if ((*((volatile uint32_t *) 0x50400064)) != 0x00000000) return CNN_FAIL; // 0,6,1
    if ((*((volatile uint32_t *) 0x50400068)) != 0x00000000) return CNN_FAIL; // 0,6,2
    if ((*((volatile uint32_t *) 0x5040006c)) != 0x00000000) return CNN_FAIL; // 0,6,3
    if ((*((volatile uint32_t *) 0x50408060)) != 0x00000000) return CNN_FAIL; // 0,6,4
    if ((*((volatile uint32_t *) 0x50408064)) != 0x00000000) return CNN_FAIL; // 0,6,5
    if ((*((volatile uint32_t *) 0x50408068)) != 0x00000000) return CNN_FAIL; // 0,6,6
    if ((*((volatile uint32_t *) 0x5040806c)) != 0x00000000) return CNN_FAIL; // 0,6,7
    if ((*((volatile uint32_t *) 0x50410060)) != 0x00000000) return CNN_FAIL; // 0,6,8
    if ((*((volatile uint32_t *) 0x50410064)) != 0x00000000) return CNN_FAIL; // 0,6,9
    if ((*((volatile uint32_t *) 0x50410068)) != 0x00000000) return CNN_FAIL; // 0,6,10
    if ((*((volatile uint32_t *) 0x5041006c)) != 0x00000000) return CNN_FAIL; // 0,6,11
    if ((*((volatile uint32_t *) 0x50418060)) != 0x00000000) return CNN_FAIL; // 0,6,12
    if ((*((volatile uint32_t *) 0x50418064)) != 0x00000000) return CNN_FAIL; // 0,6,13
    if ((*((volatile uint32_t *) 0x50418068)) != 0x00000000) return CNN_FAIL; // 0,6,14
    if ((*((volatile uint32_t *) 0x5041806c)) != 0x00000000) return CNN_FAIL; // 0,6,15
    if ((*((volatile uint32_t *) 0x50800060)) != 0x00000000) return CNN_FAIL; // 0,6,16
    if ((*((volatile uint32_t *) 0x50800064)) != 0x00000000) return CNN_FAIL; // 0,6,17
    if ((*((volatile uint32_t *) 0x50800068)) != 0x00000000) return CNN_FAIL; // 0,6,18
    if ((*((volatile uint32_t *) 0x5080006c)) != 0x00000000) return CNN_FAIL; // 0,6,19
    if ((*((volatile uint32_t *) 0x50808060)) != 0x00000000) return CNN_FAIL; // 0,6,20
    if ((*((volatile uint32_t *) 0x50808064)) != 0x00000000) return CNN_FAIL; // 0,6,21
    if ((*((volatile uint32_t *) 0x50808068)) != 0x00000000) return CNN_FAIL; // 0,6,22
    if ((*((volatile uint32_t *) 0x5080806c)) != 0x00000000) return CNN_FAIL; // 0,6,23
    if ((*((volatile uint32_t *) 0x50810060)) != 0x00000000) return CNN_FAIL; // 0,6,24
    if ((*((volatile uint32_t *) 0x50810064)) != 0x00000000) return CNN_FAIL; // 0,6,25
    if ((*((volatile uint32_t *) 0x50810068)) != 0x00000000) return CNN_FAIL; // 0,6,26
    if ((*((volatile uint32_t *) 0x5081006c)) != 0x00000000) return CNN_FAIL; // 0,6,27
    if ((*((volatile uint32_t *) 0x50818060)) != 0x00000000) return CNN_FAIL; // 0,6,28
    if ((*((volatile uint32_t *) 0x50818064)) != 0x00000000) return CNN_FAIL; // 0,6,29
    if ((*((volatile uint32_t *) 0x50400070)) != 0x00000000) return CNN_FAIL; // 1,0,0
    if ((*((volatile uint32_t *) 0x50400074)) != 0x00000000) return CNN_FAIL; // 1,0,1
    if ((*((volatile uint32_t *) 0x50400078)) != 0x00000000) return CNN_FAIL; // 1,0,2
    if ((*((volatile uint32_t *) 0x5040007c)) != 0x00000000) return CNN_FAIL; // 1,0,3
    if ((*((volatile uint32_t *) 0x50408070)) != 0x00000000) return CNN_FAIL; // 1,0,4
    if ((*((volatile uint32_t *) 0x50408074)) != 0x00000000) return CNN_FAIL; // 1,0,5
    if ((*((volatile uint32_t *) 0x50408078)) != 0x00000000) return CNN_FAIL; // 1,0,6
    if ((*((volatile uint32_t *) 0x5040807c)) != 0x00000000) return CNN_FAIL; // 1,0,7
    if ((*((volatile uint32_t *) 0x50410070)) != 0x00000000) return CNN_FAIL; // 1,0,8
    if ((*((volatile uint32_t *) 0x50410074)) != 0x00000000) return CNN_FAIL; // 1,0,9
    if ((*((volatile uint32_t *) 0x50410078)) != 0x00000000) return CNN_FAIL; // 1,0,10
    if ((*((volatile uint32_t *) 0x5041007c)) != 0x00000000) return CNN_FAIL; // 1,0,11
    if ((*((volatile uint32_t *) 0x50418070)) != 0x00000000) return CNN_FAIL; // 1,0,12
    if ((*((volatile uint32_t *) 0x50418074)) != 0x00000000) return CNN_FAIL; // 1,0,13
    if ((*((volatile uint32_t *) 0x50418078)) != 0x00000000) return CNN_FAIL; // 1,0,14
    if ((*((volatile uint32_t *) 0x5041807c)) != 0x00000000) return CNN_FAIL; // 1,0,15
    if ((*((volatile uint32_t *) 0x50800070)) != 0x00000000) return CNN_FAIL; // 1,0,16
    if ((*((volatile uint32_t *) 0x50800074)) != 0x00000000) return CNN_FAIL; // 1,0,17
    if ((*((volatile uint32_t *) 0x50800078)) != 0x00000000) return CNN_FAIL; // 1,0,18
    if ((*((volatile uint32_t *) 0x5080007c)) != 0x00000000) return CNN_FAIL; // 1,0,19
    if ((*((volatile uint32_t *) 0x50808070)) != 0x00000000) return CNN_FAIL; // 1,0,20
    if ((*((volatile uint32_t *) 0x50808074)) != 0x00000000) return CNN_FAIL; // 1,0,21
    if ((*((volatile uint32_t *) 0x50808078)) != 0x00000000) return CNN_FAIL; // 1,0,22
    if ((*((volatile uint32_t *) 0x5080807c)) != 0x00000000) return CNN_FAIL; // 1,0,23
    if ((*((volatile uint32_t *) 0x50810070)) != 0x00000000) return CNN_FAIL; // 1,0,24
    if ((*((volatile uint32_t *) 0x50810074)) != 0x00000000) return CNN_FAIL; // 1,0,25
    if ((*((volatile uint32_t *) 0x50810078)) != 0x00000000) return CNN_FAIL; // 1,0,26
    if ((*((volatile uint32_t *) 0x5081007c)) != 0x00000000) return CNN_FAIL; // 1,0,27
    if ((*((volatile uint32_t *) 0x50818070)) != 0x00000000) return CNN_FAIL; // 1,0,28
    if ((*((volatile uint32_t *) 0x50818074)) != 0x00000000) return CNN_FAIL; // 1,0,29
    if ((*((volatile uint32_t *) 0x50400080)) != 0x00000000) return CNN_FAIL; // 1,1,0
    if ((*((volatile uint32_t *) 0x50400084)) != 0x00000000) return CNN_FAIL; // 1,1,1
    if ((*((volatile uint32_t *) 0x50400088)) != 0x00000000) return CNN_FAIL; // 1,1,2
    if ((*((volatile uint32_t *) 0x5040008c)) != 0x00000000) return CNN_FAIL; // 1,1,3
    if ((*((volatile uint32_t *) 0x50408080)) != 0x00000000) return CNN_FAIL; // 1,1,4
    if ((*((volatile uint32_t *) 0x50408084)) != 0x00000000) return CNN_FAIL; // 1,1,5
    if ((*((volatile uint32_t *) 0x50408088)) != 0x00000000) return CNN_FAIL; // 1,1,6
    if ((*((volatile uint32_t *) 0x5040808c)) != 0x00000000) return CNN_FAIL; // 1,1,7
    if ((*((volatile uint32_t *) 0x50410080)) != 0x00000000) return CNN_FAIL; // 1,1,8
    if ((*((volatile uint32_t *) 0x50410084)) != 0x00000000) return CNN_FAIL; // 1,1,9
    if ((*((volatile uint32_t *) 0x50410088)) != 0x00000000) return CNN_FAIL; // 1,1,10
    if ((*((volatile uint32_t *) 0x5041008c)) != 0x00000000) return CNN_FAIL; // 1,1,11
    if ((*((volatile uint32_t *) 0x50418080)) != 0x00000000) return CNN_FAIL; // 1,1,12
    if ((*((volatile uint32_t *) 0x50418084)) != 0x00000000) return CNN_FAIL; // 1,1,13
    if ((*((volatile uint32_t *) 0x50418088)) != 0x00000000) return CNN_FAIL; // 1,1,14
    if ((*((volatile uint32_t *) 0x5041808c)) != 0x00000000) return CNN_FAIL; // 1,1,15
    if ((*((volatile uint32_t *) 0x50800080)) != 0x00000000) return CNN_FAIL; // 1,1,16
    if ((*((volatile uint32_t *) 0x50800084)) != 0x00000000) return CNN_FAIL; // 1,1,17
    if ((*((volatile uint32_t *) 0x50800088)) != 0x00000000) return CNN_FAIL; // 1,1,18
    if ((*((volatile uint32_t *) 0x5080008c)) != 0x00000000) return CNN_FAIL; // 1,1,19
    if ((*((volatile uint32_t *) 0x50808080)) != 0x00000000) return CNN_FAIL; // 1,1,20
    if ((*((volatile uint32_t *) 0x50808084)) != 0x00000000) return CNN_FAIL; // 1,1,21
    if ((*((volatile uint32_t *) 0x50808088)) != 0x00000000) return CNN_FAIL; // 1,1,22
    if ((*((volatile uint32_t *) 0x5080808c)) != 0x00000000) return CNN_FAIL; // 1,1,23
    if ((*((volatile uint32_t *) 0x50810080)) != 0x00000000) return CNN_FAIL; // 1,1,24
    if ((*((volatile uint32_t *) 0x50810084)) != 0x00000000) return CNN_FAIL; // 1,1,25
    if ((*((volatile uint32_t *) 0x50810088)) != 0x00000000) return CNN_FAIL; // 1,1,26
    if ((*((volatile uint32_t *) 0x5081008c)) != 0x00000000) return CNN_FAIL; // 1,1,27
    if ((*((volatile uint32_t *) 0x50818080)) != 0x00000000) return CNN_FAIL; // 1,1,28
    if ((*((volatile uint32_t *) 0x50818084)) != 0x00000000) return CNN_FAIL; // 1,1,29
    if ((*((volatile uint32_t *) 0x50400090)) != 0x00000000) return CNN_FAIL; // 1,2,0
    if ((*((volatile uint32_t *) 0x50400094)) != 0x00000000) return CNN_FAIL; // 1,2,1
    if ((*((volatile uint32_t *) 0x50400098)) != 0x00000000) return CNN_FAIL; // 1,2,2
    if ((*((volatile uint32_t *) 0x5040009c)) != 0x00000000) return CNN_FAIL; // 1,2,3
    if ((*((volatile uint32_t *) 0x50408090)) != 0x00000000) return CNN_FAIL; // 1,2,4
    if ((*((volatile uint32_t *) 0x50408094)) != 0x00000000) return CNN_FAIL; // 1,2,5
    if ((*((volatile uint32_t *) 0x50408098)) != 0x00000000) return CNN_FAIL; // 1,2,6
    if ((*((volatile uint32_t *) 0x5040809c)) != 0x00000000) return CNN_FAIL; // 1,2,7
    if ((*((volatile uint32_t *) 0x50410090)) != 0x00000000) return CNN_FAIL; // 1,2,8
    if ((*((volatile uint32_t *) 0x50410094)) != 0x00000000) return CNN_FAIL; // 1,2,9
    if ((*((volatile uint32_t *) 0x50410098)) != 0x00000000) return CNN_FAIL; // 1,2,10
    if ((*((volatile uint32_t *) 0x5041009c)) != 0x00000000) return CNN_FAIL; // 1,2,11
    if ((*((volatile uint32_t *) 0x50418090)) != 0x00000000) return CNN_FAIL; // 1,2,12
    if ((*((volatile uint32_t *) 0x50418094)) != 0x00000000) return CNN_FAIL; // 1,2,13
    if ((*((volatile uint32_t *) 0x50418098)) != 0x00000000) return CNN_FAIL; // 1,2,14
    if ((*((volatile uint32_t *) 0x5041809c)) != 0x00000000) return CNN_FAIL; // 1,2,15
    if ((*((volatile uint32_t *) 0x50800090)) != 0x00000000) return CNN_FAIL; // 1,2,16
    if ((*((volatile uint32_t *) 0x50800094)) != 0x00000000) return CNN_FAIL; // 1,2,17
    if ((*((volatile uint32_t *) 0x50800098)) != 0x00000000) return CNN_FAIL; // 1,2,18
    if ((*((volatile uint32_t *) 0x5080009c)) != 0x00000000) return CNN_FAIL; // 1,2,19
    if ((*((volatile uint32_t *) 0x50808090)) != 0x00000000) return CNN_FAIL; // 1,2,20
    if ((*((volatile uint32_t *) 0x50808094)) != 0x00000000) return CNN_FAIL; // 1,2,21
    if ((*((volatile uint32_t *) 0x50808098)) != 0x00000000) return CNN_FAIL; // 1,2,22
    if ((*((volatile uint32_t *) 0x5080809c)) != 0x00000000) return CNN_FAIL; // 1,2,23
    if ((*((volatile uint32_t *) 0x50810090)) != 0x00000000) return CNN_FAIL; // 1,2,24
    if ((*((volatile uint32_t *) 0x50810094)) != 0x00000000) return CNN_FAIL; // 1,2,25
    if ((*((volatile uint32_t *) 0x50810098)) != 0x00000000) return CNN_FAIL; // 1,2,26
    if ((*((volatile uint32_t *) 0x5081009c)) != 0x00000000) return CNN_FAIL; // 1,2,27
    if ((*((volatile uint32_t *) 0x50818090)) != 0x00000000) return CNN_FAIL; // 1,2,28
    if ((*((volatile uint32_t *) 0x50818094)) != 0x00000000) return CNN_FAIL; // 1,2,29
    if ((*((volatile uint32_t *) 0x504000a0)) != 0x00000000) return CNN_FAIL; // 1,3,0
    if ((*((volatile uint32_t *) 0x504000a4)) != 0x00000000) return CNN_FAIL; // 1,3,1
    if ((*((volatile uint32_t *) 0x504000a8)) != 0x00000000) return CNN_FAIL; // 1,3,2
    if ((*((volatile uint32_t *) 0x504000ac)) != 0x00000000) return CNN_FAIL; // 1,3,3
    if ((*((volatile uint32_t *) 0x504080a0)) != 0x00000000) return CNN_FAIL; // 1,3,4
    if ((*((volatile uint32_t *) 0x504080a4)) != 0x00000000) return CNN_FAIL; // 1,3,5
    if ((*((volatile uint32_t *) 0x504080a8)) != 0x00000000) return CNN_FAIL; // 1,3,6
    if ((*((volatile uint32_t *) 0x504080ac)) != 0x00000000) return CNN_FAIL; // 1,3,7
    if ((*((volatile uint32_t *) 0x504100a0)) != 0x00000000) return CNN_FAIL; // 1,3,8
    if ((*((volatile uint32_t *) 0x504100a4)) != 0x00000000) return CNN_FAIL; // 1,3,9
    if ((*((volatile uint32_t *) 0x504100a8)) != 0x00000000) return CNN_FAIL; // 1,3,10
    if ((*((volatile uint32_t *) 0x504100ac)) != 0x00000000) return CNN_FAIL; // 1,3,11
    if ((*((volatile uint32_t *) 0x504180a0)) != 0x00000000) return CNN_FAIL; // 1,3,12
    if ((*((volatile uint32_t *) 0x504180a4)) != 0x00000000) return CNN_FAIL; // 1,3,13
    if ((*((volatile uint32_t *) 0x504180a8)) != 0x00000000) return CNN_FAIL; // 1,3,14
    if ((*((volatile uint32_t *) 0x504180ac)) != 0x00000000) return CNN_FAIL; // 1,3,15
    if ((*((volatile uint32_t *) 0x508000a0)) != 0x00000000) return CNN_FAIL; // 1,3,16
    if ((*((volatile uint32_t *) 0x508000a4)) != 0x00000000) return CNN_FAIL; // 1,3,17
    if ((*((volatile uint32_t *) 0x508000a8)) != 0x00000000) return CNN_FAIL; // 1,3,18
    if ((*((volatile uint32_t *) 0x508000ac)) != 0x00000000) return CNN_FAIL; // 1,3,19
    if ((*((volatile uint32_t *) 0x508080a0)) != 0x00000000) return CNN_FAIL; // 1,3,20
    if ((*((volatile uint32_t *) 0x508080a4)) != 0x00000000) return CNN_FAIL; // 1,3,21
    if ((*((volatile uint32_t *) 0x508080a8)) != 0x00000000) return CNN_FAIL; // 1,3,22
    if ((*((volatile uint32_t *) 0x508080ac)) != 0x00000000) return CNN_FAIL; // 1,3,23
    if ((*((volatile uint32_t *) 0x508100a0)) != 0x00000000) return CNN_FAIL; // 1,3,24
    if ((*((volatile uint32_t *) 0x508100a4)) != 0x00000000) return CNN_FAIL; // 1,3,25
    if ((*((volatile uint32_t *) 0x508100a8)) != 0x00000000) return CNN_FAIL; // 1,3,26
    if ((*((volatile uint32_t *) 0x508100ac)) != 0x00000000) return CNN_FAIL; // 1,3,27
    if ((*((volatile uint32_t *) 0x508180a0)) != 0x00000000) return CNN_FAIL; // 1,3,28
    if ((*((volatile uint32_t *) 0x508180a4)) != 0x00000000) return CNN_FAIL; // 1,3,29
    if ((*((volatile uint32_t *) 0x504000b0)) != 0x00000000) return CNN_FAIL; // 1,4,0
    if ((*((volatile uint32_t *) 0x504000b4)) != 0x00000000) return CNN_FAIL; // 1,4,1
    if ((*((volatile uint32_t *) 0x504000b8)) != 0x00000000) return CNN_FAIL; // 1,4,2
    if ((*((volatile uint32_t *) 0x504000bc)) != 0x00000000) return CNN_FAIL; // 1,4,3
    if ((*((volatile uint32_t *) 0x504080b0)) != 0x00000000) return CNN_FAIL; // 1,4,4
    if ((*((volatile uint32_t *) 0x504080b4)) != 0x00000000) return CNN_FAIL; // 1,4,5
    if ((*((volatile uint32_t *) 0x504080b8)) != 0x00000000) return CNN_FAIL; // 1,4,6
    if ((*((volatile uint32_t *) 0x504080bc)) != 0x00000000) return CNN_FAIL; // 1,4,7
    if ((*((volatile uint32_t *) 0x504100b0)) != 0x00000000) return CNN_FAIL; // 1,4,8
    if ((*((volatile uint32_t *) 0x504100b4)) != 0x00000000) return CNN_FAIL; // 1,4,9
    if ((*((volatile uint32_t *) 0x504100b8)) != 0x00000000) return CNN_FAIL; // 1,4,10
    if ((*((volatile uint32_t *) 0x504100bc)) != 0x00000000) return CNN_FAIL; // 1,4,11
    if ((*((volatile uint32_t *) 0x504180b0)) != 0x00000000) return CNN_FAIL; // 1,4,12
    if ((*((volatile uint32_t *) 0x504180b4)) != 0x00000000) return CNN_FAIL; // 1,4,13
    if ((*((volatile uint32_t *) 0x504180b8)) != 0x00000000) return CNN_FAIL; // 1,4,14
    if ((*((volatile uint32_t *) 0x504180bc)) != 0x00000000) return CNN_FAIL; // 1,4,15
    if ((*((volatile uint32_t *) 0x508000b0)) != 0x00000000) return CNN_FAIL; // 1,4,16
    if ((*((volatile uint32_t *) 0x508000b4)) != 0x00000000) return CNN_FAIL; // 1,4,17
    if ((*((volatile uint32_t *) 0x508000b8)) != 0x00000000) return CNN_FAIL; // 1,4,18
    if ((*((volatile uint32_t *) 0x508000bc)) != 0x00000000) return CNN_FAIL; // 1,4,19
    if ((*((volatile uint32_t *) 0x508080b0)) != 0x00000000) return CNN_FAIL; // 1,4,20
    if ((*((volatile uint32_t *) 0x508080b4)) != 0x00000000) return CNN_FAIL; // 1,4,21
    if ((*((volatile uint32_t *) 0x508080b8)) != 0x00000000) return CNN_FAIL; // 1,4,22
    if ((*((volatile uint32_t *) 0x508080bc)) != 0x00000000) return CNN_FAIL; // 1,4,23
    if ((*((volatile uint32_t *) 0x508100b0)) != 0x00000000) return CNN_FAIL; // 1,4,24
    if ((*((volatile uint32_t *) 0x508100b4)) != 0x00000000) return CNN_FAIL; // 1,4,25
    if ((*((volatile uint32_t *) 0x508100b8)) != 0x00000000) return CNN_FAIL; // 1,4,26
    if ((*((volatile uint32_t *) 0x508100bc)) != 0x00000000) return CNN_FAIL; // 1,4,27
    if ((*((volatile uint32_t *) 0x508180b0)) != 0x00000000) return CNN_FAIL; // 1,4,28
    if ((*((volatile uint32_t *) 0x508180b4)) != 0x00000000) return CNN_FAIL; // 1,4,29
    if ((*((volatile uint32_t *) 0x504000c0)) != 0x00000000) return CNN_FAIL; // 1,5,0
    if ((*((volatile uint32_t *) 0x504000c4)) != 0x00000000) return CNN_FAIL; // 1,5,1
    if ((*((volatile uint32_t *) 0x504000c8)) != 0x00000000) return CNN_FAIL; // 1,5,2
    if ((*((volatile uint32_t *) 0x504000cc)) != 0x00000000) return CNN_FAIL; // 1,5,3
    if ((*((volatile uint32_t *) 0x504080c0)) != 0x00000000) return CNN_FAIL; // 1,5,4
    if ((*((volatile uint32_t *) 0x504080c4)) != 0x00000000) return CNN_FAIL; // 1,5,5
    if ((*((volatile uint32_t *) 0x504080c8)) != 0x00000000) return CNN_FAIL; // 1,5,6
    if ((*((volatile uint32_t *) 0x504080cc)) != 0x00000000) return CNN_FAIL; // 1,5,7
    if ((*((volatile uint32_t *) 0x504100c0)) != 0x00000000) return CNN_FAIL; // 1,5,8
    if ((*((volatile uint32_t *) 0x504100c4)) != 0x00000000) return CNN_FAIL; // 1,5,9
    if ((*((volatile uint32_t *) 0x504100c8)) != 0x00000000) return CNN_FAIL; // 1,5,10
    if ((*((volatile uint32_t *) 0x504100cc)) != 0x00000000) return CNN_FAIL; // 1,5,11
    if ((*((volatile uint32_t *) 0x504180c0)) != 0x00000000) return CNN_FAIL; // 1,5,12
    if ((*((volatile uint32_t *) 0x504180c4)) != 0x00000000) return CNN_FAIL; // 1,5,13
    if ((*((volatile uint32_t *) 0x504180c8)) != 0x00000000) return CNN_FAIL; // 1,5,14
    if ((*((volatile uint32_t *) 0x504180cc)) != 0x00000000) return CNN_FAIL; // 1,5,15
    if ((*((volatile uint32_t *) 0x508000c0)) != 0x00000000) return CNN_FAIL; // 1,5,16
    if ((*((volatile uint32_t *) 0x508000c4)) != 0x00000000) return CNN_FAIL; // 1,5,17
    if ((*((volatile uint32_t *) 0x508000c8)) != 0x00000000) return CNN_FAIL; // 1,5,18
    if ((*((volatile uint32_t *) 0x508000cc)) != 0x00000000) return CNN_FAIL; // 1,5,19
    if ((*((volatile uint32_t *) 0x508080c0)) != 0x00000000) return CNN_FAIL; // 1,5,20
    if ((*((volatile uint32_t *) 0x508080c4)) != 0x00000000) return CNN_FAIL; // 1,5,21
    if ((*((volatile uint32_t *) 0x508080c8)) != 0x00000000) return CNN_FAIL; // 1,5,22
    if ((*((volatile uint32_t *) 0x508080cc)) != 0x00000000) return CNN_FAIL; // 1,5,23
    if ((*((volatile uint32_t *) 0x508100c0)) != 0x00000000) return CNN_FAIL; // 1,5,24
    if ((*((volatile uint32_t *) 0x508100c4)) != 0x00000000) return CNN_FAIL; // 1,5,25
    if ((*((volatile uint32_t *) 0x508100c8)) != 0x00000000) return CNN_FAIL; // 1,5,26
    if ((*((volatile uint32_t *) 0x508100cc)) != 0x00000000) return CNN_FAIL; // 1,5,27
    if ((*((volatile uint32_t *) 0x508180c0)) != 0x00000000) return CNN_FAIL; // 1,5,28
    if ((*((volatile uint32_t *) 0x508180c4)) != 0x00000000) return CNN_FAIL; // 1,5,29
    if ((*((volatile uint32_t *) 0x504000d0)) != 0x00000000) return CNN_FAIL; // 1,6,0
    if ((*((volatile uint32_t *) 0x504000d4)) != 0x00000000) return CNN_FAIL; // 1,6,1
    if ((*((volatile uint32_t *) 0x504000d8)) != 0x00000000) return CNN_FAIL; // 1,6,2
    if ((*((volatile uint32_t *) 0x504000dc)) != 0x00000000) return CNN_FAIL; // 1,6,3
    if ((*((volatile uint32_t *) 0x504080d0)) != 0x00000000) return CNN_FAIL; // 1,6,4
    if ((*((volatile uint32_t *) 0x504080d4)) != 0x00000000) return CNN_FAIL; // 1,6,5
    if ((*((volatile uint32_t *) 0x504080d8)) != 0x00000000) return CNN_FAIL; // 1,6,6
    if ((*((volatile uint32_t *) 0x504080dc)) != 0x00000000) return CNN_FAIL; // 1,6,7
    if ((*((volatile uint32_t *) 0x504100d0)) != 0x00000000) return CNN_FAIL; // 1,6,8
    if ((*((volatile uint32_t *) 0x504100d4)) != 0x00000000) return CNN_FAIL; // 1,6,9
    if ((*((volatile uint32_t *) 0x504100d8)) != 0x00000000) return CNN_FAIL; // 1,6,10
    if ((*((volatile uint32_t *) 0x504100dc)) != 0x00000000) return CNN_FAIL; // 1,6,11
    if ((*((volatile uint32_t *) 0x504180d0)) != 0x00000000) return CNN_FAIL; // 1,6,12
    if ((*((volatile uint32_t *) 0x504180d4)) != 0x00000000) return CNN_FAIL; // 1,6,13
    if ((*((volatile uint32_t *) 0x504180d8)) != 0x00000000) return CNN_FAIL; // 1,6,14
    if ((*((volatile uint32_t *) 0x504180dc)) != 0x00000000) return CNN_FAIL; // 1,6,15
    if ((*((volatile uint32_t *) 0x508000d0)) != 0x00000000) return CNN_FAIL; // 1,6,16
    if ((*((volatile uint32_t *) 0x508000d4)) != 0x00000000) return CNN_FAIL; // 1,6,17
    if ((*((volatile uint32_t *) 0x508000d8)) != 0x00000000) return CNN_FAIL; // 1,6,18
    if ((*((volatile uint32_t *) 0x508000dc)) != 0x00000000) return CNN_FAIL; // 1,6,19
    if ((*((volatile uint32_t *) 0x508080d0)) != 0x00000000) return CNN_FAIL; // 1,6,20
    if ((*((volatile uint32_t *) 0x508080d4)) != 0x00000000) return CNN_FAIL; // 1,6,21
    if ((*((volatile uint32_t *) 0x508080d8)) != 0x00000000) return CNN_FAIL; // 1,6,22
    if ((*((volatile uint32_t *) 0x508080dc)) != 0x00000000) return CNN_FAIL; // 1,6,23
    if ((*((volatile uint32_t *) 0x508100d0)) != 0x00000000) return CNN_FAIL; // 1,6,24
    if ((*((volatile uint32_t *) 0x508100d4)) != 0x00000000) return CNN_FAIL; // 1,6,25
    if ((*((volatile uint32_t *) 0x508100d8)) != 0x00000000) return CNN_FAIL; // 1,6,26
    if ((*((volatile uint32_t *) 0x508100dc)) != 0x00000000) return CNN_FAIL; // 1,6,27
    if ((*((volatile uint32_t *) 0x508180d0)) != 0x00000000) return CNN_FAIL; // 1,6,28
    if ((*((volatile uint32_t *) 0x508180d4)) != 0x00000000) return CNN_FAIL; // 1,6,29
    if ((*((volatile uint32_t *) 0x504000e0)) != 0x00000000) return CNN_FAIL; // 2,0,0
    if ((*((volatile uint32_t *) 0x504000e4)) != 0x00000000) return CNN_FAIL; // 2,0,1
    if ((*((volatile uint32_t *) 0x504000e8)) != 0x00000000) return CNN_FAIL; // 2,0,2
    if ((*((volatile uint32_t *) 0x504000ec)) != 0x00000000) return CNN_FAIL; // 2,0,3
    if ((*((volatile uint32_t *) 0x504080e0)) != 0x00000000) return CNN_FAIL; // 2,0,4
    if ((*((volatile uint32_t *) 0x504080e4)) != 0x00000000) return CNN_FAIL; // 2,0,5
    if ((*((volatile uint32_t *) 0x504080e8)) != 0x00000000) return CNN_FAIL; // 2,0,6
    if ((*((volatile uint32_t *) 0x504080ec)) != 0x00000000) return CNN_FAIL; // 2,0,7
    if ((*((volatile uint32_t *) 0x504100e0)) != 0x00000000) return CNN_FAIL; // 2,0,8
    if ((*((volatile uint32_t *) 0x504100e4)) != 0x00000000) return CNN_FAIL; // 2,0,9
    if ((*((volatile uint32_t *) 0x504100e8)) != 0x00000000) return CNN_FAIL; // 2,0,10
    if ((*((volatile uint32_t *) 0x504100ec)) != 0x00000000) return CNN_FAIL; // 2,0,11
    if ((*((volatile uint32_t *) 0x504180e0)) != 0x00000000) return CNN_FAIL; // 2,0,12
    if ((*((volatile uint32_t *) 0x504180e4)) != 0x00000000) return CNN_FAIL; // 2,0,13
    if ((*((volatile uint32_t *) 0x504180e8)) != 0x00000000) return CNN_FAIL; // 2,0,14
    if ((*((volatile uint32_t *) 0x504180ec)) != 0x00000000) return CNN_FAIL; // 2,0,15
    if ((*((volatile uint32_t *) 0x508000e0)) != 0x00000000) return CNN_FAIL; // 2,0,16
    if ((*((volatile uint32_t *) 0x508000e4)) != 0x00000000) return CNN_FAIL; // 2,0,17
    if ((*((volatile uint32_t *) 0x508000e8)) != 0x00000000) return CNN_FAIL; // 2,0,18
    if ((*((volatile uint32_t *) 0x508000ec)) != 0x00000000) return CNN_FAIL; // 2,0,19
    if ((*((volatile uint32_t *) 0x508080e0)) != 0x00000000) return CNN_FAIL; // 2,0,20
    if ((*((volatile uint32_t *) 0x508080e4)) != 0x00000000) return CNN_FAIL; // 2,0,21
    if ((*((volatile uint32_t *) 0x508080e8)) != 0x00000000) return CNN_FAIL; // 2,0,22
    if ((*((volatile uint32_t *) 0x508080ec)) != 0x00000000) return CNN_FAIL; // 2,0,23
    if ((*((volatile uint32_t *) 0x508100e0)) != 0x00000000) return CNN_FAIL; // 2,0,24
    if ((*((volatile uint32_t *) 0x508100e4)) != 0x00000000) return CNN_FAIL; // 2,0,25
    if ((*((volatile uint32_t *) 0x508100e8)) != 0x00000000) return CNN_FAIL; // 2,0,26
    if ((*((volatile uint32_t *) 0x508100ec)) != 0x00000000) return CNN_FAIL; // 2,0,27
    if ((*((volatile uint32_t *) 0x508180e0)) != 0x00000000) return CNN_FAIL; // 2,0,28
    if ((*((volatile uint32_t *) 0x508180e4)) != 0x00000000) return CNN_FAIL; // 2,0,29
    if ((*((volatile uint32_t *) 0x504000f0)) != 0x00000000) return CNN_FAIL; // 2,1,0
    if ((*((volatile uint32_t *) 0x504000f4)) != 0x00000000) return CNN_FAIL; // 2,1,1
    if ((*((volatile uint32_t *) 0x504000f8)) != 0x00000000) return CNN_FAIL; // 2,1,2
    if ((*((volatile uint32_t *) 0x504000fc)) != 0x00000000) return CNN_FAIL; // 2,1,3
    if ((*((volatile uint32_t *) 0x504080f0)) != 0x00000000) return CNN_FAIL; // 2,1,4
    if ((*((volatile uint32_t *) 0x504080f4)) != 0x00000000) return CNN_FAIL; // 2,1,5
    if ((*((volatile uint32_t *) 0x504080f8)) != 0x00000000) return CNN_FAIL; // 2,1,6
    if ((*((volatile uint32_t *) 0x504080fc)) != 0x00000000) return CNN_FAIL; // 2,1,7
    if ((*((volatile uint32_t *) 0x504100f0)) != 0x00000000) return CNN_FAIL; // 2,1,8
    if ((*((volatile uint32_t *) 0x504100f4)) != 0x00000000) return CNN_FAIL; // 2,1,9
    if ((*((volatile uint32_t *) 0x504100f8)) != 0x00000000) return CNN_FAIL; // 2,1,10
    if ((*((volatile uint32_t *) 0x504100fc)) != 0x00000000) return CNN_FAIL; // 2,1,11
    if ((*((volatile uint32_t *) 0x504180f0)) != 0x00000000) return CNN_FAIL; // 2,1,12
    if ((*((volatile uint32_t *) 0x504180f4)) != 0x00000000) return CNN_FAIL; // 2,1,13
    if ((*((volatile uint32_t *) 0x504180f8)) != 0x00000000) return CNN_FAIL; // 2,1,14
    if ((*((volatile uint32_t *) 0x504180fc)) != 0x00000000) return CNN_FAIL; // 2,1,15
    if ((*((volatile uint32_t *) 0x508000f0)) != 0x00000000) return CNN_FAIL; // 2,1,16
    if ((*((volatile uint32_t *) 0x508000f4)) != 0x00000000) return CNN_FAIL; // 2,1,17
    if ((*((volatile uint32_t *) 0x508000f8)) != 0x00000000) return CNN_FAIL; // 2,1,18
    if ((*((volatile uint32_t *) 0x508000fc)) != 0x00000000) return CNN_FAIL; // 2,1,19
    if ((*((volatile uint32_t *) 0x508080f0)) != 0x00000000) return CNN_FAIL; // 2,1,20
    if ((*((volatile uint32_t *) 0x508080f4)) != 0x00000000) return CNN_FAIL; // 2,1,21
    if ((*((volatile uint32_t *) 0x508080f8)) != 0x00000000) return CNN_FAIL; // 2,1,22
    if ((*((volatile uint32_t *) 0x508080fc)) != 0x00000000) return CNN_FAIL; // 2,1,23
    if ((*((volatile uint32_t *) 0x508100f0)) != 0x00000000) return CNN_FAIL; // 2,1,24
    if ((*((volatile uint32_t *) 0x508100f4)) != 0x00000000) return CNN_FAIL; // 2,1,25
    if ((*((volatile uint32_t *) 0x508100f8)) != 0x00000000) return CNN_FAIL; // 2,1,26
    if ((*((volatile uint32_t *) 0x508100fc)) != 0x00000000) return CNN_FAIL; // 2,1,27
    if ((*((volatile uint32_t *) 0x508180f0)) != 0x00000000) return CNN_FAIL; // 2,1,28
    if ((*((volatile uint32_t *) 0x508180f4)) != 0x00000000) return CNN_FAIL; // 2,1,29
    if ((*((volatile uint32_t *) 0x50400100)) != 0x00000000) return CNN_FAIL; // 2,2,0
    if ((*((volatile uint32_t *) 0x50400104)) != 0x00000000) return CNN_FAIL; // 2,2,1
    if ((*((volatile uint32_t *) 0x50400108)) != 0x00000000) return CNN_FAIL; // 2,2,2
    if ((*((volatile uint32_t *) 0x5040010c)) != 0x00000000) return CNN_FAIL; // 2,2,3
    if ((*((volatile uint32_t *) 0x50408100)) != 0x00000000) return CNN_FAIL; // 2,2,4
    if ((*((volatile uint32_t *) 0x50408104)) != 0x00000000) return CNN_FAIL; // 2,2,5
    if ((*((volatile uint32_t *) 0x50408108)) != 0x00000000) return CNN_FAIL; // 2,2,6
    if ((*((volatile uint32_t *) 0x5040810c)) != 0x00000000) return CNN_FAIL; // 2,2,7
    if ((*((volatile uint32_t *) 0x50410100)) != 0x00000000) return CNN_FAIL; // 2,2,8
    if ((*((volatile uint32_t *) 0x50410104)) != 0x00000000) return CNN_FAIL; // 2,2,9
    if ((*((volatile uint32_t *) 0x50410108)) != 0x00000000) return CNN_FAIL; // 2,2,10
    if ((*((volatile uint32_t *) 0x5041010c)) != 0x00000000) return CNN_FAIL; // 2,2,11
    if ((*((volatile uint32_t *) 0x50418100)) != 0x00000000) return CNN_FAIL; // 2,2,12
    if ((*((volatile uint32_t *) 0x50418104)) != 0x00000000) return CNN_FAIL; // 2,2,13
    if ((*((volatile uint32_t *) 0x50418108)) != 0x00000000) return CNN_FAIL; // 2,2,14
    if ((*((volatile uint32_t *) 0x5041810c)) != 0x00000000) return CNN_FAIL; // 2,2,15
    if ((*((volatile uint32_t *) 0x50800100)) != 0x00000000) return CNN_FAIL; // 2,2,16
    if ((*((volatile uint32_t *) 0x50800104)) != 0x00000000) return CNN_FAIL; // 2,2,17
    if ((*((volatile uint32_t *) 0x50800108)) != 0x00000000) return CNN_FAIL; // 2,2,18
    if ((*((volatile uint32_t *) 0x5080010c)) != 0x00000000) return CNN_FAIL; // 2,2,19
    if ((*((volatile uint32_t *) 0x50808100)) != 0x00000000) return CNN_FAIL; // 2,2,20
    if ((*((volatile uint32_t *) 0x50808104)) != 0x00000000) return CNN_FAIL; // 2,2,21
    if ((*((volatile uint32_t *) 0x50808108)) != 0x00000000) return CNN_FAIL; // 2,2,22
    if ((*((volatile uint32_t *) 0x5080810c)) != 0x00000000) return CNN_FAIL; // 2,2,23
    if ((*((volatile uint32_t *) 0x50810100)) != 0x00000000) return CNN_FAIL; // 2,2,24
    if ((*((volatile uint32_t *) 0x50810104)) != 0x00000000) return CNN_FAIL; // 2,2,25
    if ((*((volatile uint32_t *) 0x50810108)) != 0x00000000) return CNN_FAIL; // 2,2,26
    if ((*((volatile uint32_t *) 0x5081010c)) != 0x00000000) return CNN_FAIL; // 2,2,27
    if ((*((volatile uint32_t *) 0x50818100)) != 0x00000000) return CNN_FAIL; // 2,2,28
    if ((*((volatile uint32_t *) 0x50818104)) != 0x00000000) return CNN_FAIL; // 2,2,29
    if ((*((volatile uint32_t *) 0x50400110)) != 0x00000000) return CNN_FAIL; // 2,3,0
    if ((*((volatile uint32_t *) 0x50400114)) != 0x00000000) return CNN_FAIL; // 2,3,1
    if ((*((volatile uint32_t *) 0x50400118)) != 0x00000000) return CNN_FAIL; // 2,3,2
    if ((*((volatile uint32_t *) 0x5040011c)) != 0x00000000) return CNN_FAIL; // 2,3,3
    if ((*((volatile uint32_t *) 0x50408110)) != 0x00000000) return CNN_FAIL; // 2,3,4
    if ((*((volatile uint32_t *) 0x50408114)) != 0x00000000) return CNN_FAIL; // 2,3,5
    if ((*((volatile uint32_t *) 0x50408118)) != 0x00000000) return CNN_FAIL; // 2,3,6
    if ((*((volatile uint32_t *) 0x5040811c)) != 0x00000000) return CNN_FAIL; // 2,3,7
    if ((*((volatile uint32_t *) 0x50410110)) != 0x00000000) return CNN_FAIL; // 2,3,8
    if ((*((volatile uint32_t *) 0x50410114)) != 0x00000000) return CNN_FAIL; // 2,3,9
    if ((*((volatile uint32_t *) 0x50410118)) != 0x00000000) return CNN_FAIL; // 2,3,10
    if ((*((volatile uint32_t *) 0x5041011c)) != 0x00000000) return CNN_FAIL; // 2,3,11
    if ((*((volatile uint32_t *) 0x50418110)) != 0x00000000) return CNN_FAIL; // 2,3,12
    if ((*((volatile uint32_t *) 0x50418114)) != 0x00000000) return CNN_FAIL; // 2,3,13
    if ((*((volatile uint32_t *) 0x50418118)) != 0x00000000) return CNN_FAIL; // 2,3,14
    if ((*((volatile uint32_t *) 0x5041811c)) != 0x00000000) return CNN_FAIL; // 2,3,15
    if ((*((volatile uint32_t *) 0x50800110)) != 0x00000000) return CNN_FAIL; // 2,3,16
    if ((*((volatile uint32_t *) 0x50800114)) != 0x00000000) return CNN_FAIL; // 2,3,17
    if ((*((volatile uint32_t *) 0x50800118)) != 0x00000000) return CNN_FAIL; // 2,3,18
    if ((*((volatile uint32_t *) 0x5080011c)) != 0x00000000) return CNN_FAIL; // 2,3,19
    if ((*((volatile uint32_t *) 0x50808110)) != 0x00000000) return CNN_FAIL; // 2,3,20
    if ((*((volatile uint32_t *) 0x50808114)) != 0x00000000) return CNN_FAIL; // 2,3,21
    if ((*((volatile uint32_t *) 0x50808118)) != 0x00000000) return CNN_FAIL; // 2,3,22
    if ((*((volatile uint32_t *) 0x5080811c)) != 0x00000000) return CNN_FAIL; // 2,3,23
    if ((*((volatile uint32_t *) 0x50810110)) != 0x00000000) return CNN_FAIL; // 2,3,24
    if ((*((volatile uint32_t *) 0x50810114)) != 0x00000000) return CNN_FAIL; // 2,3,25
    if ((*((volatile uint32_t *) 0x50810118)) != 0x00000000) return CNN_FAIL; // 2,3,26
    if ((*((volatile uint32_t *) 0x5081011c)) != 0x00000000) return CNN_FAIL; // 2,3,27
    if ((*((volatile uint32_t *) 0x50818110)) != 0x00000000) return CNN_FAIL; // 2,3,28
    if ((*((volatile uint32_t *) 0x50818114)) != 0x00000000) return CNN_FAIL; // 2,3,29
    if ((*((volatile uint32_t *) 0x50400120)) != 0x00000000) return CNN_FAIL; // 2,4,0
    if ((*((volatile uint32_t *) 0x50400124)) != 0x00000000) return CNN_FAIL; // 2,4,1
    if ((*((volatile uint32_t *) 0x50400128)) != 0x00000000) return CNN_FAIL; // 2,4,2
    if ((*((volatile uint32_t *) 0x5040012c)) != 0x00000000) return CNN_FAIL; // 2,4,3
    if ((*((volatile uint32_t *) 0x50408120)) != 0x00000000) return CNN_FAIL; // 2,4,4
    if ((*((volatile uint32_t *) 0x50408124)) != 0x00000000) return CNN_FAIL; // 2,4,5
    if ((*((volatile uint32_t *) 0x50408128)) != 0x00000000) return CNN_FAIL; // 2,4,6
    if ((*((volatile uint32_t *) 0x5040812c)) != 0x00000000) return CNN_FAIL; // 2,4,7
    if ((*((volatile uint32_t *) 0x50410120)) != 0x00000000) return CNN_FAIL; // 2,4,8
    if ((*((volatile uint32_t *) 0x50410124)) != 0x00000000) return CNN_FAIL; // 2,4,9
    if ((*((volatile uint32_t *) 0x50410128)) != 0x00000000) return CNN_FAIL; // 2,4,10
    if ((*((volatile uint32_t *) 0x5041012c)) != 0x00000000) return CNN_FAIL; // 2,4,11
    if ((*((volatile uint32_t *) 0x50418120)) != 0x00000000) return CNN_FAIL; // 2,4,12
    if ((*((volatile uint32_t *) 0x50418124)) != 0x00000000) return CNN_FAIL; // 2,4,13
    if ((*((volatile uint32_t *) 0x50418128)) != 0x00000000) return CNN_FAIL; // 2,4,14
    if ((*((volatile uint32_t *) 0x5041812c)) != 0x00000000) return CNN_FAIL; // 2,4,15
    if ((*((volatile uint32_t *) 0x50800120)) != 0x00000000) return CNN_FAIL; // 2,4,16
    if ((*((volatile uint32_t *) 0x50800124)) != 0x00000000) return CNN_FAIL; // 2,4,17
    if ((*((volatile uint32_t *) 0x50800128)) != 0x00000000) return CNN_FAIL; // 2,4,18
    if ((*((volatile uint32_t *) 0x5080012c)) != 0x00000000) return CNN_FAIL; // 2,4,19
    if ((*((volatile uint32_t *) 0x50808120)) != 0x00000000) return CNN_FAIL; // 2,4,20
    if ((*((volatile uint32_t *) 0x50808124)) != 0x00000000) return CNN_FAIL; // 2,4,21
    if ((*((volatile uint32_t *) 0x50808128)) != 0x00000000) return CNN_FAIL; // 2,4,22
    if ((*((volatile uint32_t *) 0x5080812c)) != 0x00000000) return CNN_FAIL; // 2,4,23
    if ((*((volatile uint32_t *) 0x50810120)) != 0x00000000) return CNN_FAIL; // 2,4,24
    if ((*((volatile uint32_t *) 0x50810124)) != 0x00000000) return CNN_FAIL; // 2,4,25
    if ((*((volatile uint32_t *) 0x50810128)) != 0x00000000) return CNN_FAIL; // 2,4,26
    if ((*((volatile uint32_t *) 0x5081012c)) != 0x00000000) return CNN_FAIL; // 2,4,27
    if ((*((volatile uint32_t *) 0x50818120)) != 0x00000000) return CNN_FAIL; // 2,4,28
    if ((*((volatile uint32_t *) 0x50818124)) != 0x00000000) return CNN_FAIL; // 2,4,29
    if ((*((volatile uint32_t *) 0x50400130)) != 0x00000000) return CNN_FAIL; // 2,5,0
    if ((*((volatile uint32_t *) 0x50400134)) != 0x00000000) return CNN_FAIL; // 2,5,1
    if ((*((volatile uint32_t *) 0x50400138)) != 0x00000000) return CNN_FAIL; // 2,5,2
    if ((*((volatile uint32_t *) 0x5040013c)) != 0x00000000) return CNN_FAIL; // 2,5,3
    if ((*((volatile uint32_t *) 0x50408130)) != 0x00000000) return CNN_FAIL; // 2,5,4
    if ((*((volatile uint32_t *) 0x50408134)) != 0x00000000) return CNN_FAIL; // 2,5,5
    if ((*((volatile uint32_t *) 0x50408138)) != 0x00000000) return CNN_FAIL; // 2,5,6
    if ((*((volatile uint32_t *) 0x5040813c)) != 0x00000000) return CNN_FAIL; // 2,5,7
    if ((*((volatile uint32_t *) 0x50410130)) != 0x00000000) return CNN_FAIL; // 2,5,8
    if ((*((volatile uint32_t *) 0x50410134)) != 0x00000000) return CNN_FAIL; // 2,5,9
    if ((*((volatile uint32_t *) 0x50410138)) != 0x00000000) return CNN_FAIL; // 2,5,10
    if ((*((volatile uint32_t *) 0x5041013c)) != 0x00000000) return CNN_FAIL; // 2,5,11
    if ((*((volatile uint32_t *) 0x50418130)) != 0x00000000) return CNN_FAIL; // 2,5,12
    if ((*((volatile uint32_t *) 0x50418134)) != 0x00000000) return CNN_FAIL; // 2,5,13
    if ((*((volatile uint32_t *) 0x50418138)) != 0x00000000) return CNN_FAIL; // 2,5,14
    if ((*((volatile uint32_t *) 0x5041813c)) != 0x00000000) return CNN_FAIL; // 2,5,15
    if ((*((volatile uint32_t *) 0x50800130)) != 0x00000000) return CNN_FAIL; // 2,5,16
    if ((*((volatile uint32_t *) 0x50800134)) != 0x00000000) return CNN_FAIL; // 2,5,17
    if ((*((volatile uint32_t *) 0x50800138)) != 0x00000000) return CNN_FAIL; // 2,5,18
    if ((*((volatile uint32_t *) 0x5080013c)) != 0x00000000) return CNN_FAIL; // 2,5,19
    if ((*((volatile uint32_t *) 0x50808130)) != 0x00000000) return CNN_FAIL; // 2,5,20
    if ((*((volatile uint32_t *) 0x50808134)) != 0x00000000) return CNN_FAIL; // 2,5,21
    if ((*((volatile uint32_t *) 0x50808138)) != 0x00000000) return CNN_FAIL; // 2,5,22
    if ((*((volatile uint32_t *) 0x5080813c)) != 0x00000000) return CNN_FAIL; // 2,5,23
    if ((*((volatile uint32_t *) 0x50810130)) != 0x00000000) return CNN_FAIL; // 2,5,24
    if ((*((volatile uint32_t *) 0x50810134)) != 0x00000000) return CNN_FAIL; // 2,5,25
    if ((*((volatile uint32_t *) 0x50810138)) != 0x00000000) return CNN_FAIL; // 2,5,26
    if ((*((volatile uint32_t *) 0x5081013c)) != 0x00000000) return CNN_FAIL; // 2,5,27
    if ((*((volatile uint32_t *) 0x50818130)) != 0x00000000) return CNN_FAIL; // 2,5,28
    if ((*((volatile uint32_t *) 0x50818134)) != 0x00000000) return CNN_FAIL; // 2,5,29
    if ((*((volatile uint32_t *) 0x50400140)) != 0x00000000) return CNN_FAIL; // 2,6,0
    if ((*((volatile uint32_t *) 0x50400144)) != 0x00000000) return CNN_FAIL; // 2,6,1
    if ((*((volatile uint32_t *) 0x50400148)) != 0x00000000) return CNN_FAIL; // 2,6,2
    if ((*((volatile uint32_t *) 0x5040014c)) != 0x00000000) return CNN_FAIL; // 2,6,3
    if ((*((volatile uint32_t *) 0x50408140)) != 0x00000000) return CNN_FAIL; // 2,6,4
    if ((*((volatile uint32_t *) 0x50408144)) != 0x00000000) return CNN_FAIL; // 2,6,5
    if ((*((volatile uint32_t *) 0x50408148)) != 0x00000000) return CNN_FAIL; // 2,6,6
    if ((*((volatile uint32_t *) 0x5040814c)) != 0x00000000) return CNN_FAIL; // 2,6,7
    if ((*((volatile uint32_t *) 0x50410140)) != 0x00000000) return CNN_FAIL; // 2,6,8
    if ((*((volatile uint32_t *) 0x50410144)) != 0x00000000) return CNN_FAIL; // 2,6,9
    if ((*((volatile uint32_t *) 0x50410148)) != 0x00000000) return CNN_FAIL; // 2,6,10
    if ((*((volatile uint32_t *) 0x5041014c)) != 0x00000000) return CNN_FAIL; // 2,6,11
    if ((*((volatile uint32_t *) 0x50418140)) != 0x00000000) return CNN_FAIL; // 2,6,12
    if ((*((volatile uint32_t *) 0x50418144)) != 0x00000000) return CNN_FAIL; // 2,6,13
    if ((*((volatile uint32_t *) 0x50418148)) != 0x00000000) return CNN_FAIL; // 2,6,14
    if ((*((volatile uint32_t *) 0x5041814c)) != 0x00000000) return CNN_FAIL; // 2,6,15
    if ((*((volatile uint32_t *) 0x50800140)) != 0x00000000) return CNN_FAIL; // 2,6,16
    if ((*((volatile uint32_t *) 0x50800144)) != 0x00000000) return CNN_FAIL; // 2,6,17
    if ((*((volatile uint32_t *) 0x50800148)) != 0x00000000) return CNN_FAIL; // 2,6,18
    if ((*((volatile uint32_t *) 0x5080014c)) != 0x00000000) return CNN_FAIL; // 2,6,19
    if ((*((volatile uint32_t *) 0x50808140)) != 0x00000000) return CNN_FAIL; // 2,6,20
    if ((*((volatile uint32_t *) 0x50808144)) != 0x00000000) return CNN_FAIL; // 2,6,21
    if ((*((volatile uint32_t *) 0x50808148)) != 0x00000000) return CNN_FAIL; // 2,6,22
    if ((*((volatile uint32_t *) 0x5080814c)) != 0x00000000) return CNN_FAIL; // 2,6,23
    if ((*((volatile uint32_t *) 0x50810140)) != 0x00000000) return CNN_FAIL; // 2,6,24
    if ((*((volatile uint32_t *) 0x50810144)) != 0x00000000) return CNN_FAIL; // 2,6,25
    if ((*((volatile uint32_t *) 0x50810148)) != 0x00000000) return CNN_FAIL; // 2,6,26
    if ((*((volatile uint32_t *) 0x5081014c)) != 0x00000000) return CNN_FAIL; // 2,6,27
    if ((*((volatile uint32_t *) 0x50818140)) != 0x00000000) return CNN_FAIL; // 2,6,28
    if ((*((volatile uint32_t *) 0x50818144)) != 0x00000000) return CNN_FAIL; // 2,6,29
    if ((*((volatile uint32_t *) 0x50400150)) != 0x00000000) return CNN_FAIL; // 3,0,0
    if ((*((volatile uint32_t *) 0x50400154)) != 0x00000000) return CNN_FAIL; // 3,0,1
    if ((*((volatile uint32_t *) 0x50400158)) != 0x00000000) return CNN_FAIL; // 3,0,2
    if ((*((volatile uint32_t *) 0x5040015c)) != 0x00000000) return CNN_FAIL; // 3,0,3
    if ((*((volatile uint32_t *) 0x50408150)) != 0x00000000) return CNN_FAIL; // 3,0,4
    if ((*((volatile uint32_t *) 0x50408154)) != 0x00000000) return CNN_FAIL; // 3,0,5
    if ((*((volatile uint32_t *) 0x50408158)) != 0x00000000) return CNN_FAIL; // 3,0,6
    if ((*((volatile uint32_t *) 0x5040815c)) != 0x00000000) return CNN_FAIL; // 3,0,7
    if ((*((volatile uint32_t *) 0x50410150)) != 0x00000000) return CNN_FAIL; // 3,0,8
    if ((*((volatile uint32_t *) 0x50410154)) != 0x00000000) return CNN_FAIL; // 3,0,9
    if ((*((volatile uint32_t *) 0x50410158)) != 0x00000000) return CNN_FAIL; // 3,0,10
    if ((*((volatile uint32_t *) 0x5041015c)) != 0x00000000) return CNN_FAIL; // 3,0,11
    if ((*((volatile uint32_t *) 0x50418150)) != 0x00000000) return CNN_FAIL; // 3,0,12
    if ((*((volatile uint32_t *) 0x50418154)) != 0x00000000) return CNN_FAIL; // 3,0,13
    if ((*((volatile uint32_t *) 0x50418158)) != 0x00000000) return CNN_FAIL; // 3,0,14
    if ((*((volatile uint32_t *) 0x5041815c)) != 0x00000000) return CNN_FAIL; // 3,0,15
    if ((*((volatile uint32_t *) 0x50800150)) != 0x00000000) return CNN_FAIL; // 3,0,16
    if ((*((volatile uint32_t *) 0x50800154)) != 0x00000000) return CNN_FAIL; // 3,0,17
    if ((*((volatile uint32_t *) 0x50800158)) != 0x00000000) return CNN_FAIL; // 3,0,18
    if ((*((volatile uint32_t *) 0x5080015c)) != 0x00000000) return CNN_FAIL; // 3,0,19
    if ((*((volatile uint32_t *) 0x50808150)) != 0x00000000) return CNN_FAIL; // 3,0,20
    if ((*((volatile uint32_t *) 0x50808154)) != 0x00000000) return CNN_FAIL; // 3,0,21
    if ((*((volatile uint32_t *) 0x50808158)) != 0x00000000) return CNN_FAIL; // 3,0,22
    if ((*((volatile uint32_t *) 0x5080815c)) != 0x00000000) return CNN_FAIL; // 3,0,23
    if ((*((volatile uint32_t *) 0x50810150)) != 0x00000000) return CNN_FAIL; // 3,0,24
    if ((*((volatile uint32_t *) 0x50810154)) != 0x00000000) return CNN_FAIL; // 3,0,25
    if ((*((volatile uint32_t *) 0x50810158)) != 0x00000000) return CNN_FAIL; // 3,0,26
    if ((*((volatile uint32_t *) 0x5081015c)) != 0x00000000) return CNN_FAIL; // 3,0,27
    if ((*((volatile uint32_t *) 0x50818150)) != 0x00000000) return CNN_FAIL; // 3,0,28
    if ((*((volatile uint32_t *) 0x50818154)) != 0x00000000) return CNN_FAIL; // 3,0,29
    if ((*((volatile uint32_t *) 0x50400160)) != 0x00000000) return CNN_FAIL; // 3,1,0
    if ((*((volatile uint32_t *) 0x50400164)) != 0x00000000) return CNN_FAIL; // 3,1,1
    if ((*((volatile uint32_t *) 0x50400168)) != 0x00000000) return CNN_FAIL; // 3,1,2
    if ((*((volatile uint32_t *) 0x5040016c)) != 0x00000000) return CNN_FAIL; // 3,1,3
    if ((*((volatile uint32_t *) 0x50408160)) != 0x00000000) return CNN_FAIL; // 3,1,4
    if ((*((volatile uint32_t *) 0x50408164)) != 0x00000000) return CNN_FAIL; // 3,1,5
    if ((*((volatile uint32_t *) 0x50408168)) != 0x00000000) return CNN_FAIL; // 3,1,6
    if ((*((volatile uint32_t *) 0x5040816c)) != 0x00000000) return CNN_FAIL; // 3,1,7
    if ((*((volatile uint32_t *) 0x50410160)) != 0x00000000) return CNN_FAIL; // 3,1,8
    if ((*((volatile uint32_t *) 0x50410164)) != 0x00000000) return CNN_FAIL; // 3,1,9
    if ((*((volatile uint32_t *) 0x50410168)) != 0x00000000) return CNN_FAIL; // 3,1,10
    if ((*((volatile uint32_t *) 0x5041016c)) != 0x00000000) return CNN_FAIL; // 3,1,11
    if ((*((volatile uint32_t *) 0x50418160)) != 0x00000000) return CNN_FAIL; // 3,1,12
    if ((*((volatile uint32_t *) 0x50418164)) != 0x00000000) return CNN_FAIL; // 3,1,13
    if ((*((volatile uint32_t *) 0x50418168)) != 0x00000000) return CNN_FAIL; // 3,1,14
    if ((*((volatile uint32_t *) 0x5041816c)) != 0x00000000) return CNN_FAIL; // 3,1,15
    if ((*((volatile uint32_t *) 0x50800160)) != 0x00000000) return CNN_FAIL; // 3,1,16
    if ((*((volatile uint32_t *) 0x50800164)) != 0x00000000) return CNN_FAIL; // 3,1,17
    if ((*((volatile uint32_t *) 0x50800168)) != 0x00000000) return CNN_FAIL; // 3,1,18
    if ((*((volatile uint32_t *) 0x5080016c)) != 0x00000000) return CNN_FAIL; // 3,1,19
    if ((*((volatile uint32_t *) 0x50808160)) != 0x00000000) return CNN_FAIL; // 3,1,20
    if ((*((volatile uint32_t *) 0x50808164)) != 0x00000000) return CNN_FAIL; // 3,1,21
    if ((*((volatile uint32_t *) 0x50808168)) != 0x00000000) return CNN_FAIL; // 3,1,22
    if ((*((volatile uint32_t *) 0x5080816c)) != 0x00000000) return CNN_FAIL; // 3,1,23
    if ((*((volatile uint32_t *) 0x50810160)) != 0x00000000) return CNN_FAIL; // 3,1,24
    if ((*((volatile uint32_t *) 0x50810164)) != 0x00000000) return CNN_FAIL; // 3,1,25
    if ((*((volatile uint32_t *) 0x50810168)) != 0x00000000) return CNN_FAIL; // 3,1,26
    if ((*((volatile uint32_t *) 0x5081016c)) != 0x00000000) return CNN_FAIL; // 3,1,27
    if ((*((volatile uint32_t *) 0x50818160)) != 0x00000000) return CNN_FAIL; // 3,1,28
    if ((*((volatile uint32_t *) 0x50818164)) != 0x00000000) return CNN_FAIL; // 3,1,29
    if ((*((volatile uint32_t *) 0x50400170)) != 0x00000000) return CNN_FAIL; // 3,2,0
    if ((*((volatile uint32_t *) 0x50400174)) != 0x00000000) return CNN_FAIL; // 3,2,1
    if ((*((volatile uint32_t *) 0x50400178)) != 0x00000000) return CNN_FAIL; // 3,2,2
    if ((*((volatile uint32_t *) 0x5040017c)) != 0x00000000) return CNN_FAIL; // 3,2,3
    if ((*((volatile uint32_t *) 0x50408170)) != 0x00000000) return CNN_FAIL; // 3,2,4
    if ((*((volatile uint32_t *) 0x50408174)) != 0x00000000) return CNN_FAIL; // 3,2,5
    if ((*((volatile uint32_t *) 0x50408178)) != 0x00000000) return CNN_FAIL; // 3,2,6
    if ((*((volatile uint32_t *) 0x5040817c)) != 0x00000000) return CNN_FAIL; // 3,2,7
    if ((*((volatile uint32_t *) 0x50410170)) != 0x00000000) return CNN_FAIL; // 3,2,8
    if ((*((volatile uint32_t *) 0x50410174)) != 0x00000000) return CNN_FAIL; // 3,2,9
    if ((*((volatile uint32_t *) 0x50410178)) != 0x00000000) return CNN_FAIL; // 3,2,10
    if ((*((volatile uint32_t *) 0x5041017c)) != 0x00000000) return CNN_FAIL; // 3,2,11
    if ((*((volatile uint32_t *) 0x50418170)) != 0x00000000) return CNN_FAIL; // 3,2,12
    if ((*((volatile uint32_t *) 0x50418174)) != 0x00000000) return CNN_FAIL; // 3,2,13
    if ((*((volatile uint32_t *) 0x50418178)) != 0x00000000) return CNN_FAIL; // 3,2,14
    if ((*((volatile uint32_t *) 0x5041817c)) != 0x00000000) return CNN_FAIL; // 3,2,15
    if ((*((volatile uint32_t *) 0x50800170)) != 0x00000000) return CNN_FAIL; // 3,2,16
    if ((*((volatile uint32_t *) 0x50800174)) != 0x00000000) return CNN_FAIL; // 3,2,17
    if ((*((volatile uint32_t *) 0x50800178)) != 0x00000000) return CNN_FAIL; // 3,2,18
    if ((*((volatile uint32_t *) 0x5080017c)) != 0x00000000) return CNN_FAIL; // 3,2,19
    if ((*((volatile uint32_t *) 0x50808170)) != 0x00000000) return CNN_FAIL; // 3,2,20
    if ((*((volatile uint32_t *) 0x50808174)) != 0x00000000) return CNN_FAIL; // 3,2,21
    if ((*((volatile uint32_t *) 0x50808178)) != 0x00000000) return CNN_FAIL; // 3,2,22
    if ((*((volatile uint32_t *) 0x5080817c)) != 0x00000000) return CNN_FAIL; // 3,2,23
    if ((*((volatile uint32_t *) 0x50810170)) != 0x00000000) return CNN_FAIL; // 3,2,24
    if ((*((volatile uint32_t *) 0x50810174)) != 0x00000000) return CNN_FAIL; // 3,2,25
    if ((*((volatile uint32_t *) 0x50810178)) != 0x00000000) return CNN_FAIL; // 3,2,26
    if ((*((volatile uint32_t *) 0x5081017c)) != 0x00000000) return CNN_FAIL; // 3,2,27
    if ((*((volatile uint32_t *) 0x50818170)) != 0x00000000) return CNN_FAIL; // 3,2,28
    if ((*((volatile uint32_t *) 0x50818174)) != 0x00000000) return CNN_FAIL; // 3,2,29
    if ((*((volatile uint32_t *) 0x50400180)) != 0x00000000) return CNN_FAIL; // 3,3,0
    if ((*((volatile uint32_t *) 0x50400184)) != 0x00000000) return CNN_FAIL; // 3,3,1
    if ((*((volatile uint32_t *) 0x50400188)) != 0x00000000) return CNN_FAIL; // 3,3,2
    if ((*((volatile uint32_t *) 0x5040018c)) != 0x00000000) return CNN_FAIL; // 3,3,3
    if ((*((volatile uint32_t *) 0x50408180)) != 0x00000000) return CNN_FAIL; // 3,3,4
    if ((*((volatile uint32_t *) 0x50408184)) != 0x00000000) return CNN_FAIL; // 3,3,5
    if ((*((volatile uint32_t *) 0x50408188)) != 0x00000000) return CNN_FAIL; // 3,3,6
    if ((*((volatile uint32_t *) 0x5040818c)) != 0x00000000) return CNN_FAIL; // 3,3,7
    if ((*((volatile uint32_t *) 0x50410180)) != 0x00000000) return CNN_FAIL; // 3,3,8
    if ((*((volatile uint32_t *) 0x50410184)) != 0x00000000) return CNN_FAIL; // 3,3,9
    if ((*((volatile uint32_t *) 0x50410188)) != 0x00000000) return CNN_FAIL; // 3,3,10
    if ((*((volatile uint32_t *) 0x5041018c)) != 0x00000000) return CNN_FAIL; // 3,3,11
    if ((*((volatile uint32_t *) 0x50418180)) != 0x00000000) return CNN_FAIL; // 3,3,12
    if ((*((volatile uint32_t *) 0x50418184)) != 0x00000000) return CNN_FAIL; // 3,3,13
    if ((*((volatile uint32_t *) 0x50418188)) != 0x00000000) return CNN_FAIL; // 3,3,14
    if ((*((volatile uint32_t *) 0x5041818c)) != 0x00000000) return CNN_FAIL; // 3,3,15
    if ((*((volatile uint32_t *) 0x50800180)) != 0x00000000) return CNN_FAIL; // 3,3,16
    if ((*((volatile uint32_t *) 0x50800184)) != 0x00000000) return CNN_FAIL; // 3,3,17
    if ((*((volatile uint32_t *) 0x50800188)) != 0x00000000) return CNN_FAIL; // 3,3,18
    if ((*((volatile uint32_t *) 0x5080018c)) != 0x00000000) return CNN_FAIL; // 3,3,19
    if ((*((volatile uint32_t *) 0x50808180)) != 0x00000000) return CNN_FAIL; // 3,3,20
    if ((*((volatile uint32_t *) 0x50808184)) != 0x00000000) return CNN_FAIL; // 3,3,21
    if ((*((volatile uint32_t *) 0x50808188)) != 0x00000000) return CNN_FAIL; // 3,3,22
    if ((*((volatile uint32_t *) 0x5080818c)) != 0x00000000) return CNN_FAIL; // 3,3,23
    if ((*((volatile uint32_t *) 0x50810180)) != 0x00000000) return CNN_FAIL; // 3,3,24
    if ((*((volatile uint32_t *) 0x50810184)) != 0x00000000) return CNN_FAIL; // 3,3,25
    if ((*((volatile uint32_t *) 0x50810188)) != 0x00000000) return CNN_FAIL; // 3,3,26
    if ((*((volatile uint32_t *) 0x5081018c)) != 0x00000000) return CNN_FAIL; // 3,3,27
    if ((*((volatile uint32_t *) 0x50818180)) != 0x00000000) return CNN_FAIL; // 3,3,28
    if ((*((volatile uint32_t *) 0x50818184)) != 0x00000000) return CNN_FAIL; // 3,3,29
    if ((*((volatile uint32_t *) 0x50400190)) != 0x00000000) return CNN_FAIL; // 3,4,0
    if ((*((volatile uint32_t *) 0x50400194)) != 0x00000000) return CNN_FAIL; // 3,4,1
    if ((*((volatile uint32_t *) 0x50400198)) != 0x00000000) return CNN_FAIL; // 3,4,2
    if ((*((volatile uint32_t *) 0x5040019c)) != 0x00000000) return CNN_FAIL; // 3,4,3
    if ((*((volatile uint32_t *) 0x50408190)) != 0x00000000) return CNN_FAIL; // 3,4,4
    if ((*((volatile uint32_t *) 0x50408194)) != 0x00000000) return CNN_FAIL; // 3,4,5
    if ((*((volatile uint32_t *) 0x50408198)) != 0x00000000) return CNN_FAIL; // 3,4,6
    if ((*((volatile uint32_t *) 0x5040819c)) != 0x00000000) return CNN_FAIL; // 3,4,7
    if ((*((volatile uint32_t *) 0x50410190)) != 0x00000000) return CNN_FAIL; // 3,4,8
    if ((*((volatile uint32_t *) 0x50410194)) != 0x00000000) return CNN_FAIL; // 3,4,9
    if ((*((volatile uint32_t *) 0x50410198)) != 0x00000000) return CNN_FAIL; // 3,4,10
    if ((*((volatile uint32_t *) 0x5041019c)) != 0x00000000) return CNN_FAIL; // 3,4,11
    if ((*((volatile uint32_t *) 0x50418190)) != 0x00000000) return CNN_FAIL; // 3,4,12
    if ((*((volatile uint32_t *) 0x50418194)) != 0x00000000) return CNN_FAIL; // 3,4,13
    if ((*((volatile uint32_t *) 0x50418198)) != 0x00000000) return CNN_FAIL; // 3,4,14
    if ((*((volatile uint32_t *) 0x5041819c)) != 0x00000000) return CNN_FAIL; // 3,4,15
    if ((*((volatile uint32_t *) 0x50800190)) != 0x00000000) return CNN_FAIL; // 3,4,16
    if ((*((volatile uint32_t *) 0x50800194)) != 0x00000000) return CNN_FAIL; // 3,4,17
    if ((*((volatile uint32_t *) 0x50800198)) != 0x00000000) return CNN_FAIL; // 3,4,18
    if ((*((volatile uint32_t *) 0x5080019c)) != 0x00000000) return CNN_FAIL; // 3,4,19
    if ((*((volatile uint32_t *) 0x50808190)) != 0x00000000) return CNN_FAIL; // 3,4,20
    if ((*((volatile uint32_t *) 0x50808194)) != 0x00000000) return CNN_FAIL; // 3,4,21
    if ((*((volatile uint32_t *) 0x50808198)) != 0x00000000) return CNN_FAIL; // 3,4,22
    if ((*((volatile uint32_t *) 0x5080819c)) != 0x00000000) return CNN_FAIL; // 3,4,23
    if ((*((volatile uint32_t *) 0x50810190)) != 0x00000000) return CNN_FAIL; // 3,4,24
    if ((*((volatile uint32_t *) 0x50810194)) != 0x00000000) return CNN_FAIL; // 3,4,25
    if ((*((volatile uint32_t *) 0x50810198)) != 0x00000000) return CNN_FAIL; // 3,4,26
    if ((*((volatile uint32_t *) 0x5081019c)) != 0x00000000) return CNN_FAIL; // 3,4,27
    if ((*((volatile uint32_t *) 0x50818190)) != 0x00000000) return CNN_FAIL; // 3,4,28
    if ((*((volatile uint32_t *) 0x50818194)) != 0x00000000) return CNN_FAIL; // 3,4,29
    if ((*((volatile uint32_t *) 0x504001a0)) != 0x00000000) return CNN_FAIL; // 3,5,0
    if ((*((volatile uint32_t *) 0x504001a4)) != 0x00000000) return CNN_FAIL; // 3,5,1
    if ((*((volatile uint32_t *) 0x504001a8)) != 0x00000000) return CNN_FAIL; // 3,5,2
    if ((*((volatile uint32_t *) 0x504001ac)) != 0x00000000) return CNN_FAIL; // 3,5,3
    if ((*((volatile uint32_t *) 0x504081a0)) != 0x00000000) return CNN_FAIL; // 3,5,4
    if ((*((volatile uint32_t *) 0x504081a4)) != 0x00000000) return CNN_FAIL; // 3,5,5
    if ((*((volatile uint32_t *) 0x504081a8)) != 0x00000000) return CNN_FAIL; // 3,5,6
    if ((*((volatile uint32_t *) 0x504081ac)) != 0x00000000) return CNN_FAIL; // 3,5,7
    if ((*((volatile uint32_t *) 0x504101a0)) != 0x00000000) return CNN_FAIL; // 3,5,8
    if ((*((volatile uint32_t *) 0x504101a4)) != 0x00000000) return CNN_FAIL; // 3,5,9
    if ((*((volatile uint32_t *) 0x504101a8)) != 0x00000000) return CNN_FAIL; // 3,5,10
    if ((*((volatile uint32_t *) 0x504101ac)) != 0x00000000) return CNN_FAIL; // 3,5,11
    if ((*((volatile uint32_t *) 0x504181a0)) != 0x00000000) return CNN_FAIL; // 3,5,12
    if ((*((volatile uint32_t *) 0x504181a4)) != 0x00000000) return CNN_FAIL; // 3,5,13
    if ((*((volatile uint32_t *) 0x504181a8)) != 0x00000000) return CNN_FAIL; // 3,5,14
    if ((*((volatile uint32_t *) 0x504181ac)) != 0x00000000) return CNN_FAIL; // 3,5,15
    if ((*((volatile uint32_t *) 0x508001a0)) != 0x00000000) return CNN_FAIL; // 3,5,16
    if ((*((volatile uint32_t *) 0x508001a4)) != 0x00000000) return CNN_FAIL; // 3,5,17
    if ((*((volatile uint32_t *) 0x508001a8)) != 0x00000000) return CNN_FAIL; // 3,5,18
    if ((*((volatile uint32_t *) 0x508001ac)) != 0x00000000) return CNN_FAIL; // 3,5,19
    if ((*((volatile uint32_t *) 0x508081a0)) != 0x00000000) return CNN_FAIL; // 3,5,20
    if ((*((volatile uint32_t *) 0x508081a4)) != 0x00000000) return CNN_FAIL; // 3,5,21
    if ((*((volatile uint32_t *) 0x508081a8)) != 0x00000000) return CNN_FAIL; // 3,5,22
    if ((*((volatile uint32_t *) 0x508081ac)) != 0x00000000) return CNN_FAIL; // 3,5,23
    if ((*((volatile uint32_t *) 0x508101a0)) != 0x00000000) return CNN_FAIL; // 3,5,24
    if ((*((volatile uint32_t *) 0x508101a4)) != 0x00000000) return CNN_FAIL; // 3,5,25
    if ((*((volatile uint32_t *) 0x508101a8)) != 0x00000000) return CNN_FAIL; // 3,5,26
    if ((*((volatile uint32_t *) 0x508101ac)) != 0x00000000) return CNN_FAIL; // 3,5,27
    if ((*((volatile uint32_t *) 0x508181a0)) != 0x00000000) return CNN_FAIL; // 3,5,28
    if ((*((volatile uint32_t *) 0x508181a4)) != 0x00000000) return CNN_FAIL; // 3,5,29
    if ((*((volatile uint32_t *) 0x504001b0)) != 0x00000000) return CNN_FAIL; // 3,6,0
    if ((*((volatile uint32_t *) 0x504001b4)) != 0x00000000) return CNN_FAIL; // 3,6,1
    if ((*((volatile uint32_t *) 0x504001b8)) != 0x00000000) return CNN_FAIL; // 3,6,2
    if ((*((volatile uint32_t *) 0x504001bc)) != 0x00000000) return CNN_FAIL; // 3,6,3
    if ((*((volatile uint32_t *) 0x504081b0)) != 0x00000000) return CNN_FAIL; // 3,6,4
    if ((*((volatile uint32_t *) 0x504081b4)) != 0x00000000) return CNN_FAIL; // 3,6,5
    if ((*((volatile uint32_t *) 0x504081b8)) != 0x00000000) return CNN_FAIL; // 3,6,6
    if ((*((volatile uint32_t *) 0x504081bc)) != 0x00000000) return CNN_FAIL; // 3,6,7
    if ((*((volatile uint32_t *) 0x504101b0)) != 0x00000000) return CNN_FAIL; // 3,6,8
    if ((*((volatile uint32_t *) 0x504101b4)) != 0x00000000) return CNN_FAIL; // 3,6,9
    if ((*((volatile uint32_t *) 0x504101b8)) != 0x00000000) return CNN_FAIL; // 3,6,10
    if ((*((volatile uint32_t *) 0x504101bc)) != 0x00000000) return CNN_FAIL; // 3,6,11
    if ((*((volatile uint32_t *) 0x504181b0)) != 0x00000000) return CNN_FAIL; // 3,6,12
    if ((*((volatile uint32_t *) 0x504181b4)) != 0x00000000) return CNN_FAIL; // 3,6,13
    if ((*((volatile uint32_t *) 0x504181b8)) != 0x00000000) return CNN_FAIL; // 3,6,14
    if ((*((volatile uint32_t *) 0x504181bc)) != 0x00000000) return CNN_FAIL; // 3,6,15
    if ((*((volatile uint32_t *) 0x508001b0)) != 0x00000000) return CNN_FAIL; // 3,6,16
    if ((*((volatile uint32_t *) 0x508001b4)) != 0x00000000) return CNN_FAIL; // 3,6,17
    if ((*((volatile uint32_t *) 0x508001b8)) != 0x00000000) return CNN_FAIL; // 3,6,18
    if ((*((volatile uint32_t *) 0x508001bc)) != 0x00000000) return CNN_FAIL; // 3,6,19
    if ((*((volatile uint32_t *) 0x508081b0)) != 0x00000000) return CNN_FAIL; // 3,6,20
    if ((*((volatile uint32_t *) 0x508081b4)) != 0x00000000) return CNN_FAIL; // 3,6,21
    if ((*((volatile uint32_t *) 0x508081b8)) != 0x00000000) return CNN_FAIL; // 3,6,22
    if ((*((volatile uint32_t *) 0x508081bc)) != 0x00000000) return CNN_FAIL; // 3,6,23
    if ((*((volatile uint32_t *) 0x508101b0)) != 0x00000000) return CNN_FAIL; // 3,6,24
    if ((*((volatile uint32_t *) 0x508101b4)) != 0x00000000) return CNN_FAIL; // 3,6,25
    if ((*((volatile uint32_t *) 0x508101b8)) != 0x00000000) return CNN_FAIL; // 3,6,26
    if ((*((volatile uint32_t *) 0x508101bc)) != 0x00000000) return CNN_FAIL; // 3,6,27
    if ((*((volatile uint32_t *) 0x508181b0)) != 0x00000000) return CNN_FAIL; // 3,6,28
    if ((*((volatile uint32_t *) 0x508181b4)) != 0x00000000) return CNN_FAIL; // 3,6,29
    if ((*((volatile uint32_t *) 0x504001c0)) != 0x00000000) return CNN_FAIL; // 4,0,0
    if ((*((volatile uint32_t *) 0x504001c4)) != 0x00000000) return CNN_FAIL; // 4,0,1
    if ((*((volatile uint32_t *) 0x504001c8)) != 0x00000000) return CNN_FAIL; // 4,0,2
    if ((*((volatile uint32_t *) 0x504001cc)) != 0x00000000) return CNN_FAIL; // 4,0,3
    if ((*((volatile uint32_t *) 0x504081c0)) != 0x00000000) return CNN_FAIL; // 4,0,4
    if ((*((volatile uint32_t *) 0x504081c4)) != 0x00000000) return CNN_FAIL; // 4,0,5
    if ((*((volatile uint32_t *) 0x504081c8)) != 0x00000000) return CNN_FAIL; // 4,0,6
    if ((*((volatile uint32_t *) 0x504081cc)) != 0x00000000) return CNN_FAIL; // 4,0,7
    if ((*((volatile uint32_t *) 0x504101c0)) != 0x00000000) return CNN_FAIL; // 4,0,8
    if ((*((volatile uint32_t *) 0x504101c4)) != 0x00000000) return CNN_FAIL; // 4,0,9
    if ((*((volatile uint32_t *) 0x504101c8)) != 0x00000000) return CNN_FAIL; // 4,0,10
    if ((*((volatile uint32_t *) 0x504101cc)) != 0x00000000) return CNN_FAIL; // 4,0,11
    if ((*((volatile uint32_t *) 0x504181c0)) != 0x00000000) return CNN_FAIL; // 4,0,12
    if ((*((volatile uint32_t *) 0x504181c4)) != 0x00000000) return CNN_FAIL; // 4,0,13
    if ((*((volatile uint32_t *) 0x504181c8)) != 0x00000000) return CNN_FAIL; // 4,0,14
    if ((*((volatile uint32_t *) 0x504181cc)) != 0x00000000) return CNN_FAIL; // 4,0,15
    if ((*((volatile uint32_t *) 0x508001c0)) != 0x00000000) return CNN_FAIL; // 4,0,16
    if ((*((volatile uint32_t *) 0x508001c4)) != 0x00000000) return CNN_FAIL; // 4,0,17
    if ((*((volatile uint32_t *) 0x508001c8)) != 0x00000000) return CNN_FAIL; // 4,0,18
    if ((*((volatile uint32_t *) 0x508001cc)) != 0x00000000) return CNN_FAIL; // 4,0,19
    if ((*((volatile uint32_t *) 0x508081c0)) != 0x00000000) return CNN_FAIL; // 4,0,20
    if ((*((volatile uint32_t *) 0x508081c4)) != 0x00000000) return CNN_FAIL; // 4,0,21
    if ((*((volatile uint32_t *) 0x508081c8)) != 0x00000000) return CNN_FAIL; // 4,0,22
    if ((*((volatile uint32_t *) 0x508081cc)) != 0x00000000) return CNN_FAIL; // 4,0,23
    if ((*((volatile uint32_t *) 0x508101c0)) != 0x00000000) return CNN_FAIL; // 4,0,24
    if ((*((volatile uint32_t *) 0x508101c4)) != 0x00000000) return CNN_FAIL; // 4,0,25
    if ((*((volatile uint32_t *) 0x508101c8)) != 0x00000000) return CNN_FAIL; // 4,0,26
    if ((*((volatile uint32_t *) 0x508101cc)) != 0x00000000) return CNN_FAIL; // 4,0,27
    if ((*((volatile uint32_t *) 0x508181c0)) != 0x00000000) return CNN_FAIL; // 4,0,28
    if ((*((volatile uint32_t *) 0x508181c4)) != 0x00000000) return CNN_FAIL; // 4,0,29
    if ((*((volatile uint32_t *) 0x504001d0)) != 0x00000000) return CNN_FAIL; // 4,1,0
    if ((*((volatile uint32_t *) 0x504001d4)) != 0x00000000) return CNN_FAIL; // 4,1,1
    if ((*((volatile uint32_t *) 0x504001d8)) != 0x00000000) return CNN_FAIL; // 4,1,2
    if ((*((volatile uint32_t *) 0x504001dc)) != 0x00000000) return CNN_FAIL; // 4,1,3
    if ((*((volatile uint32_t *) 0x504081d0)) != 0x00000000) return CNN_FAIL; // 4,1,4
    if ((*((volatile uint32_t *) 0x504081d4)) != 0x00000000) return CNN_FAIL; // 4,1,5
    if ((*((volatile uint32_t *) 0x504081d8)) != 0x00000000) return CNN_FAIL; // 4,1,6
    if ((*((volatile uint32_t *) 0x504081dc)) != 0x00000000) return CNN_FAIL; // 4,1,7
    if ((*((volatile uint32_t *) 0x504101d0)) != 0x00000000) return CNN_FAIL; // 4,1,8
    if ((*((volatile uint32_t *) 0x504101d4)) != 0x00000000) return CNN_FAIL; // 4,1,9
    if ((*((volatile uint32_t *) 0x504101d8)) != 0x00000000) return CNN_FAIL; // 4,1,10
    if ((*((volatile uint32_t *) 0x504101dc)) != 0x00000000) return CNN_FAIL; // 4,1,11
    if ((*((volatile uint32_t *) 0x504181d0)) != 0x00000000) return CNN_FAIL; // 4,1,12
    if ((*((volatile uint32_t *) 0x504181d4)) != 0x00000000) return CNN_FAIL; // 4,1,13
    if ((*((volatile uint32_t *) 0x504181d8)) != 0x00000000) return CNN_FAIL; // 4,1,14
    if ((*((volatile uint32_t *) 0x504181dc)) != 0x00000000) return CNN_FAIL; // 4,1,15
    if ((*((volatile uint32_t *) 0x508001d0)) != 0x00000000) return CNN_FAIL; // 4,1,16
    if ((*((volatile uint32_t *) 0x508001d4)) != 0x00000000) return CNN_FAIL; // 4,1,17
    if ((*((volatile uint32_t *) 0x508001d8)) != 0x00000000) return CNN_FAIL; // 4,1,18
    if ((*((volatile uint32_t *) 0x508001dc)) != 0x00000000) return CNN_FAIL; // 4,1,19
    if ((*((volatile uint32_t *) 0x508081d0)) != 0x00000000) return CNN_FAIL; // 4,1,20
    if ((*((volatile uint32_t *) 0x508081d4)) != 0x00000000) return CNN_FAIL; // 4,1,21
    if ((*((volatile uint32_t *) 0x508081d8)) != 0x00000000) return CNN_FAIL; // 4,1,22
    if ((*((volatile uint32_t *) 0x508081dc)) != 0x00000000) return CNN_FAIL; // 4,1,23
    if ((*((volatile uint32_t *) 0x508101d0)) != 0x00000000) return CNN_FAIL; // 4,1,24
    if ((*((volatile uint32_t *) 0x508101d4)) != 0x00000000) return CNN_FAIL; // 4,1,25
    if ((*((volatile uint32_t *) 0x508101d8)) != 0x00000000) return CNN_FAIL; // 4,1,26
    if ((*((volatile uint32_t *) 0x508101dc)) != 0x00000000) return CNN_FAIL; // 4,1,27
    if ((*((volatile uint32_t *) 0x508181d0)) != 0x00000000) return CNN_FAIL; // 4,1,28
    if ((*((volatile uint32_t *) 0x508181d4)) != 0x00000000) return CNN_FAIL; // 4,1,29
    if ((*((volatile uint32_t *) 0x504001e0)) != 0x00000000) return CNN_FAIL; // 4,2,0
    if ((*((volatile uint32_t *) 0x504001e4)) != 0x00000000) return CNN_FAIL; // 4,2,1
    if ((*((volatile uint32_t *) 0x504001e8)) != 0x00000000) return CNN_FAIL; // 4,2,2
    if ((*((volatile uint32_t *) 0x504001ec)) != 0x00000000) return CNN_FAIL; // 4,2,3
    if ((*((volatile uint32_t *) 0x504081e0)) != 0x00000000) return CNN_FAIL; // 4,2,4
    if ((*((volatile uint32_t *) 0x504081e4)) != 0x00000000) return CNN_FAIL; // 4,2,5
    if ((*((volatile uint32_t *) 0x504081e8)) != 0x00000000) return CNN_FAIL; // 4,2,6
    if ((*((volatile uint32_t *) 0x504081ec)) != 0x00000000) return CNN_FAIL; // 4,2,7
    if ((*((volatile uint32_t *) 0x504101e0)) != 0x00000000) return CNN_FAIL; // 4,2,8
    if ((*((volatile uint32_t *) 0x504101e4)) != 0x00000000) return CNN_FAIL; // 4,2,9
    if ((*((volatile uint32_t *) 0x504101e8)) != 0x00000000) return CNN_FAIL; // 4,2,10
    if ((*((volatile uint32_t *) 0x504101ec)) != 0x00000000) return CNN_FAIL; // 4,2,11
    if ((*((volatile uint32_t *) 0x504181e0)) != 0x00000000) return CNN_FAIL; // 4,2,12
    if ((*((volatile uint32_t *) 0x504181e4)) != 0x00000000) return CNN_FAIL; // 4,2,13
    if ((*((volatile uint32_t *) 0x504181e8)) != 0x00000000) return CNN_FAIL; // 4,2,14
    if ((*((volatile uint32_t *) 0x504181ec)) != 0x00000000) return CNN_FAIL; // 4,2,15
    if ((*((volatile uint32_t *) 0x508001e0)) != 0x00000000) return CNN_FAIL; // 4,2,16
    if ((*((volatile uint32_t *) 0x508001e4)) != 0x00000000) return CNN_FAIL; // 4,2,17
    if ((*((volatile uint32_t *) 0x508001e8)) != 0x00000000) return CNN_FAIL; // 4,2,18
    if ((*((volatile uint32_t *) 0x508001ec)) != 0x00000000) return CNN_FAIL; // 4,2,19
    if ((*((volatile uint32_t *) 0x508081e0)) != 0x00000000) return CNN_FAIL; // 4,2,20
    if ((*((volatile uint32_t *) 0x508081e4)) != 0x00000000) return CNN_FAIL; // 4,2,21
    if ((*((volatile uint32_t *) 0x508081e8)) != 0x00000000) return CNN_FAIL; // 4,2,22
    if ((*((volatile uint32_t *) 0x508081ec)) != 0x00000000) return CNN_FAIL; // 4,2,23
    if ((*((volatile uint32_t *) 0x508101e0)) != 0x00000000) return CNN_FAIL; // 4,2,24
    if ((*((volatile uint32_t *) 0x508101e4)) != 0x00000000) return CNN_FAIL; // 4,2,25
    if ((*((volatile uint32_t *) 0x508101e8)) != 0x00000000) return CNN_FAIL; // 4,2,26
    if ((*((volatile uint32_t *) 0x508101ec)) != 0x00000000) return CNN_FAIL; // 4,2,27
    if ((*((volatile uint32_t *) 0x508181e0)) != 0x00000000) return CNN_FAIL; // 4,2,28
    if ((*((volatile uint32_t *) 0x508181e4)) != 0x00000000) return CNN_FAIL; // 4,2,29
    if ((*((volatile uint32_t *) 0x504001f0)) != 0x00000000) return CNN_FAIL; // 4,3,0
    if ((*((volatile uint32_t *) 0x504001f4)) != 0x00000000) return CNN_FAIL; // 4,3,1
    if ((*((volatile uint32_t *) 0x504001f8)) != 0x00000000) return CNN_FAIL; // 4,3,2
    if ((*((volatile uint32_t *) 0x504001fc)) != 0x00000000) return CNN_FAIL; // 4,3,3
    if ((*((volatile uint32_t *) 0x504081f0)) != 0x00000000) return CNN_FAIL; // 4,3,4
    if ((*((volatile uint32_t *) 0x504081f4)) != 0x00000000) return CNN_FAIL; // 4,3,5
    if ((*((volatile uint32_t *) 0x504081f8)) != 0x00000000) return CNN_FAIL; // 4,3,6
    if ((*((volatile uint32_t *) 0x504081fc)) != 0x00000000) return CNN_FAIL; // 4,3,7
    if ((*((volatile uint32_t *) 0x504101f0)) != 0x00000000) return CNN_FAIL; // 4,3,8
    if ((*((volatile uint32_t *) 0x504101f4)) != 0x00000000) return CNN_FAIL; // 4,3,9
    if ((*((volatile uint32_t *) 0x504101f8)) != 0x00000000) return CNN_FAIL; // 4,3,10
    if ((*((volatile uint32_t *) 0x504101fc)) != 0x00000000) return CNN_FAIL; // 4,3,11
    if ((*((volatile uint32_t *) 0x504181f0)) != 0x00000000) return CNN_FAIL; // 4,3,12
    if ((*((volatile uint32_t *) 0x504181f4)) != 0x00000000) return CNN_FAIL; // 4,3,13
    if ((*((volatile uint32_t *) 0x504181f8)) != 0x00000000) return CNN_FAIL; // 4,3,14
    if ((*((volatile uint32_t *) 0x504181fc)) != 0x00000000) return CNN_FAIL; // 4,3,15
    if ((*((volatile uint32_t *) 0x508001f0)) != 0x00000000) return CNN_FAIL; // 4,3,16
    if ((*((volatile uint32_t *) 0x508001f4)) != 0x00000000) return CNN_FAIL; // 4,3,17
    if ((*((volatile uint32_t *) 0x508001f8)) != 0x00000000) return CNN_FAIL; // 4,3,18
    if ((*((volatile uint32_t *) 0x508001fc)) != 0x00000000) return CNN_FAIL; // 4,3,19
    if ((*((volatile uint32_t *) 0x508081f0)) != 0x00000000) return CNN_FAIL; // 4,3,20
    if ((*((volatile uint32_t *) 0x508081f4)) != 0x00000000) return CNN_FAIL; // 4,3,21
    if ((*((volatile uint32_t *) 0x508081f8)) != 0x00000000) return CNN_FAIL; // 4,3,22
    if ((*((volatile uint32_t *) 0x508081fc)) != 0x00000000) return CNN_FAIL; // 4,3,23
    if ((*((volatile uint32_t *) 0x508101f0)) != 0x00000000) return CNN_FAIL; // 4,3,24
    if ((*((volatile uint32_t *) 0x508101f4)) != 0x00000000) return CNN_FAIL; // 4,3,25
    if ((*((volatile uint32_t *) 0x508101f8)) != 0x00000000) return CNN_FAIL; // 4,3,26
    if ((*((volatile uint32_t *) 0x508101fc)) != 0x00000000) return CNN_FAIL; // 4,3,27
    if ((*((volatile uint32_t *) 0x508181f0)) != 0x00000000) return CNN_FAIL; // 4,3,28
    if ((*((volatile uint32_t *) 0x508181f4)) != 0x00000000) return CNN_FAIL; // 4,3,29
    if ((*((volatile uint32_t *) 0x50400200)) != 0x00000000) return CNN_FAIL; // 4,4,0
    if ((*((volatile uint32_t *) 0x50400204)) != 0x00000000) return CNN_FAIL; // 4,4,1
    if ((*((volatile uint32_t *) 0x50400208)) != 0x00000000) return CNN_FAIL; // 4,4,2
    if ((*((volatile uint32_t *) 0x5040020c)) != 0x00000000) return CNN_FAIL; // 4,4,3
    if ((*((volatile uint32_t *) 0x50408200)) != 0x00000000) return CNN_FAIL; // 4,4,4
    if ((*((volatile uint32_t *) 0x50408204)) != 0x00000000) return CNN_FAIL; // 4,4,5
    if ((*((volatile uint32_t *) 0x50408208)) != 0x00000000) return CNN_FAIL; // 4,4,6
    if ((*((volatile uint32_t *) 0x5040820c)) != 0x00000000) return CNN_FAIL; // 4,4,7
    if ((*((volatile uint32_t *) 0x50410200)) != 0x00000000) return CNN_FAIL; // 4,4,8
    if ((*((volatile uint32_t *) 0x50410204)) != 0x00000000) return CNN_FAIL; // 4,4,9
    if ((*((volatile uint32_t *) 0x50410208)) != 0x00000000) return CNN_FAIL; // 4,4,10
    if ((*((volatile uint32_t *) 0x5041020c)) != 0x00000000) return CNN_FAIL; // 4,4,11
    if ((*((volatile uint32_t *) 0x50418200)) != 0x00000000) return CNN_FAIL; // 4,4,12
    if ((*((volatile uint32_t *) 0x50418204)) != 0x00000000) return CNN_FAIL; // 4,4,13
    if ((*((volatile uint32_t *) 0x50418208)) != 0x00000000) return CNN_FAIL; // 4,4,14
    if ((*((volatile uint32_t *) 0x5041820c)) != 0x00000000) return CNN_FAIL; // 4,4,15
    if ((*((volatile uint32_t *) 0x50800200)) != 0x00000000) return CNN_FAIL; // 4,4,16
    if ((*((volatile uint32_t *) 0x50800204)) != 0x00000000) return CNN_FAIL; // 4,4,17
    if ((*((volatile uint32_t *) 0x50800208)) != 0x00000000) return CNN_FAIL; // 4,4,18
    if ((*((volatile uint32_t *) 0x5080020c)) != 0x00000000) return CNN_FAIL; // 4,4,19
    if ((*((volatile uint32_t *) 0x50808200)) != 0x00000000) return CNN_FAIL; // 4,4,20
    if ((*((volatile uint32_t *) 0x50808204)) != 0x00000000) return CNN_FAIL; // 4,4,21
    if ((*((volatile uint32_t *) 0x50808208)) != 0x00000000) return CNN_FAIL; // 4,4,22
    if ((*((volatile uint32_t *) 0x5080820c)) != 0x00000000) return CNN_FAIL; // 4,4,23
    if ((*((volatile uint32_t *) 0x50810200)) != 0x00000000) return CNN_FAIL; // 4,4,24
    if ((*((volatile uint32_t *) 0x50810204)) != 0x00000000) return CNN_FAIL; // 4,4,25
    if ((*((volatile uint32_t *) 0x50810208)) != 0x00000000) return CNN_FAIL; // 4,4,26
    if ((*((volatile uint32_t *) 0x5081020c)) != 0x00000000) return CNN_FAIL; // 4,4,27
    if ((*((volatile uint32_t *) 0x50818200)) != 0x00000000) return CNN_FAIL; // 4,4,28
    if ((*((volatile uint32_t *) 0x50818204)) != 0x00000000) return CNN_FAIL; // 4,4,29
    if ((*((volatile uint32_t *) 0x50400210)) != 0x00000000) return CNN_FAIL; // 4,5,0
    if ((*((volatile uint32_t *) 0x50400214)) != 0x00000000) return CNN_FAIL; // 4,5,1
    if ((*((volatile uint32_t *) 0x50400218)) != 0x00000000) return CNN_FAIL; // 4,5,2
    if ((*((volatile uint32_t *) 0x5040021c)) != 0x00000000) return CNN_FAIL; // 4,5,3
    if ((*((volatile uint32_t *) 0x50408210)) != 0x00000000) return CNN_FAIL; // 4,5,4
    if ((*((volatile uint32_t *) 0x50408214)) != 0x00000000) return CNN_FAIL; // 4,5,5
    if ((*((volatile uint32_t *) 0x50408218)) != 0x00000000) return CNN_FAIL; // 4,5,6
    if ((*((volatile uint32_t *) 0x5040821c)) != 0x00000000) return CNN_FAIL; // 4,5,7
    if ((*((volatile uint32_t *) 0x50410210)) != 0x00000000) return CNN_FAIL; // 4,5,8
    if ((*((volatile uint32_t *) 0x50410214)) != 0x00000000) return CNN_FAIL; // 4,5,9
    if ((*((volatile uint32_t *) 0x50410218)) != 0x00000000) return CNN_FAIL; // 4,5,10
    if ((*((volatile uint32_t *) 0x5041021c)) != 0x00000000) return CNN_FAIL; // 4,5,11
    if ((*((volatile uint32_t *) 0x50418210)) != 0x00000000) return CNN_FAIL; // 4,5,12
    if ((*((volatile uint32_t *) 0x50418214)) != 0x00000000) return CNN_FAIL; // 4,5,13
    if ((*((volatile uint32_t *) 0x50418218)) != 0x00000000) return CNN_FAIL; // 4,5,14
    if ((*((volatile uint32_t *) 0x5041821c)) != 0x00000000) return CNN_FAIL; // 4,5,15
    if ((*((volatile uint32_t *) 0x50800210)) != 0x00000000) return CNN_FAIL; // 4,5,16
    if ((*((volatile uint32_t *) 0x50800214)) != 0x00000000) return CNN_FAIL; // 4,5,17
    if ((*((volatile uint32_t *) 0x50800218)) != 0x00000000) return CNN_FAIL; // 4,5,18
    if ((*((volatile uint32_t *) 0x5080021c)) != 0x00000000) return CNN_FAIL; // 4,5,19
    if ((*((volatile uint32_t *) 0x50808210)) != 0x00000000) return CNN_FAIL; // 4,5,20
    if ((*((volatile uint32_t *) 0x50808214)) != 0x00000000) return CNN_FAIL; // 4,5,21
    if ((*((volatile uint32_t *) 0x50808218)) != 0x00000000) return CNN_FAIL; // 4,5,22
    if ((*((volatile uint32_t *) 0x5080821c)) != 0x00000000) return CNN_FAIL; // 4,5,23
    if ((*((volatile uint32_t *) 0x50810210)) != 0x00000000) return CNN_FAIL; // 4,5,24
    if ((*((volatile uint32_t *) 0x50810214)) != 0x00000000) return CNN_FAIL; // 4,5,25
    if ((*((volatile uint32_t *) 0x50810218)) != 0x00000000) return CNN_FAIL; // 4,5,26
    if ((*((volatile uint32_t *) 0x5081021c)) != 0x00000000) return CNN_FAIL; // 4,5,27
    if ((*((volatile uint32_t *) 0x50818210)) != 0x00000000) return CNN_FAIL; // 4,5,28
    if ((*((volatile uint32_t *) 0x50818214)) != 0x00000000) return CNN_FAIL; // 4,5,29
    if ((*((volatile uint32_t *) 0x50400220)) != 0x00000000) return CNN_FAIL; // 4,6,0
    if ((*((volatile uint32_t *) 0x50400224)) != 0x00000000) return CNN_FAIL; // 4,6,1
    if ((*((volatile uint32_t *) 0x50400228)) != 0x00000000) return CNN_FAIL; // 4,6,2
    if ((*((volatile uint32_t *) 0x5040022c)) != 0x00000000) return CNN_FAIL; // 4,6,3
    if ((*((volatile uint32_t *) 0x50408220)) != 0x00000000) return CNN_FAIL; // 4,6,4
    if ((*((volatile uint32_t *) 0x50408224)) != 0x00000000) return CNN_FAIL; // 4,6,5
    if ((*((volatile uint32_t *) 0x50408228)) != 0x00000000) return CNN_FAIL; // 4,6,6
    if ((*((volatile uint32_t *) 0x5040822c)) != 0x00000000) return CNN_FAIL; // 4,6,7
    if ((*((volatile uint32_t *) 0x50410220)) != 0x00000000) return CNN_FAIL; // 4,6,8
    if ((*((volatile uint32_t *) 0x50410224)) != 0x00000000) return CNN_FAIL; // 4,6,9
    if ((*((volatile uint32_t *) 0x50410228)) != 0x00000000) return CNN_FAIL; // 4,6,10
    if ((*((volatile uint32_t *) 0x5041022c)) != 0x00000000) return CNN_FAIL; // 4,6,11
    if ((*((volatile uint32_t *) 0x50418220)) != 0x00000000) return CNN_FAIL; // 4,6,12
    if ((*((volatile uint32_t *) 0x50418224)) != 0x00000000) return CNN_FAIL; // 4,6,13
    if ((*((volatile uint32_t *) 0x50418228)) != 0x00000000) return CNN_FAIL; // 4,6,14
    if ((*((volatile uint32_t *) 0x5041822c)) != 0x00000000) return CNN_FAIL; // 4,6,15
    if ((*((volatile uint32_t *) 0x50800220)) != 0x00000000) return CNN_FAIL; // 4,6,16
    if ((*((volatile uint32_t *) 0x50800224)) != 0x00000000) return CNN_FAIL; // 4,6,17
    if ((*((volatile uint32_t *) 0x50800228)) != 0x00000000) return CNN_FAIL; // 4,6,18
    if ((*((volatile uint32_t *) 0x5080022c)) != 0x00000000) return CNN_FAIL; // 4,6,19
    if ((*((volatile uint32_t *) 0x50808220)) != 0x00000000) return CNN_FAIL; // 4,6,20
    if ((*((volatile uint32_t *) 0x50808224)) != 0x00000000) return CNN_FAIL; // 4,6,21
    if ((*((volatile uint32_t *) 0x50808228)) != 0x00000000) return CNN_FAIL; // 4,6,22
    if ((*((volatile uint32_t *) 0x5080822c)) != 0x00000000) return CNN_FAIL; // 4,6,23
    if ((*((volatile uint32_t *) 0x50810220)) != 0x00000000) return CNN_FAIL; // 4,6,24
    if ((*((volatile uint32_t *) 0x50810224)) != 0x00000000) return CNN_FAIL; // 4,6,25
    if ((*((volatile uint32_t *) 0x50810228)) != 0x00000000) return CNN_FAIL; // 4,6,26
    if ((*((volatile uint32_t *) 0x5081022c)) != 0x00000000) return CNN_FAIL; // 4,6,27
    if ((*((volatile uint32_t *) 0x50818220)) != 0x00000000) return CNN_FAIL; // 4,6,28
    if ((*((volatile uint32_t *) 0x50818224)) != 0x00000000) return CNN_FAIL; // 4,6,29
    if ((*((volatile uint32_t *) 0x50400230)) != 0x00000000) return CNN_FAIL; // 5,0,0
    if ((*((volatile uint32_t *) 0x50400234)) != 0x00000000) return CNN_FAIL; // 5,0,1
    if ((*((volatile uint32_t *) 0x50400238)) != 0x00000000) return CNN_FAIL; // 5,0,2
    if ((*((volatile uint32_t *) 0x5040023c)) != 0x00000000) return CNN_FAIL; // 5,0,3
    if ((*((volatile uint32_t *) 0x50408230)) != 0x00000000) return CNN_FAIL; // 5,0,4
    if ((*((volatile uint32_t *) 0x50408234)) != 0x00000000) return CNN_FAIL; // 5,0,5
    if ((*((volatile uint32_t *) 0x50408238)) != 0x00000000) return CNN_FAIL; // 5,0,6
    if ((*((volatile uint32_t *) 0x5040823c)) != 0x00000000) return CNN_FAIL; // 5,0,7
    if ((*((volatile uint32_t *) 0x50410230)) != 0x00000000) return CNN_FAIL; // 5,0,8
    if ((*((volatile uint32_t *) 0x50410234)) != 0x00000000) return CNN_FAIL; // 5,0,9
    if ((*((volatile uint32_t *) 0x50410238)) != 0x00000000) return CNN_FAIL; // 5,0,10
    if ((*((volatile uint32_t *) 0x5041023c)) != 0x00000000) return CNN_FAIL; // 5,0,11
    if ((*((volatile uint32_t *) 0x50418230)) != 0x00000000) return CNN_FAIL; // 5,0,12
    if ((*((volatile uint32_t *) 0x50418234)) != 0x00000000) return CNN_FAIL; // 5,0,13
    if ((*((volatile uint32_t *) 0x50418238)) != 0x00000000) return CNN_FAIL; // 5,0,14
    if ((*((volatile uint32_t *) 0x5041823c)) != 0x00000000) return CNN_FAIL; // 5,0,15
    if ((*((volatile uint32_t *) 0x50800230)) != 0x00000000) return CNN_FAIL; // 5,0,16
    if ((*((volatile uint32_t *) 0x50800234)) != 0x00000000) return CNN_FAIL; // 5,0,17
    if ((*((volatile uint32_t *) 0x50800238)) != 0x00000000) return CNN_FAIL; // 5,0,18
    if ((*((volatile uint32_t *) 0x5080023c)) != 0x00000000) return CNN_FAIL; // 5,0,19
    if ((*((volatile uint32_t *) 0x50808230)) != 0x00000000) return CNN_FAIL; // 5,0,20
    if ((*((volatile uint32_t *) 0x50808234)) != 0x00000000) return CNN_FAIL; // 5,0,21
    if ((*((volatile uint32_t *) 0x50808238)) != 0x00000000) return CNN_FAIL; // 5,0,22
    if ((*((volatile uint32_t *) 0x5080823c)) != 0x00000000) return CNN_FAIL; // 5,0,23
    if ((*((volatile uint32_t *) 0x50810230)) != 0x00000000) return CNN_FAIL; // 5,0,24
    if ((*((volatile uint32_t *) 0x50810234)) != 0x00000000) return CNN_FAIL; // 5,0,25
    if ((*((volatile uint32_t *) 0x50810238)) != 0x00000000) return CNN_FAIL; // 5,0,26
    if ((*((volatile uint32_t *) 0x5081023c)) != 0x00000000) return CNN_FAIL; // 5,0,27
    if ((*((volatile uint32_t *) 0x50818230)) != 0x00000000) return CNN_FAIL; // 5,0,28
    if ((*((volatile uint32_t *) 0x50818234)) != 0x00000000) return CNN_FAIL; // 5,0,29
    if ((*((volatile uint32_t *) 0x50400240)) != 0x00000000) return CNN_FAIL; // 5,1,0
    if ((*((volatile uint32_t *) 0x50400244)) != 0x00000000) return CNN_FAIL; // 5,1,1
    if ((*((volatile uint32_t *) 0x50400248)) != 0x00000000) return CNN_FAIL; // 5,1,2
    if ((*((volatile uint32_t *) 0x5040024c)) != 0x00000000) return CNN_FAIL; // 5,1,3
    if ((*((volatile uint32_t *) 0x50408240)) != 0x00000000) return CNN_FAIL; // 5,1,4
    if ((*((volatile uint32_t *) 0x50408244)) != 0x00000000) return CNN_FAIL; // 5,1,5
    if ((*((volatile uint32_t *) 0x50408248)) != 0x00000000) return CNN_FAIL; // 5,1,6
    if ((*((volatile uint32_t *) 0x5040824c)) != 0x00000000) return CNN_FAIL; // 5,1,7
    if ((*((volatile uint32_t *) 0x50410240)) != 0x00000000) return CNN_FAIL; // 5,1,8
    if ((*((volatile uint32_t *) 0x50410244)) != 0x00000000) return CNN_FAIL; // 5,1,9
    if ((*((volatile uint32_t *) 0x50410248)) != 0x00000000) return CNN_FAIL; // 5,1,10
    if ((*((volatile uint32_t *) 0x5041024c)) != 0x00000000) return CNN_FAIL; // 5,1,11
    if ((*((volatile uint32_t *) 0x50418240)) != 0x00000000) return CNN_FAIL; // 5,1,12
    if ((*((volatile uint32_t *) 0x50418244)) != 0x00000000) return CNN_FAIL; // 5,1,13
    if ((*((volatile uint32_t *) 0x50418248)) != 0x00000000) return CNN_FAIL; // 5,1,14
    if ((*((volatile uint32_t *) 0x5041824c)) != 0x00000000) return CNN_FAIL; // 5,1,15
    if ((*((volatile uint32_t *) 0x50800240)) != 0x00000000) return CNN_FAIL; // 5,1,16
    if ((*((volatile uint32_t *) 0x50800244)) != 0x00000000) return CNN_FAIL; // 5,1,17
    if ((*((volatile uint32_t *) 0x50800248)) != 0x00000000) return CNN_FAIL; // 5,1,18
    if ((*((volatile uint32_t *) 0x5080024c)) != 0x00000000) return CNN_FAIL; // 5,1,19
    if ((*((volatile uint32_t *) 0x50808240)) != 0x00000000) return CNN_FAIL; // 5,1,20
    if ((*((volatile uint32_t *) 0x50808244)) != 0x00000000) return CNN_FAIL; // 5,1,21
    if ((*((volatile uint32_t *) 0x50808248)) != 0x00000000) return CNN_FAIL; // 5,1,22
    if ((*((volatile uint32_t *) 0x5080824c)) != 0x00000000) return CNN_FAIL; // 5,1,23
    if ((*((volatile uint32_t *) 0x50810240)) != 0x00000000) return CNN_FAIL; // 5,1,24
    if ((*((volatile uint32_t *) 0x50810244)) != 0x00000000) return CNN_FAIL; // 5,1,25
    if ((*((volatile uint32_t *) 0x50810248)) != 0x00000000) return CNN_FAIL; // 5,1,26
    if ((*((volatile uint32_t *) 0x5081024c)) != 0x00000000) return CNN_FAIL; // 5,1,27
    if ((*((volatile uint32_t *) 0x50818240)) != 0x00000000) return CNN_FAIL; // 5,1,28
    if ((*((volatile uint32_t *) 0x50818244)) != 0x00000000) return CNN_FAIL; // 5,1,29
    if ((*((volatile uint32_t *) 0x50400250)) != 0x00000000) return CNN_FAIL; // 5,2,0
    if ((*((volatile uint32_t *) 0x50400254)) != 0x00000000) return CNN_FAIL; // 5,2,1
    if ((*((volatile uint32_t *) 0x50400258)) != 0x00000000) return CNN_FAIL; // 5,2,2
    if ((*((volatile uint32_t *) 0x5040025c)) != 0x00000000) return CNN_FAIL; // 5,2,3
    if ((*((volatile uint32_t *) 0x50408250)) != 0x00000000) return CNN_FAIL; // 5,2,4
    if ((*((volatile uint32_t *) 0x50408254)) != 0x00000000) return CNN_FAIL; // 5,2,5
    if ((*((volatile uint32_t *) 0x50408258)) != 0x00000000) return CNN_FAIL; // 5,2,6
    if ((*((volatile uint32_t *) 0x5040825c)) != 0x00000000) return CNN_FAIL; // 5,2,7
    if ((*((volatile uint32_t *) 0x50410250)) != 0x00000000) return CNN_FAIL; // 5,2,8
    if ((*((volatile uint32_t *) 0x50410254)) != 0x00000000) return CNN_FAIL; // 5,2,9
    if ((*((volatile uint32_t *) 0x50410258)) != 0x00000000) return CNN_FAIL; // 5,2,10
    if ((*((volatile uint32_t *) 0x5041025c)) != 0x00000000) return CNN_FAIL; // 5,2,11
    if ((*((volatile uint32_t *) 0x50418250)) != 0x00000000) return CNN_FAIL; // 5,2,12
    if ((*((volatile uint32_t *) 0x50418254)) != 0x00000000) return CNN_FAIL; // 5,2,13
    if ((*((volatile uint32_t *) 0x50418258)) != 0x00000000) return CNN_FAIL; // 5,2,14
    if ((*((volatile uint32_t *) 0x5041825c)) != 0x00000000) return CNN_FAIL; // 5,2,15
    if ((*((volatile uint32_t *) 0x50800250)) != 0x00000000) return CNN_FAIL; // 5,2,16
    if ((*((volatile uint32_t *) 0x50800254)) != 0x00000000) return CNN_FAIL; // 5,2,17
    if ((*((volatile uint32_t *) 0x50800258)) != 0x00000000) return CNN_FAIL; // 5,2,18
    if ((*((volatile uint32_t *) 0x5080025c)) != 0x00000000) return CNN_FAIL; // 5,2,19
    if ((*((volatile uint32_t *) 0x50808250)) != 0x00000000) return CNN_FAIL; // 5,2,20
    if ((*((volatile uint32_t *) 0x50808254)) != 0x00000000) return CNN_FAIL; // 5,2,21
    if ((*((volatile uint32_t *) 0x50808258)) != 0x00000000) return CNN_FAIL; // 5,2,22
    if ((*((volatile uint32_t *) 0x5080825c)) != 0x00000000) return CNN_FAIL; // 5,2,23
    if ((*((volatile uint32_t *) 0x50810250)) != 0x00000000) return CNN_FAIL; // 5,2,24
    if ((*((volatile uint32_t *) 0x50810254)) != 0x00000000) return CNN_FAIL; // 5,2,25
    if ((*((volatile uint32_t *) 0x50810258)) != 0x00000000) return CNN_FAIL; // 5,2,26
    if ((*((volatile uint32_t *) 0x5081025c)) != 0x00000000) return CNN_FAIL; // 5,2,27
    if ((*((volatile uint32_t *) 0x50818250)) != 0x00000000) return CNN_FAIL; // 5,2,28
    if ((*((volatile uint32_t *) 0x50818254)) != 0x00000000) return CNN_FAIL; // 5,2,29
    if ((*((volatile uint32_t *) 0x50400260)) != 0x00000000) return CNN_FAIL; // 5,3,0
    if ((*((volatile uint32_t *) 0x50400264)) != 0x00000000) return CNN_FAIL; // 5,3,1
    if ((*((volatile uint32_t *) 0x50400268)) != 0x00000000) return CNN_FAIL; // 5,3,2
    if ((*((volatile uint32_t *) 0x5040026c)) != 0x00000000) return CNN_FAIL; // 5,3,3
    if ((*((volatile uint32_t *) 0x50408260)) != 0x00000000) return CNN_FAIL; // 5,3,4
    if ((*((volatile uint32_t *) 0x50408264)) != 0x00000000) return CNN_FAIL; // 5,3,5
    if ((*((volatile uint32_t *) 0x50408268)) != 0x00000000) return CNN_FAIL; // 5,3,6
    if ((*((volatile uint32_t *) 0x5040826c)) != 0x00000000) return CNN_FAIL; // 5,3,7
    if ((*((volatile uint32_t *) 0x50410260)) != 0x00000000) return CNN_FAIL; // 5,3,8
    if ((*((volatile uint32_t *) 0x50410264)) != 0x00000000) return CNN_FAIL; // 5,3,9
    if ((*((volatile uint32_t *) 0x50410268)) != 0x00000000) return CNN_FAIL; // 5,3,10
    if ((*((volatile uint32_t *) 0x5041026c)) != 0x00000000) return CNN_FAIL; // 5,3,11
    if ((*((volatile uint32_t *) 0x50418260)) != 0x00000000) return CNN_FAIL; // 5,3,12
    if ((*((volatile uint32_t *) 0x50418264)) != 0x00000000) return CNN_FAIL; // 5,3,13
    if ((*((volatile uint32_t *) 0x50418268)) != 0x00000000) return CNN_FAIL; // 5,3,14
    if ((*((volatile uint32_t *) 0x5041826c)) != 0x00000000) return CNN_FAIL; // 5,3,15
    if ((*((volatile uint32_t *) 0x50800260)) != 0x00000000) return CNN_FAIL; // 5,3,16
    if ((*((volatile uint32_t *) 0x50800264)) != 0x00000000) return CNN_FAIL; // 5,3,17
    if ((*((volatile uint32_t *) 0x50800268)) != 0x00000000) return CNN_FAIL; // 5,3,18
    if ((*((volatile uint32_t *) 0x5080026c)) != 0x00000000) return CNN_FAIL; // 5,3,19
    if ((*((volatile uint32_t *) 0x50808260)) != 0x00000000) return CNN_FAIL; // 5,3,20
    if ((*((volatile uint32_t *) 0x50808264)) != 0x00000000) return CNN_FAIL; // 5,3,21
    if ((*((volatile uint32_t *) 0x50808268)) != 0x00000000) return CNN_FAIL; // 5,3,22
    if ((*((volatile uint32_t *) 0x5080826c)) != 0x00000000) return CNN_FAIL; // 5,3,23
    if ((*((volatile uint32_t *) 0x50810260)) != 0x00000000) return CNN_FAIL; // 5,3,24
    if ((*((volatile uint32_t *) 0x50810264)) != 0x00000000) return CNN_FAIL; // 5,3,25
    if ((*((volatile uint32_t *) 0x50810268)) != 0x00000000) return CNN_FAIL; // 5,3,26
    if ((*((volatile uint32_t *) 0x5081026c)) != 0x00000000) return CNN_FAIL; // 5,3,27
    if ((*((volatile uint32_t *) 0x50818260)) != 0x00000000) return CNN_FAIL; // 5,3,28
    if ((*((volatile uint32_t *) 0x50818264)) != 0x00000000) return CNN_FAIL; // 5,3,29
    if ((*((volatile uint32_t *) 0x50400270)) != 0x00000000) return CNN_FAIL; // 5,4,0
    if ((*((volatile uint32_t *) 0x50400274)) != 0x00000000) return CNN_FAIL; // 5,4,1
    if ((*((volatile uint32_t *) 0x50400278)) != 0x00000000) return CNN_FAIL; // 5,4,2
    if ((*((volatile uint32_t *) 0x5040027c)) != 0x00000000) return CNN_FAIL; // 5,4,3
    if ((*((volatile uint32_t *) 0x50408270)) != 0x00000000) return CNN_FAIL; // 5,4,4
    if ((*((volatile uint32_t *) 0x50408274)) != 0x00000000) return CNN_FAIL; // 5,4,5
    if ((*((volatile uint32_t *) 0x50408278)) != 0x00000000) return CNN_FAIL; // 5,4,6
    if ((*((volatile uint32_t *) 0x5040827c)) != 0x00000000) return CNN_FAIL; // 5,4,7
    if ((*((volatile uint32_t *) 0x50410270)) != 0x00000000) return CNN_FAIL; // 5,4,8
    if ((*((volatile uint32_t *) 0x50410274)) != 0x00000000) return CNN_FAIL; // 5,4,9
    if ((*((volatile uint32_t *) 0x50410278)) != 0x00000000) return CNN_FAIL; // 5,4,10
    if ((*((volatile uint32_t *) 0x5041027c)) != 0x00000000) return CNN_FAIL; // 5,4,11
    if ((*((volatile uint32_t *) 0x50418270)) != 0x00000000) return CNN_FAIL; // 5,4,12
    if ((*((volatile uint32_t *) 0x50418274)) != 0x00000000) return CNN_FAIL; // 5,4,13
    if ((*((volatile uint32_t *) 0x50418278)) != 0x00000000) return CNN_FAIL; // 5,4,14
    if ((*((volatile uint32_t *) 0x5041827c)) != 0x00000000) return CNN_FAIL; // 5,4,15
    if ((*((volatile uint32_t *) 0x50800270)) != 0x00000000) return CNN_FAIL; // 5,4,16
    if ((*((volatile uint32_t *) 0x50800274)) != 0x00000000) return CNN_FAIL; // 5,4,17
    if ((*((volatile uint32_t *) 0x50800278)) != 0x00000000) return CNN_FAIL; // 5,4,18
    if ((*((volatile uint32_t *) 0x5080027c)) != 0x00000000) return CNN_FAIL; // 5,4,19
    if ((*((volatile uint32_t *) 0x50808270)) != 0x00000000) return CNN_FAIL; // 5,4,20
    if ((*((volatile uint32_t *) 0x50808274)) != 0x00000000) return CNN_FAIL; // 5,4,21
    if ((*((volatile uint32_t *) 0x50808278)) != 0x00000000) return CNN_FAIL; // 5,4,22
    if ((*((volatile uint32_t *) 0x5080827c)) != 0x00000000) return CNN_FAIL; // 5,4,23
    if ((*((volatile uint32_t *) 0x50810270)) != 0x00000000) return CNN_FAIL; // 5,4,24
    if ((*((volatile uint32_t *) 0x50810274)) != 0x00000000) return CNN_FAIL; // 5,4,25
    if ((*((volatile uint32_t *) 0x50810278)) != 0x00000000) return CNN_FAIL; // 5,4,26
    if ((*((volatile uint32_t *) 0x5081027c)) != 0x00000000) return CNN_FAIL; // 5,4,27
    if ((*((volatile uint32_t *) 0x50818270)) != 0x00000000) return CNN_FAIL; // 5,4,28
    if ((*((volatile uint32_t *) 0x50818274)) != 0x00000000) return CNN_FAIL; // 5,4,29
    if ((*((volatile uint32_t *) 0x50400280)) != 0x00000000) return CNN_FAIL; // 5,5,0
    if ((*((volatile uint32_t *) 0x50400284)) != 0x00000000) return CNN_FAIL; // 5,5,1
    if ((*((volatile uint32_t *) 0x50400288)) != 0x00000000) return CNN_FAIL; // 5,5,2
    if ((*((volatile uint32_t *) 0x5040028c)) != 0x00000000) return CNN_FAIL; // 5,5,3
    if ((*((volatile uint32_t *) 0x50408280)) != 0x00000000) return CNN_FAIL; // 5,5,4
    if ((*((volatile uint32_t *) 0x50408284)) != 0x00000000) return CNN_FAIL; // 5,5,5
    if ((*((volatile uint32_t *) 0x50408288)) != 0x00000000) return CNN_FAIL; // 5,5,6
    if ((*((volatile uint32_t *) 0x5040828c)) != 0x00000000) return CNN_FAIL; // 5,5,7
    if ((*((volatile uint32_t *) 0x50410280)) != 0x00000000) return CNN_FAIL; // 5,5,8
    if ((*((volatile uint32_t *) 0x50410284)) != 0x00000000) return CNN_FAIL; // 5,5,9
    if ((*((volatile uint32_t *) 0x50410288)) != 0x00000000) return CNN_FAIL; // 5,5,10
    if ((*((volatile uint32_t *) 0x5041028c)) != 0x00000000) return CNN_FAIL; // 5,5,11
    if ((*((volatile uint32_t *) 0x50418280)) != 0x00000000) return CNN_FAIL; // 5,5,12
    if ((*((volatile uint32_t *) 0x50418284)) != 0x00000000) return CNN_FAIL; // 5,5,13
    if ((*((volatile uint32_t *) 0x50418288)) != 0x00000000) return CNN_FAIL; // 5,5,14
    if ((*((volatile uint32_t *) 0x5041828c)) != 0x00000000) return CNN_FAIL; // 5,5,15
    if ((*((volatile uint32_t *) 0x50800280)) != 0x00000000) return CNN_FAIL; // 5,5,16
    if ((*((volatile uint32_t *) 0x50800284)) != 0x00000000) return CNN_FAIL; // 5,5,17
    if ((*((volatile uint32_t *) 0x50800288)) != 0x00000000) return CNN_FAIL; // 5,5,18
    if ((*((volatile uint32_t *) 0x5080028c)) != 0x00000000) return CNN_FAIL; // 5,5,19
    if ((*((volatile uint32_t *) 0x50808280)) != 0x00000000) return CNN_FAIL; // 5,5,20
    if ((*((volatile uint32_t *) 0x50808284)) != 0x00000000) return CNN_FAIL; // 5,5,21
    if ((*((volatile uint32_t *) 0x50808288)) != 0x00000000) return CNN_FAIL; // 5,5,22
    if ((*((volatile uint32_t *) 0x5080828c)) != 0x00000000) return CNN_FAIL; // 5,5,23
    if ((*((volatile uint32_t *) 0x50810280)) != 0x00000000) return CNN_FAIL; // 5,5,24
    if ((*((volatile uint32_t *) 0x50810284)) != 0x00000000) return CNN_FAIL; // 5,5,25
    if ((*((volatile uint32_t *) 0x50810288)) != 0x00000000) return CNN_FAIL; // 5,5,26
    if ((*((volatile uint32_t *) 0x5081028c)) != 0x00000000) return CNN_FAIL; // 5,5,27
    if ((*((volatile uint32_t *) 0x50818280)) != 0x00000000) return CNN_FAIL; // 5,5,28
    if ((*((volatile uint32_t *) 0x50818284)) != 0x00000000) return CNN_FAIL; // 5,5,29
    if ((*((volatile uint32_t *) 0x50400290)) != 0x00000000) return CNN_FAIL; // 5,6,0
    if ((*((volatile uint32_t *) 0x50400294)) != 0x00000000) return CNN_FAIL; // 5,6,1
    if ((*((volatile uint32_t *) 0x50400298)) != 0x00000000) return CNN_FAIL; // 5,6,2
    if ((*((volatile uint32_t *) 0x5040029c)) != 0x00000000) return CNN_FAIL; // 5,6,3
    if ((*((volatile uint32_t *) 0x50408290)) != 0x00000000) return CNN_FAIL; // 5,6,4
    if ((*((volatile uint32_t *) 0x50408294)) != 0x00000000) return CNN_FAIL; // 5,6,5
    if ((*((volatile uint32_t *) 0x50408298)) != 0x00000000) return CNN_FAIL; // 5,6,6
    if ((*((volatile uint32_t *) 0x5040829c)) != 0x00000000) return CNN_FAIL; // 5,6,7
    if ((*((volatile uint32_t *) 0x50410290)) != 0x00000000) return CNN_FAIL; // 5,6,8
    if ((*((volatile uint32_t *) 0x50410294)) != 0x00000000) return CNN_FAIL; // 5,6,9
    if ((*((volatile uint32_t *) 0x50410298)) != 0x00000000) return CNN_FAIL; // 5,6,10
    if ((*((volatile uint32_t *) 0x5041029c)) != 0x00000000) return CNN_FAIL; // 5,6,11
    if ((*((volatile uint32_t *) 0x50418290)) != 0x00000000) return CNN_FAIL; // 5,6,12
    if ((*((volatile uint32_t *) 0x50418294)) != 0x00000000) return CNN_FAIL; // 5,6,13
    if ((*((volatile uint32_t *) 0x50418298)) != 0x00000000) return CNN_FAIL; // 5,6,14
    if ((*((volatile uint32_t *) 0x5041829c)) != 0x00000000) return CNN_FAIL; // 5,6,15
    if ((*((volatile uint32_t *) 0x50800290)) != 0x00000000) return CNN_FAIL; // 5,6,16
    if ((*((volatile uint32_t *) 0x50800294)) != 0x00000000) return CNN_FAIL; // 5,6,17
    if ((*((volatile uint32_t *) 0x50800298)) != 0x00000000) return CNN_FAIL; // 5,6,18
    if ((*((volatile uint32_t *) 0x5080029c)) != 0x00000000) return CNN_FAIL; // 5,6,19
    if ((*((volatile uint32_t *) 0x50808290)) != 0x00000000) return CNN_FAIL; // 5,6,20
    if ((*((volatile uint32_t *) 0x50808294)) != 0x00000000) return CNN_FAIL; // 5,6,21
    if ((*((volatile uint32_t *) 0x50808298)) != 0x00000000) return CNN_FAIL; // 5,6,22
    if ((*((volatile uint32_t *) 0x5080829c)) != 0x00000000) return CNN_FAIL; // 5,6,23
    if ((*((volatile uint32_t *) 0x50810290)) != 0x00000000) return CNN_FAIL; // 5,6,24
    if ((*((volatile uint32_t *) 0x50810294)) != 0x00000000) return CNN_FAIL; // 5,6,25
    if ((*((volatile uint32_t *) 0x50810298)) != 0x00000000) return CNN_FAIL; // 5,6,26
    if ((*((volatile uint32_t *) 0x5081029c)) != 0x00000000) return CNN_FAIL; // 5,6,27
    if ((*((volatile uint32_t *) 0x50818290)) != 0x00000000) return CNN_FAIL; // 5,6,28
    if ((*((volatile uint32_t *) 0x50818294)) != 0x00000000) return CNN_FAIL; // 5,6,29
    if ((*((volatile uint32_t *) 0x504002a0)) != 0x00000000) return CNN_FAIL; // 6,0,0
    if ((*((volatile uint32_t *) 0x504002a4)) != 0x00000000) return CNN_FAIL; // 6,0,1
    if ((*((volatile uint32_t *) 0x504002a8)) != 0x00000000) return CNN_FAIL; // 6,0,2
    if ((*((volatile uint32_t *) 0x504002ac)) != 0x00000000) return CNN_FAIL; // 6,0,3
    if ((*((volatile uint32_t *) 0x504082a0)) != 0x00000000) return CNN_FAIL; // 6,0,4
    if ((*((volatile uint32_t *) 0x504082a4)) != 0x00000000) return CNN_FAIL; // 6,0,5
    if ((*((volatile uint32_t *) 0x504082a8)) != 0x00000000) return CNN_FAIL; // 6,0,6
    if ((*((volatile uint32_t *) 0x504082ac)) != 0x00000000) return CNN_FAIL; // 6,0,7
    if ((*((volatile uint32_t *) 0x504102a0)) != 0x00000000) return CNN_FAIL; // 6,0,8
    if ((*((volatile uint32_t *) 0x504102a4)) != 0x00000000) return CNN_FAIL; // 6,0,9
    if ((*((volatile uint32_t *) 0x504102a8)) != 0x00000000) return CNN_FAIL; // 6,0,10
    if ((*((volatile uint32_t *) 0x504102ac)) != 0x00000000) return CNN_FAIL; // 6,0,11
    if ((*((volatile uint32_t *) 0x504182a0)) != 0x00000000) return CNN_FAIL; // 6,0,12
    if ((*((volatile uint32_t *) 0x504182a4)) != 0x00000000) return CNN_FAIL; // 6,0,13
    if ((*((volatile uint32_t *) 0x504182a8)) != 0x00000000) return CNN_FAIL; // 6,0,14
    if ((*((volatile uint32_t *) 0x504182ac)) != 0x00000000) return CNN_FAIL; // 6,0,15
    if ((*((volatile uint32_t *) 0x508002a0)) != 0x00000000) return CNN_FAIL; // 6,0,16
    if ((*((volatile uint32_t *) 0x508002a4)) != 0x00000000) return CNN_FAIL; // 6,0,17
    if ((*((volatile uint32_t *) 0x508002a8)) != 0x00000000) return CNN_FAIL; // 6,0,18
    if ((*((volatile uint32_t *) 0x508002ac)) != 0x00000000) return CNN_FAIL; // 6,0,19
    if ((*((volatile uint32_t *) 0x508082a0)) != 0x00000000) return CNN_FAIL; // 6,0,20
    if ((*((volatile uint32_t *) 0x508082a4)) != 0x00000000) return CNN_FAIL; // 6,0,21
    if ((*((volatile uint32_t *) 0x508082a8)) != 0x00000000) return CNN_FAIL; // 6,0,22
    if ((*((volatile uint32_t *) 0x508082ac)) != 0x00000000) return CNN_FAIL; // 6,0,23
    if ((*((volatile uint32_t *) 0x508102a0)) != 0x00000000) return CNN_FAIL; // 6,0,24
    if ((*((volatile uint32_t *) 0x508102a4)) != 0x00000000) return CNN_FAIL; // 6,0,25
    if ((*((volatile uint32_t *) 0x508102a8)) != 0x00000000) return CNN_FAIL; // 6,0,26
    if ((*((volatile uint32_t *) 0x508102ac)) != 0x00000000) return CNN_FAIL; // 6,0,27
    if ((*((volatile uint32_t *) 0x508182a0)) != 0x00000000) return CNN_FAIL; // 6,0,28
    if ((*((volatile uint32_t *) 0x508182a4)) != 0x00000000) return CNN_FAIL; // 6,0,29
    if ((*((volatile uint32_t *) 0x504002b0)) != 0x00000000) return CNN_FAIL; // 6,1,0
    if ((*((volatile uint32_t *) 0x504002b4)) != 0x00000000) return CNN_FAIL; // 6,1,1
    if ((*((volatile uint32_t *) 0x504002b8)) != 0x00000000) return CNN_FAIL; // 6,1,2
    if ((*((volatile uint32_t *) 0x504002bc)) != 0x00000000) return CNN_FAIL; // 6,1,3
    if ((*((volatile uint32_t *) 0x504082b0)) != 0x00000000) return CNN_FAIL; // 6,1,4
    if ((*((volatile uint32_t *) 0x504082b4)) != 0x00000000) return CNN_FAIL; // 6,1,5
    if ((*((volatile uint32_t *) 0x504082b8)) != 0x00000000) return CNN_FAIL; // 6,1,6
    if ((*((volatile uint32_t *) 0x504082bc)) != 0x00000000) return CNN_FAIL; // 6,1,7
    if ((*((volatile uint32_t *) 0x504102b0)) != 0x00000000) return CNN_FAIL; // 6,1,8
    if ((*((volatile uint32_t *) 0x504102b4)) != 0x00000000) return CNN_FAIL; // 6,1,9
    if ((*((volatile uint32_t *) 0x504102b8)) != 0x00000000) return CNN_FAIL; // 6,1,10
    if ((*((volatile uint32_t *) 0x504102bc)) != 0x00000000) return CNN_FAIL; // 6,1,11
    if ((*((volatile uint32_t *) 0x504182b0)) != 0x00000000) return CNN_FAIL; // 6,1,12
    if ((*((volatile uint32_t *) 0x504182b4)) != 0x00000000) return CNN_FAIL; // 6,1,13
    if ((*((volatile uint32_t *) 0x504182b8)) != 0x00000000) return CNN_FAIL; // 6,1,14
    if ((*((volatile uint32_t *) 0x504182bc)) != 0x00000000) return CNN_FAIL; // 6,1,15
    if ((*((volatile uint32_t *) 0x508002b0)) != 0x00000000) return CNN_FAIL; // 6,1,16
    if ((*((volatile uint32_t *) 0x508002b4)) != 0x00000000) return CNN_FAIL; // 6,1,17
    if ((*((volatile uint32_t *) 0x508002b8)) != 0x00000000) return CNN_FAIL; // 6,1,18
    if ((*((volatile uint32_t *) 0x508002bc)) != 0x00000000) return CNN_FAIL; // 6,1,19
    if ((*((volatile uint32_t *) 0x508082b0)) != 0x00000000) return CNN_FAIL; // 6,1,20
    if ((*((volatile uint32_t *) 0x508082b4)) != 0x00000000) return CNN_FAIL; // 6,1,21
    if ((*((volatile uint32_t *) 0x508082b8)) != 0x00000000) return CNN_FAIL; // 6,1,22
    if ((*((volatile uint32_t *) 0x508082bc)) != 0x00000000) return CNN_FAIL; // 6,1,23
    if ((*((volatile uint32_t *) 0x508102b0)) != 0x00000000) return CNN_FAIL; // 6,1,24
    if ((*((volatile uint32_t *) 0x508102b4)) != 0x00000000) return CNN_FAIL; // 6,1,25
    if ((*((volatile uint32_t *) 0x508102b8)) != 0x00000000) return CNN_FAIL; // 6,1,26
    if ((*((volatile uint32_t *) 0x508102bc)) != 0x00000000) return CNN_FAIL; // 6,1,27
    if ((*((volatile uint32_t *) 0x508182b0)) != 0x00000000) return CNN_FAIL; // 6,1,28
    if ((*((volatile uint32_t *) 0x508182b4)) != 0x00000000) return CNN_FAIL; // 6,1,29
    if ((*((volatile uint32_t *) 0x504002c0)) != 0x00000000) return CNN_FAIL; // 6,2,0
    if ((*((volatile uint32_t *) 0x504002c4)) != 0x00000000) return CNN_FAIL; // 6,2,1
    if ((*((volatile uint32_t *) 0x504002c8)) != 0x00000000) return CNN_FAIL; // 6,2,2
    if ((*((volatile uint32_t *) 0x504002cc)) != 0x00000000) return CNN_FAIL; // 6,2,3
    if ((*((volatile uint32_t *) 0x504082c0)) != 0x00000000) return CNN_FAIL; // 6,2,4
    if ((*((volatile uint32_t *) 0x504082c4)) != 0x00000000) return CNN_FAIL; // 6,2,5
    if ((*((volatile uint32_t *) 0x504082c8)) != 0x00000000) return CNN_FAIL; // 6,2,6
    if ((*((volatile uint32_t *) 0x504082cc)) != 0x00000000) return CNN_FAIL; // 6,2,7
    if ((*((volatile uint32_t *) 0x504102c0)) != 0x00000000) return CNN_FAIL; // 6,2,8
    if ((*((volatile uint32_t *) 0x504102c4)) != 0x00000000) return CNN_FAIL; // 6,2,9
    if ((*((volatile uint32_t *) 0x504102c8)) != 0x00000000) return CNN_FAIL; // 6,2,10
    if ((*((volatile uint32_t *) 0x504102cc)) != 0x00000000) return CNN_FAIL; // 6,2,11
    if ((*((volatile uint32_t *) 0x504182c0)) != 0x00000000) return CNN_FAIL; // 6,2,12
    if ((*((volatile uint32_t *) 0x504182c4)) != 0x00000000) return CNN_FAIL; // 6,2,13
    if ((*((volatile uint32_t *) 0x504182c8)) != 0x00000000) return CNN_FAIL; // 6,2,14
    if ((*((volatile uint32_t *) 0x504182cc)) != 0x00000000) return CNN_FAIL; // 6,2,15
    if ((*((volatile uint32_t *) 0x508002c0)) != 0x00000000) return CNN_FAIL; // 6,2,16
    if ((*((volatile uint32_t *) 0x508002c4)) != 0x00000000) return CNN_FAIL; // 6,2,17
    if ((*((volatile uint32_t *) 0x508002c8)) != 0x00000000) return CNN_FAIL; // 6,2,18
    if ((*((volatile uint32_t *) 0x508002cc)) != 0x00000000) return CNN_FAIL; // 6,2,19
    if ((*((volatile uint32_t *) 0x508082c0)) != 0x00000000) return CNN_FAIL; // 6,2,20
    if ((*((volatile uint32_t *) 0x508082c4)) != 0x00000000) return CNN_FAIL; // 6,2,21
    if ((*((volatile uint32_t *) 0x508082c8)) != 0x00000000) return CNN_FAIL; // 6,2,22
    if ((*((volatile uint32_t *) 0x508082cc)) != 0x00000000) return CNN_FAIL; // 6,2,23
    if ((*((volatile uint32_t *) 0x508102c0)) != 0x00000000) return CNN_FAIL; // 6,2,24
    if ((*((volatile uint32_t *) 0x508102c4)) != 0x00000000) return CNN_FAIL; // 6,2,25
    if ((*((volatile uint32_t *) 0x508102c8)) != 0x00000000) return CNN_FAIL; // 6,2,26
    if ((*((volatile uint32_t *) 0x508102cc)) != 0x00000000) return CNN_FAIL; // 6,2,27
    if ((*((volatile uint32_t *) 0x508182c0)) != 0x00000000) return CNN_FAIL; // 6,2,28
    if ((*((volatile uint32_t *) 0x508182c4)) != 0x00000000) return CNN_FAIL; // 6,2,29
    if ((*((volatile uint32_t *) 0x504002d0)) != 0x00000000) return CNN_FAIL; // 6,3,0
    if ((*((volatile uint32_t *) 0x504002d4)) != 0x00000000) return CNN_FAIL; // 6,3,1
    if ((*((volatile uint32_t *) 0x504002d8)) != 0x00000000) return CNN_FAIL; // 6,3,2
    if ((*((volatile uint32_t *) 0x504002dc)) != 0x00000000) return CNN_FAIL; // 6,3,3
    if ((*((volatile uint32_t *) 0x504082d0)) != 0x00000000) return CNN_FAIL; // 6,3,4
    if ((*((volatile uint32_t *) 0x504082d4)) != 0x00000000) return CNN_FAIL; // 6,3,5
    if ((*((volatile uint32_t *) 0x504082d8)) != 0x00000000) return CNN_FAIL; // 6,3,6
    if ((*((volatile uint32_t *) 0x504082dc)) != 0x00000000) return CNN_FAIL; // 6,3,7
    if ((*((volatile uint32_t *) 0x504102d0)) != 0x00000000) return CNN_FAIL; // 6,3,8
    if ((*((volatile uint32_t *) 0x504102d4)) != 0x00000000) return CNN_FAIL; // 6,3,9
    if ((*((volatile uint32_t *) 0x504102d8)) != 0x00000000) return CNN_FAIL; // 6,3,10
    if ((*((volatile uint32_t *) 0x504102dc)) != 0x00000000) return CNN_FAIL; // 6,3,11
    if ((*((volatile uint32_t *) 0x504182d0)) != 0x00000000) return CNN_FAIL; // 6,3,12
    if ((*((volatile uint32_t *) 0x504182d4)) != 0x00000000) return CNN_FAIL; // 6,3,13
    if ((*((volatile uint32_t *) 0x504182d8)) != 0x00000000) return CNN_FAIL; // 6,3,14
    if ((*((volatile uint32_t *) 0x504182dc)) != 0x00000000) return CNN_FAIL; // 6,3,15
    if ((*((volatile uint32_t *) 0x508002d0)) != 0x00000000) return CNN_FAIL; // 6,3,16
    if ((*((volatile uint32_t *) 0x508002d4)) != 0x00000000) return CNN_FAIL; // 6,3,17
    if ((*((volatile uint32_t *) 0x508002d8)) != 0x00000000) return CNN_FAIL; // 6,3,18
    if ((*((volatile uint32_t *) 0x508002dc)) != 0x00000000) return CNN_FAIL; // 6,3,19
    if ((*((volatile uint32_t *) 0x508082d0)) != 0x00000000) return CNN_FAIL; // 6,3,20
    if ((*((volatile uint32_t *) 0x508082d4)) != 0x00000000) return CNN_FAIL; // 6,3,21
    if ((*((volatile uint32_t *) 0x508082d8)) != 0x00000000) return CNN_FAIL; // 6,3,22
    if ((*((volatile uint32_t *) 0x508082dc)) != 0x00000000) return CNN_FAIL; // 6,3,23
    if ((*((volatile uint32_t *) 0x508102d0)) != 0x00000000) return CNN_FAIL; // 6,3,24
    if ((*((volatile uint32_t *) 0x508102d4)) != 0x00000000) return CNN_FAIL; // 6,3,25
    if ((*((volatile uint32_t *) 0x508102d8)) != 0x00000000) return CNN_FAIL; // 6,3,26
    if ((*((volatile uint32_t *) 0x508102dc)) != 0x00000000) return CNN_FAIL; // 6,3,27
    if ((*((volatile uint32_t *) 0x508182d0)) != 0x00000000) return CNN_FAIL; // 6,3,28
    if ((*((volatile uint32_t *) 0x508182d4)) != 0x00000000) return CNN_FAIL; // 6,3,29
    if ((*((volatile uint32_t *) 0x504002e0)) != 0x00000000) return CNN_FAIL; // 6,4,0
    if ((*((volatile uint32_t *) 0x504002e4)) != 0x00000000) return CNN_FAIL; // 6,4,1
    if ((*((volatile uint32_t *) 0x504002e8)) != 0x00000000) return CNN_FAIL; // 6,4,2
    if ((*((volatile uint32_t *) 0x504002ec)) != 0x00000000) return CNN_FAIL; // 6,4,3
    if ((*((volatile uint32_t *) 0x504082e0)) != 0x00000000) return CNN_FAIL; // 6,4,4
    if ((*((volatile uint32_t *) 0x504082e4)) != 0x00000000) return CNN_FAIL; // 6,4,5
    if ((*((volatile uint32_t *) 0x504082e8)) != 0x00000000) return CNN_FAIL; // 6,4,6
    if ((*((volatile uint32_t *) 0x504082ec)) != 0x00000000) return CNN_FAIL; // 6,4,7
    if ((*((volatile uint32_t *) 0x504102e0)) != 0x00000000) return CNN_FAIL; // 6,4,8
    if ((*((volatile uint32_t *) 0x504102e4)) != 0x00000000) return CNN_FAIL; // 6,4,9
    if ((*((volatile uint32_t *) 0x504102e8)) != 0x00000000) return CNN_FAIL; // 6,4,10
    if ((*((volatile uint32_t *) 0x504102ec)) != 0x00000000) return CNN_FAIL; // 6,4,11
    if ((*((volatile uint32_t *) 0x504182e0)) != 0x00000000) return CNN_FAIL; // 6,4,12
    if ((*((volatile uint32_t *) 0x504182e4)) != 0x00000000) return CNN_FAIL; // 6,4,13
    if ((*((volatile uint32_t *) 0x504182e8)) != 0x00000000) return CNN_FAIL; // 6,4,14
    if ((*((volatile uint32_t *) 0x504182ec)) != 0x00000000) return CNN_FAIL; // 6,4,15
    if ((*((volatile uint32_t *) 0x508002e0)) != 0x00000000) return CNN_FAIL; // 6,4,16
    if ((*((volatile uint32_t *) 0x508002e4)) != 0x00000000) return CNN_FAIL; // 6,4,17
    if ((*((volatile uint32_t *) 0x508002e8)) != 0x00000000) return CNN_FAIL; // 6,4,18
    if ((*((volatile uint32_t *) 0x508002ec)) != 0x00000000) return CNN_FAIL; // 6,4,19
    if ((*((volatile uint32_t *) 0x508082e0)) != 0x00000000) return CNN_FAIL; // 6,4,20
    if ((*((volatile uint32_t *) 0x508082e4)) != 0x00000000) return CNN_FAIL; // 6,4,21
    if ((*((volatile uint32_t *) 0x508082e8)) != 0x00000000) return CNN_FAIL; // 6,4,22
    if ((*((volatile uint32_t *) 0x508082ec)) != 0x00000000) return CNN_FAIL; // 6,4,23
    if ((*((volatile uint32_t *) 0x508102e0)) != 0x00000000) return CNN_FAIL; // 6,4,24
    if ((*((volatile uint32_t *) 0x508102e4)) != 0x00000000) return CNN_FAIL; // 6,4,25
    if ((*((volatile uint32_t *) 0x508102e8)) != 0x00000000) return CNN_FAIL; // 6,4,26
    if ((*((volatile uint32_t *) 0x508102ec)) != 0x00000000) return CNN_FAIL; // 6,4,27
    if ((*((volatile uint32_t *) 0x508182e0)) != 0x00000000) return CNN_FAIL; // 6,4,28
    if ((*((volatile uint32_t *) 0x508182e4)) != 0x00000000) return CNN_FAIL; // 6,4,29
    if ((*((volatile uint32_t *) 0x504002f0)) != 0x00000000) return CNN_FAIL; // 6,5,0
    if ((*((volatile uint32_t *) 0x504002f4)) != 0x00000000) return CNN_FAIL; // 6,5,1
    if ((*((volatile uint32_t *) 0x504002f8)) != 0x00000000) return CNN_FAIL; // 6,5,2
    if ((*((volatile uint32_t *) 0x504002fc)) != 0x00000000) return CNN_FAIL; // 6,5,3
    if ((*((volatile uint32_t *) 0x504082f0)) != 0x00000000) return CNN_FAIL; // 6,5,4
    if ((*((volatile uint32_t *) 0x504082f4)) != 0x00000000) return CNN_FAIL; // 6,5,5
    if ((*((volatile uint32_t *) 0x504082f8)) != 0x00000000) return CNN_FAIL; // 6,5,6
    if ((*((volatile uint32_t *) 0x504082fc)) != 0x00000000) return CNN_FAIL; // 6,5,7
    if ((*((volatile uint32_t *) 0x504102f0)) != 0x00000000) return CNN_FAIL; // 6,5,8
    if ((*((volatile uint32_t *) 0x504102f4)) != 0x00000000) return CNN_FAIL; // 6,5,9
    if ((*((volatile uint32_t *) 0x504102f8)) != 0x00000000) return CNN_FAIL; // 6,5,10
    if ((*((volatile uint32_t *) 0x504102fc)) != 0x00000000) return CNN_FAIL; // 6,5,11
    if ((*((volatile uint32_t *) 0x504182f0)) != 0x00000000) return CNN_FAIL; // 6,5,12
    if ((*((volatile uint32_t *) 0x504182f4)) != 0x00000000) return CNN_FAIL; // 6,5,13
    if ((*((volatile uint32_t *) 0x504182f8)) != 0x00000000) return CNN_FAIL; // 6,5,14
    if ((*((volatile uint32_t *) 0x504182fc)) != 0x00000000) return CNN_FAIL; // 6,5,15
    if ((*((volatile uint32_t *) 0x508002f0)) != 0x00000000) return CNN_FAIL; // 6,5,16
    if ((*((volatile uint32_t *) 0x508002f4)) != 0x00000000) return CNN_FAIL; // 6,5,17
    if ((*((volatile uint32_t *) 0x508002f8)) != 0x00000000) return CNN_FAIL; // 6,5,18
    if ((*((volatile uint32_t *) 0x508002fc)) != 0x00000000) return CNN_FAIL; // 6,5,19
    if ((*((volatile uint32_t *) 0x508082f0)) != 0x00000000) return CNN_FAIL; // 6,5,20
    if ((*((volatile uint32_t *) 0x508082f4)) != 0x00000000) return CNN_FAIL; // 6,5,21
    if ((*((volatile uint32_t *) 0x508082f8)) != 0x00000000) return CNN_FAIL; // 6,5,22
    if ((*((volatile uint32_t *) 0x508082fc)) != 0x00000000) return CNN_FAIL; // 6,5,23
    if ((*((volatile uint32_t *) 0x508102f0)) != 0x00000000) return CNN_FAIL; // 6,5,24
    if ((*((volatile uint32_t *) 0x508102f4)) != 0x00000000) return CNN_FAIL; // 6,5,25
    if ((*((volatile uint32_t *) 0x508102f8)) != 0x00000000) return CNN_FAIL; // 6,5,26
    if ((*((volatile uint32_t *) 0x508102fc)) != 0x00000000) return CNN_FAIL; // 6,5,27
    if ((*((volatile uint32_t *) 0x508182f0)) != 0x00000000) return CNN_FAIL; // 6,5,28
    if ((*((volatile uint32_t *) 0x508182f4)) != 0x00000000) return CNN_FAIL; // 6,5,29
    if ((*((volatile uint32_t *) 0x50400300)) != 0x00000000) return CNN_FAIL; // 6,6,0
    if ((*((volatile uint32_t *) 0x50400304)) != 0x00000000) return CNN_FAIL; // 6,6,1
    if ((*((volatile uint32_t *) 0x50400308)) != 0x00000000) return CNN_FAIL; // 6,6,2
    if ((*((volatile uint32_t *) 0x5040030c)) != 0x00000000) return CNN_FAIL; // 6,6,3
    if ((*((volatile uint32_t *) 0x50408300)) != 0x00000000) return CNN_FAIL; // 6,6,4
    if ((*((volatile uint32_t *) 0x50408304)) != 0x00000000) return CNN_FAIL; // 6,6,5
    if ((*((volatile uint32_t *) 0x50408308)) != 0x00000000) return CNN_FAIL; // 6,6,6
    if ((*((volatile uint32_t *) 0x5040830c)) != 0x00000000) return CNN_FAIL; // 6,6,7
    if ((*((volatile uint32_t *) 0x50410300)) != 0x00000000) return CNN_FAIL; // 6,6,8
    if ((*((volatile uint32_t *) 0x50410304)) != 0x00000000) return CNN_FAIL; // 6,6,9
    if ((*((volatile uint32_t *) 0x50410308)) != 0x00000000) return CNN_FAIL; // 6,6,10
    if ((*((volatile uint32_t *) 0x5041030c)) != 0x00000000) return CNN_FAIL; // 6,6,11
    if ((*((volatile uint32_t *) 0x50418300)) != 0x00000000) return CNN_FAIL; // 6,6,12
    if ((*((volatile uint32_t *) 0x50418304)) != 0x00000000) return CNN_FAIL; // 6,6,13
    if ((*((volatile uint32_t *) 0x50418308)) != 0x00000000) return CNN_FAIL; // 6,6,14
    if ((*((volatile uint32_t *) 0x5041830c)) != 0x00000000) return CNN_FAIL; // 6,6,15
    if ((*((volatile uint32_t *) 0x50800300)) != 0x00000000) return CNN_FAIL; // 6,6,16
    if ((*((volatile uint32_t *) 0x50800304)) != 0x00000000) return CNN_FAIL; // 6,6,17
    if ((*((volatile uint32_t *) 0x50800308)) != 0x00000000) return CNN_FAIL; // 6,6,18
    if ((*((volatile uint32_t *) 0x5080030c)) != 0x00000000) return CNN_FAIL; // 6,6,19
    if ((*((volatile uint32_t *) 0x50808300)) != 0x00000000) return CNN_FAIL; // 6,6,20
    if ((*((volatile uint32_t *) 0x50808304)) != 0x00000000) return CNN_FAIL; // 6,6,21
    if ((*((volatile uint32_t *) 0x50808308)) != 0x00000000) return CNN_FAIL; // 6,6,22
    if ((*((volatile uint32_t *) 0x5080830c)) != 0x00000000) return CNN_FAIL; // 6,6,23
    if ((*((volatile uint32_t *) 0x50810300)) != 0x00000000) return CNN_FAIL; // 6,6,24
    if ((*((volatile uint32_t *) 0x50810304)) != 0x00000000) return CNN_FAIL; // 6,6,25
    if ((*((volatile uint32_t *) 0x50810308)) != 0x00000000) return CNN_FAIL; // 6,6,26
    if ((*((volatile uint32_t *) 0x5081030c)) != 0x00000000) return CNN_FAIL; // 6,6,27
    if ((*((volatile uint32_t *) 0x50818300)) != 0x00000000) return CNN_FAIL; // 6,6,28
    if ((*((volatile uint32_t *) 0x50818304)) != 0x00000000) return CNN_FAIL; // 6,6,29

    return CNN_OK;
}

// *****************************************************************************
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q31_t max_box[5 + NUM_CLASSES] = {0};

int main(void)
{
	/* TFT_Demo Example */
	int key;
	State *state;

    // int ret = 0;
    // int slaveAddress;
    // int id;
    // int dma_channel;

#ifdef BOARD_FTHR_REVA
	// Wait for PMIC 1.8V to become available, about 180ms after power up.
	MXC_Delay(200000);
#endif
	/* Enable cache */
	MXC_ICC_Enable(MXC_ICC0);

	/* Set system clock to 100 MHz */
	MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
	SystemCoreClockUpdate();

	// Enable peripheral, enable CNN interrupt, turn on CNN clock
	// CNN clock: 50 MHz div 1
	cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
	cnn_init(); // Bring CNN state machine into consistent state
	cnn_load_weights(); // Load CNN kernels
	cnn_configure(); // Configure CNN state machine

	if (init_database() < 0 ) {
		PR_ERR("Could not initialize the database");
		return -1;
	}

	/* Initialize RTC */
	MXC_RTC_Init(0, 0);
	MXC_RTC_Start();

	volatile uint8_t read_byte = 0;

	while(1){
	const char *startSync = "Start_Sequence";

		while (1){
			if (MXC_UART_GetRXFIFOAvailable(MXC_UARTn) > 0) {
			read_byte = MXC_UART_ReadCharacter(MXC_UARTn);
			if (read_byte == 38 || read_byte == 48 || read_byte == 58) {
				break;
			}
			}
			
			uart_write((uint8_t*)startSync, 14);
			MXC_TMR_Delay(MXC_TMR0, MSEC(200));
		}

		switch (read_byte) {
			case 38: // Test Image
			uart_read(rxBuffer, sizeof(rxBuffer));
			uart_write(rxBuffer, sizeof(rxBuffer));
			break;

			case 48: // Test Embedding
			uart_read(rxBuffer, sizeof(rxBuffer));

			const char *cnn_receiveDataPass = "Pass_cnn_receiveData";
			uart_write((uint8_t*)cnn_receiveDataPass, 20);
			MXC_TMR_Delay(MXC_TMR0, MSEC(200));
			if (wait_for_feedback() == 0)
				return -1;

			if (!cnn_load(1)) {
				fail();
				return 0;
			}

			const char *cnn_loadPass = "Pass_cnn_load";
			uart_write((uint8_t*)cnn_loadPass, 13);
			MXC_TMR_Delay(MXC_TMR0, MSEC(200));
			if (wait_for_feedback() == 0)
				return -1;
			
			cnn_wait();
			const char *cnn_waitPass = "Pass_cnn_wait";
			uart_write((uint8_t*)cnn_waitPass, 13);
			MXC_TMR_Delay(MXC_TMR0, MSEC(200));
			if (wait_for_feedback() == 0)
				return -1;

			int success_check;
			success_check = cnn_check();
			const char *cnn_checkPass = "Pass_cnn_check";
			uart_write((uint8_t*)cnn_checkPass, 14);
			MXC_TMR_Delay(MXC_TMR0, MSEC(200));
			if (wait_for_feedback() == 0)
				return -1;

			uart_write((uint8_t*)(&success_check), sizeof(int));

			cnn_unload((uint32_t *) ml_data);
			const char *cnn_unloadPass = "Pass_cnn_unload";
			uart_write((uint8_t*)cnn_unloadPass, 15);
			MXC_TMR_Delay(MXC_TMR0, MSEC(200));
			if (wait_for_feedback() == 0)
				return -1;

			// send embedding to host device
			uart_write((uint8_t *)ml_data, sizeof(ml_data));

			break;

			case 58:
			uart_read(rxBuffer, sizeof(rxBuffer));

			if (!cnn_load(0)) {
				fail();
				return 0;
			}

			cnn_wait();

		//        if (cnn_check() != CNN_OK)
		//        {
		//          fail();
		//          return 0;
		//        }

			cnn_unload((uint32_t *) ml_data);
			NMS_max(ml_data, CNN_NUM_OUTPUTS, max_box);

			// send embedding to host device
			uart_write((uint8_t *)max_box, sizeof(max_box));

			break;
			
			default:
			break;
		}


	#ifdef TFT_ENABLE
	#ifdef BOARD_EVKIT_V1
		/* Initialize TFT display */
		MXC_TFT_Init(MXC_SPI0, 1, NULL, NULL);
		/* Set the screen rotation */
		MXC_TFT_SetRotation(SCREEN_ROTATE);
		/* Change entry mode settings */
		MXC_TFT_WriteReg(0x0011, 0x6858);
	#endif
	#ifdef BOARD_FTHR_REVA
		/* Initialize TFT display */
		MXC_TFT_Init(MXC_SPI0, 1, NULL, NULL);
		MXC_TFT_SetRotation(ROTATE_180);
		MXC_TFT_SetBackGroundColor(4);
		MXC_TFT_SetForeGroundColor(WHITE);   // set font color to white
	#endif
	#endif

	#ifdef TS_ENABLE
		/* Touch screen controller interrupt signal */
		mxc_gpio_cfg_t int_pin = {MXC_GPIO0, MXC_GPIO_PIN_17, MXC_GPIO_FUNC_IN, MXC_GPIO_PAD_NONE, MXC_GPIO_VSSEL_VDDIOH};
		/* Touch screen controller busy signal */
		mxc_gpio_cfg_t busy_pin = {MXC_GPIO0, MXC_GPIO_PIN_16, MXC_GPIO_FUNC_IN, MXC_GPIO_PAD_NONE, MXC_GPIO_VSSEL_VDDIOH};
		/* Initialize Touch Screen controller */
		MXC_TS_Init(MXC_SPI0, 2, &int_pin, &busy_pin);
		MXC_TS_Start();
	#endif

		/* Display Home page */
		state_init();

	#ifndef TS_ENABLE
		key = KEY_1;
	#endif
		while (1) { //TFT Demo
			/* Get current screen state */
			state = state_get_current();
	#ifdef TS_ENABLE
			/* Check pressed touch screen key */
			key = MXC_TS_GetKey();
	#endif

			if (key > 0) {
				state->prcss_key(key);
			}
		}

		return 0;
	}
}
