# Serial Loader



Description
-----------

This is an example of loading sample data with serial port. The synthesized main.c is modified to use serial loading if `#define SERIAL_INPUT` is enabled.

The python script needs to be modified according to the selected COM port.

First flash the FW into Evkit. Then run the python script. Then reset the Evkit. The script waits to see "READY" string received from Evkit and then starts to send sampledata.h data to Evkit.

The debug print messages of the firmware are echoed in python terminal.





