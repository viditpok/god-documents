The LC-2222a Assembler
===============

To aid in testing your processor, we have provided an *assembler* and
a *simulator* for the LC-2222a architecture. The assembler supports
converting text `.s` files into either binary (for the simulator) or
hexadecimal (for pasting into CircuitSim) formats.

Requirements
-----------

The assembler and simulator run on any version of Python 2.6+. An
instruction set architecture definition file is required along with
the assembler. The LC-2222a assembler definition is included.

Included Files
-----------

* `assembler.py`: the main assembler program
* `lc2222a.py`: the LC-2222a assembler definition

Using the Assembler
-----------

### Assemble for CircuitSim

To output assembled code in hexadecimal (for use with *CircuitSim*):

    python assembler.py -i lc2222a --hex test.s

You can then open the resulting `test.hex` file in your favorite text
editor.  In CircuitSim, right-click on your RAM, select **Edit
Contents...**, and then copy-and-paste the contents of the hex file
into the window.

Do not use the Open or Save buttons in CircuitSim, as it will not
recognize the format.

For this project, we do not provide the simulator file as the final output will depend on specific CircuitSim implementation, clock cycles, and unexpected interrupts.