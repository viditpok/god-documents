import re

"""lc2222.py: A definition of the RAMA-2200 architecture."""
__author__ = "Christopher Tam/Adithya Nott/Kavin Krishnan/Mitchell Alvarado"\
    + "/Gavin Shek/Jonathan Procter/Jackson Williams"\
    + "/Tristan Rogers/Prit Patel"


# Define the name of the architecture
__name__ = 'LC-2222' # Fall 2023

# Define overall architecture widths (in bits)
BIT_WIDTH = 32
# Define opcode widths (in bits)
OPCODE_WIDTH = 4
# Define register specifier widths (in bits)
REGISTER_WIDTH = 4

ALIASES = {
    '.word':   'fill',
    '.fill':   'fill',
}

REGISTERS = {
    '$zero':   0,
    '$at':   1,
    '$v0':   2,
    '$a0':   3,
    '$a1':   4,
    '$a2':   5,
    '$t0':   6,
    '$t1':   7,
    '$t2':   8,
    '$s0':   9,
    '$s1':   10,
    '$s2':   11,
    '$k0':   12,
    '$sp':   13,
    '$fp':   14,
    '$ra':   15
}


SYMBOL_TABLE = {}

VALID_PARAMS = {}

PARAMS = {}


# Private Variables
PC_OFFSET_SIZE = BIT_WIDTH - OPCODE_WIDTH - (REGISTER_WIDTH * 2)
assert(PC_OFFSET_SIZE > 0)  # Sanity check

IMM_OFFSET_SIZE = BIT_WIDTH - OPCODE_WIDTH - (REGISTER_WIDTH * 2)
assert(IMM_OFFSET_SIZE > 0)  # Sanity check

MEM_OFFSET_SIZE = BIT_WIDTH - OPCODE_WIDTH - (REGISTER_WIDTH * 2)
assert(MEM_OFFSET_SIZE > 0)

R_UNUSED_SIZE = BIT_WIDTH - OPCODE_WIDTH - (REGISTER_WIDTH * 3)
assert(R_UNUSED_SIZE > 0)  # Sanity check

RELATIVE_UNUSED_BITS = 4

OR_MODE_BITS = 1
OR_UNUSED_SIZE = 15

RE_BLANK = re.compile(r'^\s*(!.*)?$')
RE_PARTS = re.compile(
    r'^\s*((?P<Label>\w+):)?\s*((?P<Opcode>\.?[\w]+)(?P<Operands>[^!]*))?(!.*)?')


def zero_extend(binary, target, pad_right=False):
    if binary.startswith('0b'):
        binary = binary[2:]

    zeros = '0' * (target - len(binary))
    if pad_right:
        return binary + zeros
    else:
        return zeros + binary


def sign_extend(binary, target):
    if binary.startswith('0b'):
        binary = binary[2:]

    sign = binary[0] if len(binary) > 1 else '0'
    return sign * (target - len(binary)) + binary


def bin2hex(binary):
    return '%0*X' % ((len(binary) + 3) // 4, int(binary, 2))


def hex2bin(hexadecimal):
    return bin(int(hexadecimal, 16))[2:]


def dec2bin(num, bits):
    """Compute the 2's complement binary of an int value."""
    return format(num if num >= 0 else (1 << bits) + num, '0{}b'.format(bits))


def parse_register(operand):
    if operand in REGISTERS:
        return zero_extend(bin(REGISTERS[operand])[2:], REGISTER_WIDTH)
    else:
        raise RuntimeError(
            "Register identifier '{}' is not valid in {}.".format(operand, __name__))


def parse_value(offset, size, pc=None, unsigned=False):
    bin_offset = None

    if type(offset) is str:
        if pc is not None and offset in SYMBOL_TABLE:
            offset = SYMBOL_TABLE[offset] - pc - 1
        elif offset.startswith('0x'):
            try:
                bin_offset = hex2bin(offset)
            except:
                raise RuntimeError(
                    "'{}' is not in a valid hexadecimal format.".format(offset))

            if len(bin_offset) > size:
                raise RuntimeError(
                    "'{}' is too large for {}.".format(offset, __name__))

            bin_offset = zero_extend(bin_offset, size)
        elif offset.startswith('0b'):
            try:
                bin_offset = bin(int(offset))
            except:
                raise RuntimeError(
                    "'{}' is not in a valid binary format.".format(offset))

            if len(bin_offset) > size:
                raise RuntimeError(
                    "'{}' is too large for {}.".format(offset, __name__))

            bin_offset = zero_extend(bin_offset, size)

    if bin_offset is None:
        try:
            offset = int(offset)
        except:
            if pc is not None:
                raise RuntimeError(
                    "'{}' cannot be resolved as a label or a value.".format(offset))
            else:
                raise RuntimeError(
                    "'{}' cannot be resolved as a value.".format(offset))

        if unsigned:
            bound = (2**size)

            # >= bound because range is [0, 2^n - 1]
            if offset < 0:
                raise RuntimeError(
                    "'{}' cannot be a negative value for {}.".format(offset, __name__))
            elif offset >= bound:
                raise RuntimeError(
                    "'{}' is too large (as a value) or too far away (as a label) for {}.".format(offset, __name__))
        else:
            bound = 2**(size - 1)

            if offset < -bound:
                raise RuntimeError(
                    "'{}' is too small (as a value) or too far away (as a label) for {}.".format(offset, __name__))
            elif offset >= bound:
                raise RuntimeError(
                    "'{}' is too large (as a value) or too far away (as a label) for {}.".format(offset, __name__))

        bin_offset = dec2bin(offset, size)

    return bin_offset


class Instruction:
    """
    This is the base class that all implementations of instructions must override.
    """

    __RE_NO_OPERANDS = re.compile(r'^\s*$')

    @classmethod
    def opcode(cls):
        """Return the operation code for the given instruction as an integer."""
        raise NotImplementedError()

    def __init__(self, operands, pc, instruction):
        self.__operands = operands
        self.bin_operands = self.parse_operands(operands, pc, instruction)
        self.__pc = pc
        self.__instruction = instruction

    @classmethod
    def create(cls, operands, pc, instruction):
        """Generates a list of Instruction(s) for the given operands."""
        raise NotImplementedError()

    @classmethod
    def pc(cls, pc, **kwargs):
        """Return the new PC after assembling the given instruction"""
        # By default, return pc + 1
        return pc + 1

    @classmethod
    def parse_operands(cls, operands, pc, instruction):
        match = cls.__RE_NO_OPERANDS.match(operands)

        if match is None:
            raise RuntimeError(
                "Operands '{}' are not permitted.".format(operands.strip()))

        return ''

    def binary(self):
        """Assemble the instruction into binary form.

        Returns a string representation of the binary instruction.
        """
        raise NotImplementedError()

    def hex(self):
        """Assemble the instruction into binary form.

        Returns a string representation of the binary instruction.
        """
        return bin2hex(self.binary())


class RInstruction(Instruction):
    """
    The base class for R-type instructions.
    """

    __RE_R = re.compile(
        r'^\s*(?P<RX>\$\w+?)\s*,\s*(?P<RY>\$\w+?)\s*,\s*(?P<RZ>\$\w+?)(,\s*(?P<ShfType>LSL|LSR|ASR)\s+(?P<ShfDist>\d+))?\s*$', re.I)

    @classmethod
    def create(cls, operands, pc, instruction):
        return [cls(operands, pc, instruction)]

    @classmethod
    def parse_operands(cls, operands, pc, instruction):
        # Define result
        result_list = []

        match = cls.__RE_R.match(operands)

        if match is None:
            raise RuntimeError(
                "Operands '{}' are in an incorrect format.".format(operands.strip()))

        result_list.append(parse_register(match.group('RX')))
        result_list.append(parse_register(match.group('RY')))
        # Insert unused bits
        result_list.append('0' * R_UNUSED_SIZE)
        result_list.append(parse_register(match.group('RZ')))

        return ''.join(result_list)

    def binary(self):
        return zero_extend(bin(self.opcode()), OPCODE_WIDTH) + self.bin_operands


class MemInstruction(Instruction):
    """
    The base class for memory access instructions.
    """

    __RE_MEM = re.compile(
        r'^\s*(?P<RX>\$\w+?)\s*,\s*(?P<Offset>\S+?)\s*\((?P<RY>\$\w+?)\)\s*$')

    @classmethod
    def create(cls, operands, pc, instruction):
        return [cls(operands, pc, instruction)]

    @classmethod
    def parse_operands(cls, operands, pc, instruction):
        # Define result
        result_list = []

        match = cls.__RE_MEM.match(operands)

        if match is None:
            raise RuntimeError(
                "Operands '{}' are in an incorrect format.".format(operands.strip()))

        for op in (match.group('RX'), match.group('RY')):
            result_list.append(parse_register(op))

        result_list.append(parse_value(
            match.group('Offset'), MEM_OFFSET_SIZE, pc))

        return ''.join(result_list)

    def binary(self):
        return zero_extend(bin(self.opcode()), OPCODE_WIDTH) + self.bin_operands


class add(RInstruction):
    @classmethod
    def opcode(cls):
        return 0


class addi(Instruction):
    __RE_ADDI = re.compile(
        r'^\s*(?P<RX>\$\w+?)\s*,\s*(?P<RY>\$\w+?)\s*,\s*(?P<Offset>\S+?)\s*(,\s*(?P<ShfType>LSL|LSR|ASR)\s+(?P<ShfDist>\d+))?$', re.I)

    @classmethod
    def opcode(cls):
        return 2

    @classmethod
    def create(cls, operands, pc, instruction):
        return [cls(operands, None, instruction)]

    @classmethod
    def parse_operands(cls, operands, pc, instruction):
        # Define result
        result_list = []

        match = cls.__RE_ADDI.match(operands)

        if match is None:
            raise RuntimeError(
                "Operands '{}' are in an incorrect format.".format(operands.strip()))

        for op in (match.group('RX'), match.group('RY')):
            result_list.append(parse_register(op))

        result_list.append(parse_value(
            match.group('Offset'), IMM_OFFSET_SIZE, pc))

        return ''.join(result_list)

    def binary(self):
        return zero_extend(bin(self.opcode()), OPCODE_WIDTH) + self.bin_operands


class nand(RInstruction):
    @classmethod
    def opcode(cls):
        return 1


class beq(Instruction):
    __RE_BEQ = re.compile(r'^\s*(?P<RX>\$\w+?)\s*,\s*(?P<RY>\$\w+?)\s*,\s*(?P<Offset>\S+?)\s*$')

    @classmethod
    def opcode(cls):
        return 5

    @classmethod
    def create(cls, operands, pc, instruction):
        return [cls(operands, pc, instruction)]

    @classmethod
    def parse_operands(cls, operands, pc, instruction):
        # Define result
        result_list = []

        match = cls.__RE_BEQ.match(operands)
        if match is None:
            raise RuntimeError("Operands '{}' are in an incorrect format.".format(operands.strip()))

        # Relative PC Address.

        result_list.append(parse_register(match.group('RX')))
        result_list.append(parse_register(match.group('RY')))
        result_list.append(parse_value(match.group('Offset'), PC_OFFSET_SIZE , pc))

        return ''.join(result_list)

    def binary(self):
        return zero_extend(bin(self.opcode()), OPCODE_WIDTH) + self.bin_operands

class blt(Instruction):
    __RE_BLT = re.compile(r'^\s*(?P<RX>\$\w+?)\s*,\s*(?P<RY>\$\w+?)\s*,\s*(?P<Offset>\S+?)\s*$')

    @classmethod
    def opcode(cls):
        return 8

    @classmethod
    def create(cls, operands, pc, instruction):
        return [cls(operands, pc, instruction)]

    @classmethod
    def parse_operands(cls, operands, pc, instruction):
        # Define result
        result_list = []

        match = cls.__RE_BLT.match(operands)
        if match is None:
            raise RuntimeError("Operands '{}' are in an incorrect format.".format(operands.strip()))

        result_list.append(parse_register(match.group('RX')))
        result_list.append(parse_register(match.group('RY')))
        result_list.append(parse_value(match.group('Offset'), PC_OFFSET_SIZE , pc))

        return ''.join(result_list)
    
    def binary(self):
        return zero_extend(bin(self.opcode()), OPCODE_WIDTH) + self.bin_operands

class bgt(Instruction):
    __RE_BGT = re.compile(r'^\s*(?P<RX>\$\w+?)\s*,\s*(?P<RY>\$\w+?)\s*,\s*(?P<Offset>\S+?)\s*$')

    @classmethod
    def opcode(cls):
        return 10

    @classmethod
    def create(cls, operands, pc, instruction):
        return [cls(operands, pc, instruction)]

    @classmethod
    def parse_operands(cls, operands, pc, instruction):
        # Define result
        result_list = []

        match = cls.__RE_BGT.match(operands)
        if match is None:
            raise RuntimeError("Operands '{}' are in an incorrect format.".format(operands.strip()))

        result_list.append(parse_register(match.group('RX')))
        result_list.append(parse_register(match.group('RY')))
        result_list.append(parse_value(match.group('Offset'), PC_OFFSET_SIZE , pc))

        return ''.join(result_list)
    
    def binary(self):
        return zero_extend(bin(self.opcode()), OPCODE_WIDTH) + self.bin_operands

class lea(Instruction):
    __RE_LEA = re.compile(r'^\s*(?P<RX>\$\w+?)\s*,\s*(?P<Offset>\S+?)\s*$')

    @classmethod
    def opcode(cls):
        return 9

    @classmethod
    def create(cls, operands, pc, instruction):
        return [cls(operands, pc, instruction)]

    @classmethod
    def parse_operands(cls, operands, pc, instruction):
        match = cls.__RE_LEA.match(operands)
        if match is None:
            raise RuntimeError(
                "Operands '{}' are in an incorrect format.".format(operands.strip()))

        result_list = []

        result_list.append(parse_register(match.group('RX')))
        result_list.append('0' * RELATIVE_UNUSED_BITS)
        result_list.append(parse_value(
            match.group('Offset'), PC_OFFSET_SIZE, pc))

        return ''.join(result_list)

    def binary(self):
        return zero_extend(bin(self.opcode()), OPCODE_WIDTH) + self.bin_operands


class lw(MemInstruction):
    @classmethod
    def opcode(cls):
        return 3

    @classmethod
    def is_offset_style(cls):
        return True


class sw(MemInstruction):
    @classmethod
    def opcode(cls):
        return 4

    @classmethod
    def is_offset_style(cls):
        return True


class jalr(Instruction):
    __RE_JALR = re.compile(r'^\s*(?P<AT>\$\w+?)\s*,\s*(?P<RA>\$\w+?)\s*$')

    @classmethod
    def opcode(cls):
        return 6

    @classmethod
    def create(cls, operands, pc, instruction):
        return [cls(operands, pc, instruction)]

    @classmethod
    def parse_operands(cls, operands, pc, instruction):
        # Define result
        result_list = []

        match = cls.__RE_JALR.match(operands)

        if match is None:
            raise RuntimeError(
                "Operands '{}' are in an incorrect format.".format(operands.strip()))

        result_list.append(parse_register(match.group('AT')))
        result_list.append(parse_register(match.group('RA')))

        return ''.join(result_list)

    def binary(self):
        padded_opcode = zero_extend(bin(self.opcode()), OPCODE_WIDTH)
        return zero_extend(padded_opcode + self.bin_operands, BIT_WIDTH, pad_right=True)

class xor(Instruction):
    __RE_XOR = re.compile(r'^\s*(?P<RX>\$\w+?)\s*,\s*(?P<RY>\$\w+?)\s*,\s*(?P<RZ>\$\w+?)\s*$')

    @classmethod
    def opcode(cls):
        return 11

    @classmethod
    def create(cls, operands, pc, instruction):
        return [cls(operands, pc, instruction)]
    
    @classmethod
    def parse_operands(cls, operands, pc, instruction):
        match = cls.__RE_XOR.match(operands)
        
        if match is None:
            raise RuntimeError(
                "Operands '{}' are in an incorrect format.".format(operands.strip()))

        result_list = []

        if instruction == 'or':
            mode = bin(0)
        elif instruction == 'xor':
            mode = bin(1)

        # Append results in order
        result_list.append(parse_register(match.group('RX')))
        result_list.append(parse_register(match.group('RY')))
        result_list.append('0' * OR_UNUSED_SIZE)
        result_list.append(zero_extend(mode, OR_MODE_BITS))
        result_list.append(parse_register(match.group('RZ')))

        return ''.join(result_list)
    
    def binary(self):
        return zero_extend(bin(self.opcode()), OPCODE_WIDTH) + self.bin_operands


class halt(Instruction):
    @classmethod
    def opcode(cls):
        return 7

    @classmethod
    def create(cls, operands, pc, instruction):
        return [cls(operands, pc, instruction)]

    def binary(self):
        padded_opcode = zero_extend(bin(self.opcode()), OPCODE_WIDTH)
        return zero_extend(padded_opcode, BIT_WIDTH, pad_right=True)

class inc(Instruction):
    __RE_INC = re.compile(r'^\s*(?P<DR>\$\w+?)\s*$')
    # re.compile(r'^\s*(?P<RX>\$\w+?)\s*,\s*(?P<RY>\$\w+?)\s*,\s*(?P<Offset>\S+?)\s*(,\s*(?P<ShfType>LSL|LSR|ASR)\s+(?P<ShfDist>\d+))?$', re.I)

    @classmethod
    def opcode(cls):
        return 15

    @classmethod
    def create(cls, operands, pc, instruction):
        return [cls(operands, None, instruction)]

    @classmethod
    def parse_operands(cls, operands, pc, instruction):
        # Define result
        result_list = []

        match = cls.__RE_INC.match(operands)

        if match is None:
            raise RuntimeError(
                "Operands '{}' are in an incorrect format.".format(operands.strip()))

        result_list.append(parse_register(match.group('DR')))

        return ''.join(result_list)

    def binary(self):
        padded_opcode = zero_extend(bin(self.opcode()), OPCODE_WIDTH)
        return zero_extend(padded_opcode + self.bin_operands, BIT_WIDTH, pad_right=True)

class noop(add):
    """noop

    Equivalent to:
    add $zero, $zero, $zero
    """

    @classmethod
    def create(cls, operands, pc, instruction):
        return [cls('$zero, $zero, $zero', pc, instruction)]


class fill(Instruction):
    @classmethod
    def opcode(cls):
        return None

    @classmethod
    def create(cls, operands, pc, instruction):
        return [cls(operands, pc, instruction)]

    @classmethod
    def parse_operands(cls, operands, pc, instruction):
        if type(operands) is str:
            operands = operands.strip()

        return parse_value(operands, BIT_WIDTH)

    def binary(self):
        return self.bin_operands


# Functions expected by the assembler
def receive_params(value_table):
    if value_table:
        raise RuntimeError('Custom parameters are not supported')


def is_blank(line):
    """Return whether a line is blank and not an instruction."""
    return RE_BLANK.match(line) is not None


def get_parts(line):
    """Break down an instruction into 3 parts: Label, Opcode, Operand"""
    m = RE_PARTS.match(line)
    try:
        return m.group('Label'), m.group('Opcode'), m.group('Operands')
    except:
        return None


def instruction_class(name):
    """Translate a given instruction name to its corresponding class name."""
    return ALIASES.get(name, name)


def validate_pc(pc):
    """Returns or modifies the PC to a permitted value, if possible. Throws an error if the PC is invalid."""
    if pc >= 2**BIT_WIDTH:
        raise RuntimeError(
            "PC value {} is too large for {} bits.".format(pc, BIT_WIDTH))

    return pc


def output_generator(assembled_dict, output_format='binary'):
    """Returns a generator that creates output from {pc : assembly}-formatted dictionary."""
    pc = 0
    count = 0

    while count < len(assembled_dict):
        instr = None
        if pc in assembled_dict:
            instr = assembled_dict[pc]
            pc += 1
            count += 1
        else:
            instr = noop.create('', pc, 'noop')

            pc = instr.pc(pc)

        yield getattr(instr, output_format)()
