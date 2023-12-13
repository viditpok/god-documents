#!/usr/bin/env python

from __future__ import print_function
import argparse
import os
import importlib
import operator
import re

"""assembler.py: General, modular 2-pass assembler accepting ISA definitions to assemble code."""
__author__ = "Christopher Tam"


VERBOSE = False
FILE_NAME = ''
ISA = None
RE_PARAMS = re.compile('^(?P<key>.+)=(?P<value>.+)$')
SAVE_INST = False
inst_file = None

def verbose(s):
    if VERBOSE:
        print(s)
        
def print_inst(inst):
    if SAVE_INST:
        inst_file.write(inst)

def error(line_number, message):
    print("Error {}:{}: {}.\n".format(FILE_NAME, line_number, message))

def pass1(file):
    verbose("\nBeginning Pass 1...\n")
    # use a program counter to keep track of addresses in the file
    pc = 0
    line_count = 1
    no_errors = True

    # Seek to beginning of file
    #f.seek(0)

    for line in file:
        # Skip blank lines and comments
        if ISA.is_blank(line):
            verbose(line)
            continue
        
        # Check if PC is valid
        try:
            pc = ISA.validate_pc(pc)
        except Exception as e:
            error(line_count, str(e))
            break


        # Trim any leading and trailing whitespace
        line = line.strip()
        
        verbose('{}: {}'.format(pc, line))
        
        # Make line case-insensitive
        line = line.lower()
        
        # Parse line
        label, op, operands = ISA.get_parts(line)
        if label:
            if label in ISA.SYMBOL_TABLE:
                error(line_count, "label '{}' is defined more than once".format(label))
                no_errors = False
            else:
                ISA.SYMBOL_TABLE[label] = pc
                
        if op:
            if SAVE_INST:
                print_inst(op+operands.rstrip()+'\n')
            instr = None
            isOR = None
            try:
                if op == 'or': # workaround of not being able to name a class or
                    op = 'xor'
                    isOR = True

                instr = getattr(ISA, ISA.instruction_class(op))
            except:
                error(line_count, "instruction '{}' is not defined in the current ISA".format(op))
                no_errors = False

            try:
                if isOR:
                    op = 'or'
                    
                pc = instr.pc(pc=pc, instruction=op, operands=operands)
            except Exception as e:
                error(line_count, str(e))
                no_errors = False
                

        line_count += 1
        
    verbose("\nFinished Pass 1.\n")
        
    return no_errors


def pass2(input_file):
    verbose("\nBeginning Pass 2...\n")

    pc = 0
    line_count = 1
    success = True
    results = {}
    
    # Seek to beginning of file
    input_file.seek(0)
    
        
    for line in input_file:
        # Skip blank lines and comments
        if ISA.is_blank(line):
            verbose(line)
            continue
        
        # Trim any leading and trailing whitespace
        line = line.strip()

        verbose('{}: {}'.format(pc, line))
        
        # Make line case-insensitive
        line = line.lower()

        _, op, operands = ISA.get_parts(line)
                
        if op:
            isOR = None
            if op == 'or': # workaround of not being able to name a class or
                op = 'xor'
                isOR = True
            instr = getattr(ISA, ISA.instruction_class(op))
            assembled = None
            try:
                if isOR:
                    op = 'or'
                assembled = instr.create(operands, pc=pc, instruction=op)
            except Exception as e:
                error(line_count, str(e))
                success = False
            
            if assembled:
                for i in range(len(assembled)):
                    cur_pc = pc + i
                    if cur_pc in results:
                        error(line_count, "PC address {} is defined more than once".format(cur_pc))
                        success = False
                        break

                    results[cur_pc] = assembled[i]

                pc = instr.pc(pc=pc, instruction=op, operands=operands)
            
        line_count += 1
    
    verbose("\nFinished Pass 2.\n")
    return (success, results)

def separator(s):
    return s.replace('\s', ' ').encode().decode('unicode_escape')

def parse_params(values):
    if values is None:
        return None
    
    parsed = {}
    values = values.split(',')
    
    for val in values:
        m = RE_PARAMS.match(val)
        if m is None:
            print("Error: '{}' is not a valid custom parameter".format(val))
            exit(1)

        parsed[m.group('key')] = m.group('value')

    return parsed

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser('assembler.py', description='Assembles generic ISA-defined assembly code into hex or binary.')
    parser.add_argument('asmfile', help='the .s file to be assembled')
    parser.add_argument('-i', '--isa', required=False, type=str, default='isa', help='define the Python ISA module to load [default: isa]')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose printing of assembler')
    parser.add_argument('--hex', '--logisim', action='store_true', help='assemble code into hexadecimal (Logisim-compatible)')
    parser.add_argument('-s', '--separator', required=False, type=separator, default='\\n', help='the separator to use between instructions (accepts \s for space and standard escape characters) [default: \\n]')
    parser.add_argument('--sym', '--symbols', action='store_true', help="output an additional file containing the assembled program's symbol table")
    parser.add_argument('--params', required=False, type=str, help='custom parameters to pass to an architecture, formatted as "key1=value1, key2=value2, key3=value3"')
    parser.add_argument('--save', required=False, action='store_true')
    args = parser.parse_args()
    

    # Try to dynamically load ISA module
    try:
        ISA = importlib.import_module(args.isa)
    except Exception as e:
        print("Error: Failed to load ISA definition module '{}'. {}\n".format(args.isa, str(e)))
        exit(1)
        
    print("Assembling for {} architecture...".format(ISA.__name__))

    # Pass in custom parameters
    try:
        ISA.receive_params(parse_params(args.params))
    except Exception as e:
        print("Error: Failed to parse custom parameters for {}. {}\n".format(ISA.__name__, str(e)))
        exit(1)

    VERBOSE = args.verbose
    if args.save:
        SAVE_INST = True
        inst_file = open('inst.s', 'w', encoding='UTF8')

    FILE_NAME = os.path.basename(args.asmfile)
    
    with open(args.asmfile, 'r') as read_file:
        if not pass1(read_file):
            print("Assemble failed.\n")
            exit(1)
        
        success, results = pass2(read_file)
        if not success:
            print("Assemble failed.\n")
            exit(1)
        
    outFileName = os.path.splitext(args.asmfile)[0]
    code_ext = '.hex' if args.hex else '.bin'
    sep = args.separator

    if args.sym:
        sym_ext = '.sym'
        print("Writing symbol table to {}...".format(outFileName + sym_ext), end="")

        sym_sorted = sorted(ISA.SYMBOL_TABLE.items(), key=operator.itemgetter(1))

        with open(outFileName + sym_ext, 'w') as write_file:
            for (symbol, addr) in sym_sorted:
                write_file.write("{}: {}\n".format(symbol, hex(addr)))

        print('done!')


    print("Writing to {}...".format(outFileName + code_ext), end="")
        
    with open(outFileName + code_ext, 'w') as write_file:
        out_generator = ISA.output_generator(results, 'hex' if args.hex else 'binary')
        for r in out_generator:
            write_file.write(r + sep)

    print('done!')
