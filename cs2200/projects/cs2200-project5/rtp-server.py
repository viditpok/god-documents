#!/usr/bin/python3

from socket import *
from sys import exit
from random import random
import struct
import argparse
import codecs

HOST = '127.0.0.1'
PORT = 0
DROP_RATE = 0
PACKET_SIZE = 24
PACKET_FMT = "iii10sxx"
DATA = 0
LAST_DATA = 1
ACK = 2 
NACK = 3
TERM = 4
MAX_PAYLOAD_LENGTH = 10
alphabet = "abcdefghijklmnopqrstuvwxyz"
  
def rot13(message):
   new_message = ""
   for char in message:
        if alphabet.find(char):
           new_message += alphabet[(alphabet.find(char)+13)%26]
        else:
            new_message += char
   return new_message

def checksum(msg):
    check = 0
    count = 0
    opodd = 2
    for letter in msg:
        if (count % 2) == 0:
            if opodd:
                check += ord(letter) * 3
                opodd += 1
            else:
                check += ord(letter) // 3
                opodd = 0
        else:
            if opodd:
                check += ord(letter) * 3
                opodd = 0
            else:
                check += ord(letter) // 3
                opodd += 2
        count = count + 1
    return check

def packetize(msg):
    num = int(len(msg)/MAX_PAYLOAD_LENGTH + (1 if len(msg)%MAX_PAYLOAD_LENGTH != 0 else 0))
    packets = []
    for x in range(num-1):
        payload = msg[(x*MAX_PAYLOAD_LENGTH):((x*MAX_PAYLOAD_LENGTH)+MAX_PAYLOAD_LENGTH)]
        packet = [DATA, checksum(payload), MAX_PAYLOAD_LENGTH, payload]
        packets.append(packet)
    payload = msg[(num-1)*MAX_PAYLOAD_LENGTH:]
    packet = [LAST_DATA, checksum(payload), len(payload), payload]
    packets.append(packet)
    return packets

def runServer():
    s = socket(AF_INET, SOCK_DGRAM)
    s.bind((HOST, PORT))
    while True:
        last_type = DATA
        message = ""
        print('\n\n============ START NEW MESSAGE ============')
        while last_type == DATA:
            msg = s.recvfrom(PACKET_SIZE)
            client = msg[1]
            r_type, r_chksum, r_len, r_payload = struct.unpack(PACKET_FMT, bytes(msg[0]))

            if r_type == TERM:
                print("Server received TERM. Exiting...")
                exit(0)
            elif r_type not in [DATA, LAST_DATA]:
                print("Server received invalid packet type while receiving message! \
                        Exiting...")
                exit(1)

            if random() < DROP_RATE:
                print('SIMULATING NETWORK CORRUPTION - ERROR IN CHECKSUM OCCURRED...', end=' ') 
                r_chksum += 13
            
            r_payload = r_payload[0:r_len].decode()
            ret_type = ACK if r_chksum == checksum(r_payload) else NACK
            print("Received: '"+ r_payload +"'...", end=' ') 
            # r_payload = r_payload.strip('\0x00')
            message += r_payload if ret_type == ACK else ""
            print("Sending ACK" if ret_type == ACK else "Sending NACK")
            last_type = LAST_DATA if r_type == LAST_DATA and ret_type == ACK else DATA

            s.sendto(struct.pack(PACKET_FMT, ret_type, 0, 0, b''), client)

        print("Received LAST_DATA successfully. Sending back message to client")

        essagemay = codecs.encode(message, 'rot_13')
        essagemay = essagemay.strip()
        packets = packetize(essagemay)
        print("\nOriginal message: " + message)
        print("Response: '"+essagemay+"'\n")
        i = 0
        while(i<len(packets)):
            curr_pkt = packets[i][:]
            if random() < DROP_RATE:
                print('SIMULATING NETWORK CORRUPTION - CREATING ERROR IN CHECKSUM...', end=' ') 
                curr_pkt[1] += 13
            print("Sending packet #"+str(i)+"... "+str(curr_pkt)+".", end=' ')
            s.sendto(struct.pack(PACKET_FMT, curr_pkt[0], curr_pkt[1], 
                curr_pkt[2], curr_pkt[3].encode()), client)

            msg = s.recvfrom(PACKET_SIZE)
            r_type, r_chksum, r_len, r_payload = struct.unpack(PACKET_FMT, msg[0])
            if r_type == TERM:
                print("Server received TERM. Exiting...")
                exit(0)
            elif r_type != ACK and r_type != NACK:
                print("Server received invalid packet type while sending message! \
                        Exiting...")
                exit(1)
            i += 1 if r_type == ACK else 0
            print('Received ACK' if r_type == ACK else 'Received NACK')

    return

parser = argparse.ArgumentParser(description="\tRTP Server for CS2200 Project 5\n\n")
parser.add_argument('-p', '--port', help='\t\tInteger port that the server should listen on', type=int, required=True)
parser.add_argument('-c', '--corruption', help='\t\tFloat that determines corruption percentage. \n\
        (e.g. -c .5 will give the connection a 50% corruption rate)', type=float, default=0.0)
args = parser.parse_args()
print(args.port)
print(args.corruption)
PORT = args.port
DROP_RATE = args.corruption

runServer()
