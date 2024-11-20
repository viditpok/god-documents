import math
import random

### Release to students ###

def basic_noise(v):
    return v + random.random()*.01

def add_noise(robbie):
    robbie.vl = basic_noise(robbie.vl)
    robbie.vr = basic_noise(robbie.vr)

def add_offset_noise(robbie):
    robbie.vl = offset_noise(robbie.vl)
    robbie.vr = offset_noise(robbie.vr)

OFFSET = 0.01

def offset_noise(v):
    return v+OFFSET
    