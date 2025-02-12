#!/bin/python3

"""
This file contains the code under test for the example bug.
The sqrt() method fails on x <= 0.
"""
from math import tan as rtan
from math import cos as rcos
from math import sin as rsin


def prob(inp):
    e = str(inp).split()
    le = int(e[0]) + 1
    p = 0
    if len(e) > 1:
        p = len(e[1])

    if int(le) > int(p):
        return False
    return True
#    print("Length: ", e[0], "Payload: ", e[1])
