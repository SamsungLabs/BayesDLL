import os
import sys
import pickle
import numpy as np
from PIL import Image


def mkdir(paths):

    if not isinstance(paths, (list, tuple)):
        paths = [paths]

    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


def cprint(color, text, **kwargs):

    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }

    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    
    sys.stdout.flush()

