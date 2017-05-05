import re
import numpy as np
import struct


def load_ground_truth(filename):
    """
    Return a raster of integers from a PGM or PFM file as a numpy array
    """
    fp = open(filename, 'rb')
    if filename.endswith('.pgm'):
        assert fp.readline() == 'P5\n'
        (width, height) = [int(i) for i in fp.readline().split()]
        depth = int(fp.readline())
        assert depth <= 255
        raster = []
        for y in range(height):
            row = []
            for y in range(width):
                row.append(ord(fp.read(1)))
            raster.append(row)
        return np.asarray(raster) / 14

    elif filename.endswith('.pfm'):
        header = fp.readline().rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', fp.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(fp.readline().rstrip())
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(fp, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data[data == np.inf] = 0
        return np.flipud(np.reshape(data, shape))
    else:
        raise ValueError("Unknown file type")

if __name__ == '__main__':

    ground_truth = load_ground_truth("../experiments/tsuk_gt.pgm")
    print(ground_truth)