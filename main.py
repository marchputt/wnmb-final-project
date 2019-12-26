import os
import time
import zlib

from tqdm import tqdm
import openslide
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import io


# Configuration
if os.path.exists('list.txt'):
    fn_list = np.loadtxt('list.txt', dtype='str')
tile_size = (128, 128)


def imshow_cv(input_cv_img):
    # imshow area
    cv.imshow('Figure', input_cv_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def imshow_sk(input_skimage, cv_img_yn=0):
    """Easy MATLAB-like imshow function just for development and debugging based on matplotlib-pyplot

    :param input_skimage: input image compatible with skimage format
    :return: (Window that displays the input image)
    """
    if cv_img_yn == 1:
        # Convert OpenCV image into Scikit Image format
        input_skimage = input_skimage[:, :, ::-1]

    fig = plt.figure("Figure")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(input_skimage)
    plt.axis("off")
    plt.show()


def read_wsi(wsi_file_path='wsi.svs', dst_dir='export'):
    # Make '/export' if does not exist.
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    slide = openslide.OpenSlide(wsi_file_path)
    w, h = slide.dimensions

    start = time.time()
    # Loop through the image with tile size.
    v_loop = int(h / tile_size[1])
    h_loop = int(w / tile_size[0])
    _fn_list = list()
    wsi_tiles = list()
    for v_block in tqdm(range(0, v_loop)):
        for h_block in range(0, h_loop):
            current_tl = [h_block * tile_size[0], v_block * tile_size[1]]
            region = slide.read_region(current_tl, 0, tile_size)
            region_cv = np.array(region)
            wsi_tiles.append(region_cv)
    stop = time.time()
    print('READ Time: ' + str(stop - start))

    return wsi_tiles


def read_wsi_and_save_images(wsi_file_path='wsi.svs', dst_dir='export'):
    # Config
    tile_size = (128, 128)

    # Make '/export' if does not exist.
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    slide = openslide.OpenSlide(wsi_file_path)
    w, h = slide.dimensions

    # Loop through the image with tile size.
    v_loop = int(h/tile_size[1])
    h_loop = int(w/tile_size[0])
    _fn_list = list()
    for v_block in tqdm(range(0, v_loop)):
        for h_block in range(0, h_loop):
            current_tl = [h_block * tile_size[0], v_block * tile_size[1]]
            region = slide.read_region(current_tl, 0, tile_size)
            region_cv = np.array(region)

            # Save the image
            export_filename = os.path.join(dst_dir, str(v_block) + '_' + str(h_block) + '.png')
            cv.imwrite(export_filename, region_cv)
            _fn_list.append(export_filename)

    np.savetxt('list.txt', _fn_list, fmt='%s')
    return _fn_list


def read_and_time():
    """Read from 'export/' folder and time the process

    :return: elapsed read time in seconds.
    :rtype: float
    """
    start = time.time()
    for fn in fn_list:
        tmp_img = cv.imread(fn)
    stop = time.time()
    total = stop - start
    print('Total read time')
    print(total)
    return total


def lossy_compression():
    """Perform lossy compression using generic JPEG and Guetzli JPEG compression

    :return:
    """
    start = time.time()
    dst_dir = 'guetzli'
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    g_list = list()
    for fn in tqdm(fn_list):
        export_fn = os.path.join(dst_dir, fn[7:len(fn)-4] + '.jpg')
        g_list.append(export_fn)
        cmd = 'guetzli ' + fn + ' ' + export_fn
        os.system(cmd)
    stop = time.time()
    g_total = stop - start
    print('Total Guetzli compression time')
    print(g_total)

    # Save list of file for G compression
    np.savetxt('glist.txt', g_list, fmt='%s')

    start = time.time()
    dst_dir = 'jpeg'
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    j_list = list()
    wsi_tiles = read_wsi()
    for i, fn in enumerate(fn_list):
        region = wsi_tiles[i]
        export_fn = os.path.join(dst_dir, fn[7:len(fn) - 4] + '.jpg')
        j_list.append(export_fn)
        cv.imwrite(export_fn, region, [int(cv.IMWRITE_JPEG_QUALITY), 90])
    stop = time.time()
    j_total = stop - start
    print('Total generic JPEG compression time')
    print(j_total)

    # Save list of file for generic JPEG compression
    np.savetxt('jlist.txt', j_list, fmt='%s')

    return g_total, j_total


def lossless_compression_png_zlib():
    # Read the PNG files as images
    wsi_tiles = read_wsi()

    # Read the PNG files as bytes
    start = time.time()
    dst_dir = 'png'
    ori_data = list()
    for fn in tqdm(fn_list):
        loop_fn = os.path.join(dst_dir, fn[7:len(fn) - 4] + '.png')
        original_data = open(loop_fn, 'rb').read()
        ori_data.append(original_data)
    stop = time.time()
    png_time_total = stop - start
    print('Total PNG read time')
    print(png_time_total)

    start = time.time()
    dst_dir = 'zlib'
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    z_list = list()
    for i, fn in enumerate(fn_list):
        original_data = ori_data[i]
        compressed_data = zlib.compress(original_data, zlib.Z_BEST_COMPRESSION)
        compress_ratio = (float(len(original_data)) - float(len(compressed_data))) / float(len(original_data))

        if compress_ratio > 0:
            out_dir = os.path.join(dst_dir, fn[7:len(fn) - 4] + '.dat')
            z_list.append(out_dir)  # append to the list
            f = open(out_dir, 'wb')
            f.write(compressed_data)
            f.close()
        else:
            z_list.append(out_dir)
            out_dir = os.path.join(dst_dir, fn[7:len(fn) - 4] + '.png')
            cv.imwrite(out_dir, wsi_tiles[i], [cv.IMWRITE_PNG_COMPRESSION, 9])
    stop = time.time()
    total = stop - start
    print('Total generic ZLIB compression time: ' + str(total))

    # Save list of file for mixed compressed with zlib and PNG compression
    np.savetxt('zlist.txt', z_list, fmt='%s')


def main():
    fn_list = read_wsi_and_save_images()
    # Test read PNG
    wsi_tiles = read_wsi()
    read_time = read_and_time()

    # Test compress JPEG
    guetzli_time, jpeg_time = lossy_compression()

    # Test PNG and ZLIB compression
    lossless_compression_png_zlib()
    pass


if __name__ == '__main__':
    main()


