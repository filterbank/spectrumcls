#-*- coding: UTF-8 -*-

import os
import numpy as np
import tensorflow as tf
from PIL import Image

def file_path(filedir):
    file_names = os.listdir(filedir)
    if len(file_names):
        print('Succesfully read file names in dirctory of %s' % filedir)
        #print('The number of images is  %s' % len(file_names))
    else:
        ValueError('%s is a empty directory' % filedir)
    filepath_list=[]
    for filename in file_names:
        filepath = os.path.join(filedir, filename)
        filepath_list.append(filepath)
    return filepath_list

def read_int16(bytestream): #read 64 bytes,
   dt = np.dtype(np.int16).newbyteorder('<')
   return np.frombuffer(bytestream.read(64), dtype=dt)

def read_file(filename,sizebytes = None):
    #read datas of image ,and discard the first 32 int16 (64 bytes)
    # reurn a 1-D array
    # sizebytes = None:indicating read data until EOF
    with open(filename,'rb') as bytestream:
        discard32int16 = read_int16(bytestream) # discard the first 32 int16 (64 bytes)
        if len(discard32int16) != 32:
            ValueError('%s is a empty file' % filename)
        else:
            print ('Succesfully discard the first 32 int16 of %s' % filename)
        if sizebytes is None: # read 'sizebytes'bytes in 'filename'
            sizebytes = -1   # sizebytes == -1 ,read until EOF is reached
        dt = np.dtype(np.int16).newbyteorder('<')
        data = np.frombuffer(bytestream.read(sizebytes), dtype=dt)
        if len(data) != 604800: # the size of a image file is 604800 int16
            ValueError('The image has been damaged')
        else:
            print ('Succesfully read effective data of the image' )
    return data

def split_image(imagedata):
    # split a image into two images
    # imagedata : 1-D array
    # split princple : Every 120 datas alternately distribute leftimage
    # and rightimage as their column
    #return two 2-D array ,size is [120,2520]
    data = imagedata.reshape(-1, 120)
    i = 1
    leftimage = []
    rightimage = []
    for row in data:
        if i%2 == 1:
            leftimage.append(row)
        else:
            rightimage.append(row)
        i += 1  #leftimage = np.array(leftimage) # transform list to array
    #rightimage = np.array(rightimage)
    leftimage = np.transpose(leftimage)#  in order to transform their column by Permuting
    rightimage = np.transpose(rightimage)
    print ('Succesfully split image into leftimage and rightimage')
    return leftimage,rightimage

def channel_denoising_image(imagedata):
    # eliminate the channel effect
    # 'imagedata':2-D array
    # use method:f = g-rowmean + globlemean
    # g indicates 'imagedata','rowmean'indicates mean of each row in 'imagedata'
    # 'globlemean'indicates mean of whole 2-D array 'imagedata'
    #meanvalue = imagedata.mean()
    channelmeanvalue = imagedata.min(1)
    i=0
    while i<120:
        #imagedata[i] = imagedata[i]-channelmeanvalue[i] + meanvalue
        imagedata[i] = imagedata[i]-channelmeanvalue[i]
        i = i+1
    print ('Succesfully eliminate the channel effect')
    return imagedata

def compress_image(imagedata):
    # compress image data from [120,2520] to[120,120]
    # use method : mean fliter ,window size is 21,weigh value as follow 'mean_window'
    mean_window = np.array([0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05,
                            0.25, 0.25, 0.25, 0.25,
                            0.0125,0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125])
    #mean_window = np.array([0, 0, 0, 0, 0, 0, 0, 0.05, 0.1, 0.2, 0.4, 0.2, 0.1, 0.05, 0, 0, 0, 0, 0, 0, 0])
    data = imagedata.reshape(-1, 21)
    newdata = np.dot(data,mean_window)
    newdata = newdata.reshape(-1, 120)
    print ('Succesfully compress image')
    return newdata

def normalization_image(imagedata):
    # normalize data to 0~1 for every element of 2-D array 'imagedata'
    maxvalue = imagedata.max()
    minvalue = imagedata.min()
    data = np.multiply((imagedata - minvalue), 1.0/(maxvalue - minvalue))
    print ('Succesfully normalize image')
    return data


def save_image(wfilenarray,savefilepath,mode='ab'):
    # write 'wfilenarray' to 'savefilepath' in the mode of 'a'
    # 'wfilenarray' : wrote content
    # 'savefilepath':saved dirctory+filennme
    # 'mode='a'':append mode
    if not os.path.exists(savefilepath):
        os.mknod(savefilepath)
    if not os.path.isfile(savefilepath):
        raise ValueError('Input must be a file,but %s is a directory ' % savefilepath)
    filenarray = wfilenarray * 255   # 'wfilenarray'is normalized ,so it =<1,wfilenarray * 255<255
    filenarray = filenarray.astype(np.uint8)#so transform  'wfilenarray'of float32 into np.uint8
    #print filenarray ,'\n','\n'
    with open(savefilepath,mode) as bytestream:
        bytestream.write(filenarray)
    print ('Succesfully save as %s'% savefilepath)
    return savefilepath


def preprocess_image(filedir,savefilepath,eachfilesizebytes = None):
    filepath = file_path(filedir)
    for filename in filepath:
        imagedata = read_file(filename, eachfilesizebytes)
        leftimage, rightimage = split_image(imagedata)
        #print type(rightimage),type(leftimage),'\n',leftimage,'\n','\n',rightimage,'\n','\n','\n'

        #print leftimage.mean(),'\n','\n',leftimage.mean(1),'\n','\n','\n'
        #print rightimage.mean(),'\n','\n',rightimage.mean(1),'\n','\n','\n'
        denoiseleftimage = channel_denoising_image(leftimage)
        denoiserightimage = channel_denoising_image(rightimage)
        #print denoiseleftimage,'\n','\n',denoiserightimage, '\n','\n','\n'

        compressleftimage = compress_image(denoiseleftimage)
        compressrightimage = compress_image(denoiserightimage)
        #print compressleftimage,'\n','\n',compressrightimage,'\n','\n','\n'

        #print compressleftimage.max(), compressrightimage.min()
        normalleftimage = normalization_image(compressleftimage)
        normalrightimage = normalization_image(compressrightimage)
        #print normalleftimage,'\n','\n',normalrightimage,'\n','\n','\n'

        save_image(normalleftimage, savefilepath)
        save_image(normalrightimage, savefilepath)

    return savefilepath



def generate_lable(lable,number,filepath,mode='wb'):
    # 'lable':0-9 uint8
    # 'number': The number of 'lable'
    # 'filepath': Save 'lable' in file named 'filepath'
    # return file name 'filepath'
    # mode:the mode of writing file
    if not os.path.exists(filepath):
        os.mknod(filepath)
    if not os.path.isfile(filepath):
        raise ValueError('Input must be a file,but %s is a directory' % filepath)
    if not((lable >= 0) and (lable <10)):
        raise ValueError('Lable is out of range')
    lablearray = np.ones(number, dtype=np.uint8)*lable
    with open(filepath,mode) as bytestream:
        bytestream.write(lablearray)
    print ('Succesfully generate lables and save in %s'% filepath)
    return filepath


def read_data(filepath,sizebytes = None):
    # 'filepath': dirctory+filennme of read file
    # 'sizebytes'=None: read until EOF is reached
    # return 1-D array
    if not os.path.isfile(filepath):
        raise ValueError('Input must be a file,but %s is a directory ' % filepath)
    with open(filepath,'rb') as bytestream:
        if sizebytes is None: # read 'sizebytes'bytes in 'filename'
            sizebytes = -1   # sizebytes == -1 ,read until EOF is reached
        dt = np.dtype(np.uint8).newbyteorder('<')
        data = np.frombuffer(bytestream.read(sizebytes), dtype=dt)
    return data

def write_data(wfilenarray,filepath,mode='wb'):
    # write 'wfilenarray' to 'filepath' in the mode of 'w'
    # 'wfilenarray' : wrote content
    # 'filepath': dirctory+filennme of wrote file
    # 'mode='w'': write mode
    # return 'filepath'
    if not os.path.exists(filepath):
        os.mknod(filepath)
    if not os.path.isfile(filepath):
        raise ValueError('Input must be a file,but %s is a directory ' % filepath)
    with open(filepath,mode) as bytestream:
        bytestream.write(wfilenarray)
    print ('Succesfully save as %s'% filepath)
    return filepath

def split_dataset(datasetpath,splitdatabytes,splitdatapath,restdatapath):
    # split a file into two files
    # 'datasetpath': old file of dirtory + filename
    # 'splitdatapath': the first new file of dirtory + filename
    # 'restdatapath': the secong new file of dirtory + filename
    # 'splitdatabytes': the size(bytes) of 'splitdatapath'
    dataarray = read_data(datasetpath)
    shapesize = np.shape(dataarray)
    if(splitdatabytes < 0 or splitdatabytes > shapesize[0]):
        raise ValueError('%s is too small or too big' % splitdatabytes)
    splitdataarray = dataarray[:splitdatabytes]
    restdataarray = dataarray[splitdatabytes:]
    write_data(splitdataarray, splitdatapath, mode='wb')
    write_data(restdataarray, restdatapath, mode='wb')
    print ('Succesfully split dataset')
    return splitdatapath,restdatapath


def merge_dataset(mergefilepath,partfiledir,partfilenamelist):
    # merge several files in the same dirctory into a new file
    # 'mergefilepath' : a new file of dirctory + filename
    # 'partfiledir': the dirctory of several files
    # 'partfilenamelist' is a list ,and it save names of several files
    if not os.path.exists(mergefilepath):
        os.mknod(mergefilepath)
    if not os.path.isfile(mergefilepath):
        raise ValueError('Input must be a file,but %s is a directory' % mergefilepath)
    if not os.path.exists(partfiledir):
        raise ValueError('The directory of %s is not exists' % partfiledir)
    partfilepath = []
    for partfilename in  partfilenamelist:
        filepath = os.path.join(partfiledir, partfilename)
        if not os.path.isfile(filepath):
            raise ValueError('The file of %s is not exists' % partfilename)
        partfilepath.append(filepath)
    for filepath in partfilepath:
        dataarray = read_data(filepath)
        write_data(dataarray, mergefilepath, mode='ab')
    print ('Succesfully merge dataset')
    return mergefilepath


def pack_file(filepath,numberarray):
    # add several bytes of data to beginning of file in order to conveniently use it
    # 'filepath':the packing file of dirctory + filename
    # 'numberarray': 1-D array of adding data(Don't forget data type)
    # such as  numberarray = np.array([2051,2400,120,120],dtype = np.uint32)
    readdata = read_data(filepath)
    write_data(numberarray, filepath, mode='wb')
    write_data(readdata, filepath, mode='ab')
    return filepath

if __name__ == '__main__':
    preprocess_image(r'./brust',
                     r'./brust.txt')
    preprocess_image(r'./calibration',
                     r'./calibration.txt')
    preprocess_image(r'./non_brust',
                     r'./non_brust.txt')

    split_dataset(r'./non_brust.txt', 11520000,
                  r'./non_brust_train800.txt',
                  r'./non_brust_test5870.txt')
    split_dataset(r'./brust.txt', 11520000,
                  r'./brust_train800.txt',
                  r'./brust_test358.txt')
    split_dataset(r'./calibration.txt', 11520000,
                  r'./calibration_train800.txt',
                  r'./calibration_test188.txt')

    generate_lable(0, 800, r'./non_brust_train800_lable0.txt')
    generate_lable(0, 5870, r'./non_brust_test5870_lable0.txt')
    generate_lable(1, 800, r'./brust_train800_lable1.txt')
    generate_lable(1, 358, r'./brust_test358_lable1.txt')
    generate_lable(2, 800, r'./calibration_train800_lable2.txt')
    generate_lable(2, 188, r'./calibration_test188_lable2.txt')

    merge_dataset(r'./train210.txt',
                  r'./',
                  ['calibration_train800.txt','brust_train800.txt','non_brust_train800.txt'])

    merge_dataset(r'./test210.txt',
                  r'./',
                  ['calibration_test188.txt', 'brust_test358.txt','non_brust_test5870.txt'])

    merge_dataset(r'./trainlable210.txt',
                  r'./',
                  ['calibration_train800_lable2.txt','brust_train800_lable1.txt', 'non_brust_train800_lable0.txt',])

    merge_dataset(r'./testlable210.txt',
                  r'./',
                  ['calibration_test188_lable2.txt','brust_test358_lable1.txt','non_brust_test5870_lable0.txt'])


    pack_file(r'./train210.txt',np.array([2051,2400,120,120],dtype=np.uint32))
    pack_file(r'./test210.txt', np.array([2051,6416,120,120],dtype=np.uint32))
    pack_file(r'./trainlable210.txt',np.array([2049,2400],dtype=np.uint32))
    pack_file(r'./testlable210.txt', np.array([2049,6416],dtype=np.uint32))
