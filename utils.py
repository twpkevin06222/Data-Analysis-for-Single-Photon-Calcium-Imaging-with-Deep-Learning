import os
import sys
import time
import glob
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
# from skimage.external import tifffile as sktiff
import skimage.color
import skimage.io
from contextlib import suppress
import tensorflow as tf

def min_max_norm(images, norm_axis = 'channel_wise'):
    """
    Min max normalization of images
    Parameters:
        images: Input stacked image list
        norm_axis: axis where the normalize should be computed,
            'channel_wise': min max norm along the channel
            'frame_wise': min max norm frame wise
    Return:
        Image list after min max normalization
    """
    assert norm_axis=='channel_wise' or norm_axis=='frame_wise',\
    "Please input 'channel_wise' or 'frame_wise'"
    if norm_axis == 'channel_wise':
        m = np.max(images) #max val along the channel
        mi = np.min(images) #min val along the channel
        output = (images - mi)/ (m - mi)
    elif norm_axis == 'frame_wise':
        #tile the tensor with respect to input image
        # so that the substaction with max and min val can be broadcasted
        tile_coef = tf.constant([1,100,100,1], tf.int32)
        #tile max
        #reduce max val along the axis 1 & 2
        #(images.shape[0], 1, 1,1) images.shape[0]=>max val per frames
        max_tensor = tf.reshape(tf.math.reduce_max(tf.math.reduce_max(images, 1),1), (-1,1,1,1))
        tile_max = tf.tile(max_tensor, tile_coef)
        #tile min
        #reduce min val along the axis 1 & 2
        #(images.shape[0], 1, 1,1) images.shape[0]=>min val per frames
        min_tensor = tf.reshape(tf.math.reduce_min(tf.math.reduce_min(images, 1),1), (-1,1,1,1))
        tile_min = tf.tile(min_tensor, tile_coef)
        #compute min max frame wise
        output = (images-tile_min)/(tile_max-tile_min)
    return output


def resize(img_list, NEW_SIZE, interpolation=cv2.INTER_LINEAR):
    """
    Resize image
    Parameter:
        image list, new size for image
    Return:
        resize image list
    """
    new_img_list = []

    for img in img_list:
        new_img = cv2.resize(img, (NEW_SIZE, NEW_SIZE), interpolation=interpolation)

        new_img_list.append(new_img)

    return np.asarray(new_img_list)

def tiff(dir_path):
    '''
    Read .tif extension

    :param dir_path: directory path where data is stored
    :return:
        shape of the particular tif file, arrays of the tif file
    '''
    im = sktiff.imread(dir_path)
    return im.shape, im

def append_tiff(path, verbose = True, timer = False):
    '''
    Append tiff image from path

    :param path: data directory
    :param verbose: output directory info
    :param timer: time measurement
    :return:
        list of tiff images, list of directories of tiff images
    '''
    start = time.time()

    dir_list = []
    image_stack = []
    for main_dir in sorted(os.listdir(path)):
        if verbose:
            print('Directory of mice index:', main_dir)
            print('Directory of .tif files stored:')

        merge_dir = os.path.join(path + main_dir)

        for file in sorted(os.listdir(merge_dir)):
            tif = glob.glob('{}/*.tif'.format(os.path.join(merge_dir + '/' + file)))

            shape, im = tiff(tif)
            dir_list.append(main_dir + '/' + file)
            image_stack.append(im)

            if verbose:
                print('{}, {}'.format(tif, shape))

    images = np.asarray(image_stack)
    end = time.time()

    if timer == True:
        print('Total time elapsed: ', end - start)

    return images, dir_list


def mat_2_npy(input_path, save_path):
    '''
    convert arrays in .mat to numpy array .npy

    input_path: path where data files of LIN is store, no need on specific path of .mat!
                input path must be located at Desktop!
    save_path: where .npy is save
    '''
    for main_dir in sorted(os.listdir(input_path)):
        print('Directory of mice index:',main_dir)
        merge_dir = os.path.join(input_path + main_dir)

        print('Directory of .mat files stored:')
        print()
        for file in sorted(os.listdir(merge_dir )):
            mat_list = glob.glob('{}/*.mat'.format(os.path.join(merge_dir + '/'+ file)))
            for mat in mat_list:

                print(mat)
                #obtain file name .mat for new file name during the conversion
                mat_dir_split = mat.split(os.sep)
                mat_name = mat_dir_split[-1]
                #print(mat_name)
                date_dir_split = file.split(os.sep)
                date_name = date_dir_split[-1]
                #print('{}_{}'.format(date_name, mat_name))
                
                #returns dict
                with suppress(Exception): #ignore exception caused by preprocessedFvid.mat
                    data = scipy.io.loadmat(mat)
                    for i in data:
                        if '__' not in i and 'readme' not in i:
                            print(data[i].shape)
                            
                            save_file = (save_path + date_name + '/')
                            if not os.path.exists(save_file):
                                os.makedirs(save_file)

                            #save matlab arrays into .npy file
                            np.save(save_file + "{}_{}_{}.npy".format(date_name, mat_name, i), data[i])
            
            print()


def vid_2_frames(vid_path, output_path, extension='.jpg', verbose = False):
    '''
    Converting video to image sequences with specified extension

    Params:
    vid_path: Path where video is stored
    output_path: Path where the converted image should be stored
    extension: Desired image extension, by DEFAULT .jpg
    verbose: Print progress of image creating
    Example:
        vid_path = '7-12-17-preprocessed.avi'
        output_path = retrieve_filename(vid_path)

        vid_2_frames(vid_path, '/' + output_path, extension = '.jpg', verbose = True)

    Return:
        >> For:  7-12-17-preprocessed.avi

        >> Creating..../7-12-17-preprocessed/frame_0000.jpg
        >> Creating..../7-12-17-preprocessed/frame_0001.jpg
                ...
    '''
    # Read the video from specified path
    cam = cv2.VideoCapture(vid_path)

    try:

        # creating a folder named output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

            # if not created then raise error
    except OSError:
        print('Error: Creating directory of output path')

        # frame
    currentframe = 0

    print('For: ', vid_path)
    print()

    while (True):

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            # name = ('./'+ output_path +'/frame_' + str(currentframe) + extension)

            name = ('{}/frame_{:04d}{}').format(output_path, currentframe, extension)
            if verbose:
                print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


def retrieve_filename(file_path):
    '''
    Retrieve file name from path and remove file extension

    Example:
        file_path = 'home/user/Desktop/test.txt'
        retrieve_filename(file_path)
    Return:
        >> test
    '''
    base_name = os.path.basename(file_path)

    # extract base name without extension
    base_name = os.path.splitext(base_name)[0]

    # print(base_name)

    return base_name


def vid2frames_from_files(input_path, save_path):
    '''
    Extension of vid_2_frames, which extract .avi from files
    :param input_path: Directory where all the .avi files is stored
    :param save_path:  Specify safe path
    '''
    for main_dir in sorted(os.listdir(input_path)):
        print('Directory of mice index:', main_dir)
        merge_dir = os.path.join(input_path + main_dir)

        print('Directory of .avi files stored:')
        print()
        for file in sorted(os.listdir(merge_dir)):
            avi_list = glob.glob('{}/*.avi'.format(os.path.join(merge_dir + '/' + file)))
            for avi in avi_list:
                #print(avi)
                # obtain file name .mat for new file name during the conversion
                avi_dir_split = avi.split(os.sep)
                avi_name = avi_dir_split[-1]
                # print(avi_name)
                date_dir_split = file.split(os.sep)
                date_name = date_dir_split[-1]
                # print('{}_{}'.format(date_name, avi_name))

                vid_name = retrieve_filename(avi)
                save_dir = (save_path + '{}_{}_{}'.format(main_dir,date_name, vid_name))
                vid_2_frames(avi, save_dir, extension='.jpg')

    print()

def img_to_array(inp_img, RGB=True):
    '''
    Convert single image from RGB or from Grayscale to array

    Params:
    inp_img: Desire image to convert to array
    RGB: Convert RGB image to grayscale if FALSE
    '''
    if RGB:
        return skimage.io.imread(inp_img)
    else:
        img = skimage.io.imread(inp_img)
        grayscale = skimage.color.rgb2gray(img)

        return grayscale


def imgs_to_arrays(inp_imgs, extension='.jpg', RGB=True, save_as_npy=False, img_resize = None, save_path=None):

    '''
    Convert image stacks from RGB or from Grayscale to array

    Params:
    inp_imgs: Desire image stacks to convert to array
    extension: input images extension, by DEFAULT '.jpg'
    RGB: Convert RGB image to grayscale if FALSE
    save_as_npy: Save as .npy extension
    save_path: Specify save path
    '''
    if img_resize != None:
        IMG_SIZE = img_resize

    imgs_list = []
    for imgs in sorted(glob.glob('{}/*{}'.format(inp_imgs, extension))):
        img_array = img_to_array(imgs, RGB)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        imgs_list.append(img_array)

    imgs_list = np.asarray(imgs_list)

    if save_as_npy:
        assert save_path != None, "Save path not specified!"
        # by default
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_name = retrieve_filename(inp_imgs)
        np.save(save_path + '{}.npy'.format(save_name), imgs_list)

    return imgs_list


def masked_img(mean_imgs, mean_roi):
    '''
    Plot masked image of an input mean image
    '''

    # operations require dtype = uint8 for bitwise comparison
    scr1 = (mean_imgs * 255).astype(np.uint8)  # scr image needs to be int(0,250)
    scr2 = mean_roi  # mask image needs to be float (0,1)
    masked_output = scr1 * scr2

    return masked_output.astype(np.uint8)

def dice_coef_py(y_true, y_pred):
    '''
    Dice coefficient for numpy
    '''
    eps = 1e-07
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + eps) /(np.sum(y_true_f) + np.sum(y_pred_f) + eps)


def retrieve_centroid(inp_img, centroid_rad=3):
    '''
    Estimate centroid from contour and plot centroids on mask image

    Parameters:
        inp_img: binarized input image
        centroid_rad: specify centroid radius during plot by DEFAULT 3
    Return:
        centres list and img with centroids
    '''
    assert inp_img.max() == 1.0, "Image not binarized!"

    # image needs to be binarized and of type int!
    cast_img = (inp_img).astype(np.uint8)
    print('Shape:{}, Min:{}, Max:{}, Type:{}'.format(cast_img.shape, cast_img.min(),
                                                     cast_img.max(), cast_img.dtype))
    contours, a = cv2.findContours(cast_img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    print('Number of detected ROIs:', len(contours))
    centres = []
    for i in range(len(contours)):
        moments = cv2.moments(contours[i])
        centres.append((int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])))
        # cv2.circle(img, (x,y), radius, (b, g, r), -1)
        img_with_centroids = cv2.circle(cast_img, centres[-1], centroid_rad, (0, 0, 0), -1)

    return centres, img_with_centroids

def mean_image(imgs, img_size):
    '''
    :param imgs: Image list
    :param img_size: specify image size
    :return:
        Mean image of shape (img_size, img_size)
    '''
    sums = np.zeros((img_size, img_size))
    total_index = 0
    for i in range(len(imgs)):
        sums += np.squeeze(imgs[i])
        total_index += 1

    mean_img_ori = sums / total_index

    return mean_img_ori


def MSE_image(img1, img2, IMG_SIZE):
    '''
    :param img1: True image
    :param img2: Predicted image
    :return:
        Mean squared error of two images
    '''
    img1, img2 = np.squeeze(img1), np.squeeze(img2)
    sq_error = (img1 - img2) ** 2
    sums = np.sum(sq_error)

    return sums / (IMG_SIZE * IMG_SIZE)


def MAE_image(img1, img2, IMG_SIZE):
    '''
    :param img1: True image
    :param img2: Predicted image
    :return:
        Mean absoluate error of two images
    '''
    img1, img2 = np.squeeze(img1), np.squeeze(img2)
    ab_error = np.abs(img1 - img2)
    sums = np.sum(ab_error)

    return sums / (IMG_SIZE * IMG_SIZE)


def max_in_pro(img_stacks, n_imgs, n_rows, n_cols, norm=False):
    '''
    Calculate the maximum intensity projection of image stacks
    (not optimized for tensorflow!)
    '''
    pixel_flat = []
    mip = []
    std_dev = []
    # (i, j ,k) # of images, # of rows, # of cols
    for j in range(n_rows):
        for k in range(n_cols):
            for i in range(n_imgs):
                # print(i, j, k)
                if img_stacks.ndim == 4:
                    pixel_flat.append(img_stacks[i, j, k, :])
                else:
                    pixel_flat.append(img_stacks[i, j, k])

    # acts as max. window of size n_imgs and strides of n_imgs
    for n in range(n_cols * n_rows):
        start = n * n_imgs
        end = (start) + (n_imgs)
        # print(start, end)
        max_pixel = np.max(pixel_flat[start:end])
        mip.append(max_pixel)

        if norm:
            # print('Normalizing!')
            std_pixel = np.std(pixel_flat[start:end])
            std_dev.append(std_pixel)

    mip = np.asarray(mip)

    if norm:
        # print('Normalizing!')
        std_dev = np.asarray(std_dev)
        # mip /= std_dev
        mip = np.multiply(mip, std_dev)  # weight by std.dev

    mip_re = np.reshape(mip, (n_rows, n_cols))

    return np.expand_dims(mip_re, -1)


def batch_dataset(inp_imgs, BATCH_SIZE, IMG_SIZE):
    '''
    Custom function for creating mini-batch of dataset
    :param inp_imgs: Input image list
    :param BATCH_SIZE: batch size
    :param IMG_SIZE: input image size
    :return:
        Batched dataset of dimension (n_batch, BATCH_SIZE, IMG_SIZE, IMG_SIZE, channel)
    '''
    n_batch = int(len(inp_imgs) / BATCH_SIZE)
    mod = len(inp_imgs) % BATCH_SIZE
    if mod == 0:
        batch_imgs = np.reshape(inp_imgs, (n_batch, BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1)).astype('float32')
    else:
        # divisible part
        divisible = inp_imgs[:(len(inp_imgs) - mod)]
        divisible_re = np.reshape(divisible, (n_batch, BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1))

        # remainder part
        remainder = inp_imgs[(len(inp_imgs) - mod):]
        # remainder shape must be padded to be the same as divisible shape
        # else python will return array of type "object" which tensorflow
        # cannot convert it to tensor
        pad_dim = int(BATCH_SIZE - mod)
        pad_array = np.zeros((pad_dim, IMG_SIZE, IMG_SIZE, 1))
        remainder_pad = np.concatenate((remainder, pad_array), axis=0)
        # normalize trick for remainder to balance the mean of zeros array padding
        # such that in tf.reduce_mean, mean of remainder_pad = remainder_pad/BATCH_SIZE
        # which in this case, the true mean becomes remainder_pad/len(remainder)
        remainder_pad *= (BATCH_SIZE / len(remainder))
        remainder_pad = np.expand_dims(remainder_pad, 0)

        # stack divisible and remainder
        batch_imgs = np.concatenate((divisible_re, remainder_pad), 0).astype('float32')

    return batch_imgs

def stack4plot(one_hot_imgs):
    '''
    Functions to sum all one hot images along axis=0 for easy plot
    '''
    return tf.squeeze(tf.reduce_sum(one_hot_imgs, axis = 0))


# def similarity_multi(n_neurons, one_hot_imgs, similarity_score, img_size):
#     '''
#     @param n_neurons: number of neurons
#     @param one_hot_imgs: one hot images generated by deconve model (100,100,1)
#     @param similarity_scores: similarity scores after dot product
#     @param img_size: image size
#
#     This function multiply the similarity scores with the one hot image generate by a particular
#     coordinate
#
#     return:
#     the sum of all the one hot image activations along the last channel
#     '''
#     stack_imgs = np.zeros((img_size, img_size))
#     for idx in range(n_neurons):
#         activations = similarity_score[idx] * np.squeeze(one_hot_imgs[idx])
#         stack_imgs += activations
#
#     return stack_imgs  # (batch_size, img_size, img_size)
def similarity_multi(one_hot_imgs, similarity_score, thr=None):
    '''
    @param one_hot_imgs: one hot images generated by decoord-conv model (100,100,1) #(n_neurons, img_size, img_size, 1)
    @param similarity_scores: similarity scores after dot product #(batch_size, n_neurons)
    @param thr: threshold for sim scores multipied one hot pixel

    This function multiply the similarity scores with the one hot image generate by a particular
    coordinate

    return:
    the sum of all the one hot image activations along the last channel
    '''
    onehot_multi_sim = tf.einsum('ij,jklm->ijklm', similarity_score, one_hot_imgs) #(batch_size, n_neurons, img_size, img_size, 1)
    onehot_multi_sim = tf.squeeze(tf.reduce_sum(onehot_multi_sim, axis=1))
    if thr=='mean':
        ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for i in tf.range(len(onehot_multi_sim)):
            ta = ta.write(i, tf.where(onehot_multi_sim[i]<tf.math.reduce_mean(onehot_multi_sim[i]),0.0,onehot_multi_sim[i]))
        onehot_multi_sim = tf.convert_to_tensor(ta.stack())
    elif type(thr)==float:
        onehot_multi_sim = tf.where(onehot_multi_sim<thr,0.0,onehot_multi_sim)
    return  onehot_multi_sim# (batch_size, img_size, img_size)

def concat_recursive(a, b, max_count, count):
    '''
    Recursively concatenate the image stacks with the next image stacks
    @param a: Top first image stacks
    @param b: Following image stacks
    '''
    if count < max_count - 1:
        if (count == 0):
            c = np.concatenate((a[count], b[count + 1]), axis=0)
        else:
            c = np.concatenate((a, b[count + 1]), axis=0)
        a = c
        count += 1
        return concat_recursive(a, b, max_count, count)
    if count == max_count - 1:
        return a

def concat_batch(stack_batch_imgs):
    if tf.rank(tf.convert_to_tensor(stack_batch_imgs[0]))>=3:
        stack_list = []
        for i in range(len(stack_batch_imgs)):
            slices = stack_batch_imgs[i]
            slices = tf.convert_to_tensor(slices, tf.float32)
            concat_imgs = concat_recursive(slices, slices, len(slices), 0)
            stack_list.append(concat_imgs)
        return stack_list
    else:
        stack_batch_imgs = tf.convert_to_tensor(stack_batch_imgs, tf.float32)
        concat_imgs = concat_recursive(stack_batch_imgs, stack_batch_imgs, len(stack_batch_imgs), 0)
        return concat_imgs

def similarity_multiplication(similarity_list_npy, one_hot_imgs_list_npy, n_neurons, epoch_pos, img_size, threshold):
    stack_batch_imgs = []
    stack_batch_imgs_thr = []
    for batch_similarity in similarity_list_npy[epoch_pos]:
        stack_imgs = np.zeros((img_size,img_size))
        for idx in range(n_neurons):
            test = batch_similarity[idx]*np.squeeze(one_hot_imgs_list_npy[epoch_pos, idx])
            stack_imgs+=test
        stack_imgs_thr = np.where(stack_imgs<threshold, 0.0, 1.0)
        stack_batch_imgs.append(stack_imgs)
        stack_batch_imgs_thr.append(stack_imgs_thr)
    return np.array(stack_batch_imgs), np.array(stack_batch_imgs_thr) #(batch_size, img_size, img_size)