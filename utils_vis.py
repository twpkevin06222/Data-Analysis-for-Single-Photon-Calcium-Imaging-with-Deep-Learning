import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model, Sequential
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import tensorflow as tf
import cv2



def plot_loss(loss_list, xlabel, ylabel, title, recon_list=None):
    '''
    :param loss_list: List containing total loss values
    :param recon_list: List containing reconstruction loss
    :param xlabel: string for xlabel
    :param ylabel: string for ylabel
    :param title: string for title

    :return: loss value plot
    '''
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(linestyle='dotted')
    plt.plot(loss_list)


def plot_comparison(input_img, caption, n_row=1, n_col=2, figsize=(5, 5), cmap = 'gray', norm = None):
    '''
    Plot comparison of multiple image but only in column wise!
    :param input_img: Input image list
    :param caption: Input caption list
    :param IMG_SIZE: Image size
    :param n_row: Number of row is 1 by DEFAULT
    :param n_col: Number of columns
    :param figsize: Figure size during plotting
    :return: Plot of (n_row, n_col)
    '''
    print()
    assert len(caption) == len(input_img), "Caption length and input image length does not match"
    assert len(input_img) == n_col, "Error of input images or number of columns!"

    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, right=0.7)

    for i in range(n_col):
        axes[i].imshow(np.squeeze(input_img[i]), cmap= cmap, norm=norm)
        axes[i].set_xlabel(caption[i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_hist(inp_img, titles, n_row=1, n_col=2,
              n_bin=20, ranges=[0, 1], figsize=(5, 5)):
    '''
    Plot histogram side by side
    :param inp_img: Input image stacks as list
    :param titles: Input titles as list
    :param n_row: Number of row by DEFAULT 1
    :param n_col: Number of columns by DEFAULT 2
    :param n_bin: Number of bins by DEFAULT 20
    :param ranges: Range of pixel values by DEFAULT [0,1]
    :param figsize: Figure size while plotting by DEFAULT (5,5)
    :return:
        Plot of histograms
    '''
    assert len(titles) == len(inp_img), "Caption length and input image length does not match"
    assert len(inp_img) == n_col, "Error of input images or number of columns!"

    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, right=0.7)

    for i in range(n_col):
        inp = np.squeeze(inp_img[i])
        axes[i].hist(inp.ravel(), n_bin, ranges)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Pixel Value')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def layers_dict(model):
    '''
    :param model: deep learning model

    :return:
        Dictionary with 'key': layer names, value: layer information
    '''
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    return layer_dict

def layers_name(model):
    '''
    Retrieve key/name of layers
    :param model: Network model
    :return:
        Layers name list
    '''
    layer_dict = layers_dict(model)
    key_list = []
    for key, value in layer_dict.items():
        key_list.append(key)
        print(key)
    return key_list


def feature_maps(model, layer_name, inps):
    '''
    This function visualize the intermediate activations of the filters within the layers
    :param model: deep learning model
    :param layer_name: desired layer name, if forgotten, please refer to layers_dict function
    :param inps: feed the network with input, such as images, etc. input dimension
                 should be 4.

    :return:
        feature maps of the layer specified by layer name,
        with dimension ( batch, row size, column size, channels)
    '''
    assert inps.ndim == 4, "Input tensor dimension not equal to 4!"
    # retrieve key value from layers_dict
    layer_dict = layers_dict(model)

    # layer output with respect to the layer name
    layer_output = layer_dict[layer_name].output
    viz_model = Model(inputs=model.inputs, outputs=layer_output)
    feature_maps = viz_model.predict(inps)

    print('Shape of feature maps:', feature_maps.shape)
    # shape (batch, row size, column size, channels)
    return feature_maps


def plot_feature_maps(inps, row_num, col_num, figsize):
    '''
    This function can only plot the feature maps of a model
    :param inps: feature maps
    :param row_num: number of rows for the plot
    :param col_num: number of columns for the plot

    :return:
        grid plot of size (row_num * col_num)
    '''
    assert inps.ndim == 4, "Input tensor dimension not equal to 4!"

    print("Number of feature maps in layer: ", inps.shape[-1])

    fig, axes = plt.subplots(row_num, col_num, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, right=0.7)

    for i, ax in enumerate(axes.flat):
        img = inps[0, :, :, i]

        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

def overlapMasks(mask_truth, mask_predicted, threshold = 0):
    '''
    This function can only plot the feature maps of a model
    :param mask_predicted: prediction
    :param mask_truth: ground truth
    :threshold: threshold for predicted mask
    :return:
        Returns overlapping image of prediction and ground truth
    '''
    col = [(0.2, 0.2, 0.2),(1,1,1),(1,0,0)]
    cm = LinearSegmentedColormap.from_list('mylist', col, 3)

    #Bins for cmap
    bounds=[0,1,5,10]
    norm = colors.BoundaryNorm(bounds, cm.N)

    # mask_predicted = mask_predicted.numpy()
    mask_predicted[mask_predicted > threshold] = 5

    Image_mask = np.add(mask_truth, mask_predicted)

    plt.imshow(Image_mask, cmap=cm, norm=norm)


def overlapMasks02(mask_truth, mask_predicted):
    '''
    This function can only plot the feature maps of a model
    :param mask_predicted: prediction
    :param mask_truth: ground truth

    :return:
        Returns overlapping image of prediction and ground truth

    :extra param:
    (copy this and use this as variable!)
    from matplotlib.colors import BoundaryNorm

    col = [(0.2, 0.2, 0.2),(1,1,1),(1,0,0)]
    cm = LinearSegmentedColormap.from_list('mylist', col, 3)
    #     #Bins for cmap
    bounds=[0,1,5,10]
    norm = BoundaryNorm(bounds, cm.N)
    '''

    mask_predicted = tf.convert_to_tensor(mask_predicted, tf.float32).numpy()
    mask_predicted[mask_predicted > 0] = 5

    Image_mask = np.add(mask_truth, mask_predicted)

    return Image_mask

