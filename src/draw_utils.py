import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches


def draw_polygon(x, color_patch):

    '''

    :param x: Series (row) in  DataFrame
    :type x: Series
    :param color_patch: match of class (label) and color
    :type color_patch: dict
    :return: object of rectangle and label

    '''

    rect = patches.Rectangle((x['x'], x['y']),
                             height=x['height'],
                             width=x['width'],
                             linewidth=1,
                             edgecolor=color_patch[str(x['type_disk'])],
                             facecolor='none')
                             # label=x['type_disk'])
    return x['label'], rect, x['type_disk']


def draw_img(df, img_name, types_df, dir_img, color_patch, figsize):
    '''

    :param df: list of data frame with MRI data
    :type df: list of DataFrame
    :param img_name: name of image
    :type img_name: str
    :param dir_img: Name of image directory
    :type dir_img: str
    :param color_patch: match of class (label) and color
    :type color_patch: dict
    :return:

    '''

    path_img = os.path.join(dir_img, img_name)
    img = Image.open(path_img)
    im_np = np.array(img, dtype=np.uint8)
    num_subplots = len(df)
    fig, ax = plt.subplots(ncols=num_subplots,
                           figsize=figsize)

    type_disk = set()

    if num_subplots == 1:
        ax = [ax]
    for i in range(num_subplots):
        ax[i].imshow(im_np)
        
        subset_df = df[i][df[i]['file'] == img_name]
        if subset_df.shape[0] == 0:
            continue
        rect = subset_df.apply(draw_polygon, 
                               color_patch=color_patch,
                               axis=1)
        for j in rect:
            ax[i].add_patch(j[1])
            type_disk.add(j[2])

        ax[i].set_title('{} {}'.format(types_df[i],
                                       img_name))

    legend_patches = []
    for i in type_disk:
        legend_patches.append(patches.Patch(color=color_patch[str(i)],
                                            label=i,
                                            fill=False))

    plt.legend(handles=legend_patches, loc='upper left')

    plt.show()

    return
