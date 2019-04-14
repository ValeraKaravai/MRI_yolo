import os
import numpy as np
from PIL import Image

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

## Settings seaborn

rc = {"lines.linewidth": 2,
      'lines.markersize': 7}
sns.set_style(style="darkgrid",
              rc=rc)
sns.set_context("paper",
                rc=rc,
                font_scale=1.3)


def draw_polygon(x, color_patch, linestyle):
    '''

    :param x: Series (row) in  DataFrame
    :type x: Series
    :param color_patch: match of class (label) and color
    :type color_patch: dict
    :param linestyle: linestyle
    :type linestyle: str
    :return: object of rectangle and label

    '''

    rect = patches.Rectangle((x['x'], x['y']),
                             height=x['height'],
                             width=x['width'],
                             linewidth=2,
                             linestyle=linestyle,
                             edgecolor=color_patch[str(x['type_disk'])],
                             facecolor='none')
    return x['label'], rect, x['type_disk']


def draw_img(df, img_name, types_df, dir_img, color_patch, figsize,
             numsubplots):
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
    fig, ax = plt.subplots(ncols=numsubplots,
                           figsize=figsize)

    type_disk = set()

    if numsubplots == 1:
        ax = [ax]

    legend_patches = []
    ax_i = 0
    for i in range(len(df)):

        ax[ax_i].imshow(im_np)

        ax[ax_i].set_title('{} {}'.format(types_df[ax_i],
                                          img_name))

        subset_df = df[i][df[i]['file'] == img_name]
        if subset_df.shape[0] == 0:
            continue

        linestyle = '-' if types_df[i] == 'origin' else '--'
        rect = subset_df.apply(draw_polygon,
                               color_patch=color_patch,
                               linestyle=linestyle,
                               axis=1)
        for j in rect:
            ax[ax_i].add_patch(j[1])
            type_disk.add(j[2])

        for j in type_disk:
            legend_patches.append(patches.Patch(color=color_patch[str(j)],
                                                label=j + '_' + types_df[i],
                                                linestyle=linestyle,
                                                fill=False))
        if numsubplots > 1:

            ax[ax_i].legend(handles=legend_patches,
                            ncol=1,
                            loc='upper left')
            ax_i += 1
            legend_patches = []

        elif len(df) == (i + 1):
            ax[ax_i].legend(handles=legend_patches,
                            ncol=len(df),
                            loc='upper left')

    plt.show()

    return


def draw_metrics(x, y, hue, data, figsize, title):
    n_colors = len(data[hue].unique())
    palette = sns.color_palette("Set1",
                                n_colors=n_colors,
                                desat=.5)

    plt.figure(figsize=figsize)
    plt.title(title)
    sns.lineplot(x=x,
                 y=y,
                 data=data,
                 hue=hue,
                 marker='o',
                 palette=palette)
