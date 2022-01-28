"""
Includes everything used for plotting.
"""
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from lib.preprocessing import radian_to_degree, degree_to_radian

def plot_interpolations(levelset_images, levelset_y, img1_rotated, img2_rotated, num, angle, dir):
    """
    Create plot for the interpolations
    :param levelset_images:
    :param img1_rotated:
    :param img2_rotated:
    :param num:
    :return:
    """
    fig = plt.figure(figsize=(15, 10))
    grid = ImageGrid(fig, 111, nrows_ncols=(4 + 2, 4), axes_pad=0.1)
    grid_idx = np.arange(len(grid)).reshape(4 + 2, 4)
    # levelset_images, levelset_y = model.decode(z[:, np.newaxis, :])
    images_levelset = tf.squeeze(levelset_images.mean()).numpy()
    # imgs = None
    for levelset_images, ax_idx in zip(images_levelset, sum([list(id) for id in grid_idx[1:-1, :]], [])):
        ax = grid[ax_idx]
        ax.imshow(levelset_images, cmap="gray")
    for column in range(4):
        for ax_idx in grid_idx[:, column]:
            grid[ax_idx].axis('off')
    grid[grid_idx[0, 0]].imshow(tf.squeeze(img2_rotated), cmap="gray")
    grid[grid_idx[-1, -1]].imshow(tf.squeeze(img1_rotated), cmap="gray")
    # grid[grid_idx[0, 1]].imshow(tf.squeeze(img2_rotated_rec), cmap="gray")
    # grid[grid_idx[-1, -2]].imshow(tf.squeeze(img1_rotated_rec), cmap="gray")

    plt.suptitle(f"Rotation: {np.squeeze(angle) * 180 / np.pi:.2f}°\n ")  # Start: {img1_idx}, End: {img2_idx}

    # plt.ion()
    fig.canvas.draw()

    for ax, col in zip([grid[grid_idx[0, 0]], grid[grid_idx[-1, -1]]], ["limegreen", "royalblue"]):
        bbox = ax.get_tightbbox(fig.canvas.get_renderer())
        x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
        # slightly increase the very tight bounds:
        xpad = 0.01 * width
        ypad = 0.01 * height
        fig.add_artist(
            plt.Rectangle((x0 - xpad, y0 - ypad), width + 2 * xpad, height + 2 * ypad, edgecolor=col, linewidth=7,
                          fill=False))

    plt.savefig(dir/f"interpolation_{num}.png", bbox_inches='tight')
    plt.close()


def plot_dataset_samples(train_dataset, dir):
    fig, axs = plt.subplots(8, 8, figsize=(15, 15))
    axs = axs.flatten()
    for i, (img, radian, img_class) in enumerate(train_dataset.take(len(axs))):
        axs[i].imshow(np.squeeze(img.numpy()), cmap="gray", vmin=0, vmax=1.)
        axs[i].set_title(f"{radian_to_degree(np.squeeze(radian.numpy())):.2f}°; {img_class}")
        axs[i].set_axis_off()
    plt.tight_layout()
    plt.savefig(dir / "rotated_fmnist.png")
    plt.close()


def plot_reconstructions(X_val, ib, val_data, title="reconstruction.png"):
    z_val = tf.concat([ib.q_z(X).mean() for X, *_ in val_data], 0)
    num_imgs_reconstruct = 10
    x_hat_test = tf.squeeze(ib.decoder_x(z_val[:num_imgs_reconstruct, ...]))
    fig = plt.figure(figsize=(15, 15))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, num_imgs_reconstruct),
                     axes_pad=0.1)
    for img, ax in zip(X_val[:10, ...].numpy(), grid):
        ax.imshow(img, cmap="gray")
    for img, ax in zip(x_hat_test.numpy(), grid[-10:]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    plt.savefig(f"{title}.png")
    plt.close()
    return z_val