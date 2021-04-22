from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    """
    Returns a n x d dataset (n: the number of images in the dataset and d: the dimensions of each image)
    Each row represents an image feature vector. For this particular dataset, we have n = 2414 (no. of images) and
    d=32 x 32=1024 (32 by 32 pixels).
    :param filename: name of the file to load
    :return: a n x d dataset (n: the number of images in the dataset and d: the dimensions of each image)
    """
    dataset = np.load(filename)
    mean_val = np.mean(dataset, axis=0)
    return dataset - mean_val


def get_covariance(dataset):
    """
    Gets the covariance matrix of a dataset
    :param dataset: the dataset to get the covariance matrix of
    :return: covariance matrix of the dataset
    """
    return 1 / (len(dataset) - 1) * np.dot(np.transpose(dataset), dataset)


def get_eig(S, m):
    """
    Returns the largest m eigenvalues of S, in descending order, and the corresponding eigenvectors as columns in a
    matrix.
    :param S: covariance matrix of the dataset
    :param m: how many eigenvalues to return
    :return: largest m eigenvalues of S as a diagonal matrix and the corresponding eigenvectors as columns in a matrix
    """
    val, vec = eigh(S, subset_by_index=[len(S) - m, len(S) - 1])
    return np.diag(np.flip(val, axis=0)), np.flip(vec, axis=1)


def get_eig_perc(S, perc):
    """
    Returns eigenvalues that explain more than a certain percentage of variance
    :param S: covariance matrix of the dataset
    :param perc: the percentage
    :return: diagonal matrix of eigenvalues and the eigenvectors in corresponding columns
    """
    val, vec = eigh(S)
    sum_val = sum(val)
    for i in range(len(val)):
        if val[i] / sum_val > perc:
            break
    val, vec = eigh(S, subset_by_index=[i, len(S) - 1])
    return np.diag(np.flip(val, axis=0)), np.flip(vec, axis=1)


def project_image(img, U):
    """
    Projects the image
    :param img: image to project
    :param U: eigenvectors
    :return: a projection of the image
    """
    return np.dot(np.dot(U, np.transpose(U)), img)


def display_image(orig, proj):
    """
    Displays the images
    :param orig: origin of image
    :param proj: projection of image
    :return: nothing
    """
    orig = np.reshape(orig, (-1, 32))
    proj = np.reshape(proj, (-1, 32))
    for x in range(3):
        orig = np.rot90(orig)
        proj = np.rot90(proj)
    orig = np.flip(orig, axis=1)
    proj = np.flip(proj, axis=1)
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.set_title("Original")
    ax2.set_title("Projection")
    orig_color = ax1.imshow(orig, aspect='equal')
    proj_color = ax2.imshow(proj, aspect='equal')
    fig.colorbar(orig_color, ax=ax1)
    fig.colorbar(proj_color, ax=ax2)
    plt.show()
