import cv2
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def find_nhood_pts(x, y, label, vec, delta):
    normal = np.array([-vec[1], vec[0]]) # normal to PAF at pt
    pt = np.array([x, y])

    # get all possible coordinate combinations
    X = np.arange(label.shape[1])
    Y = np.arange(label.shape[0])
    U, V = np.meshgrid(X, Y)
    coords = np.stack((U.ravel(), V.ravel()), axis=1)

    # get relative coordinates
    coords_diff = coords - pt

    # find pts that lie on the normal to the lane at pt
    idx = np.where(np.absolute(np.dot(vec, coords_diff.transpose())) <= 0.5)[0]
    coords = coords[idx, :]
    coords_diff = coords_diff[idx, :]

    # find pts that are within distance delta from the lane at pt
    idx = np.where(np.absolute(np.dot(normal, coords_diff.transpose())) <= delta)[0]
    coords = coords[idx, :]
    coords_diff = coords_diff[idx, :]

    return coords[:, 0], coords[:, 1]


def generatePAFs(label, delta=3.0, viz=False):
    # creating PAF array
    num_lanes = np.amax(label)
    PAF = np.zeros((label.shape[0], label.shape[1], 2))

    # generating PAFs
    for l in range(1, num_lanes + 1):
        x_lane, y_lane = [], []

        # lane thinning and preprocessing
        for row in range(label.shape[0]-1, -1, -1):
            cols = np.where(label[row, :] == l)[0] # get fg columns
            if cols.size == 0:
                continue
            col = int(round(np.mean(cols))) # calculate mean if multiple fg columns
            x_lane.append(col)
            y_lane.append(row)

        # cubic spline interpolation
        #cs = CubicSpline(np.array(y_lane), np.array(x_lane))
        #y_lane_interp = np.arange(label.shape[0] - 1, -1, -1)
        #x_lane_interp = cs(y_lane_interp)
        x_lane_interp, y_lane_interp = np.array(x_lane), np.array(y_lane)

        # fill in PAFs
        for idx in range(x_lane_interp.shape[0] - 1):
            # get direction vector
            vec = np.array([x_lane_interp[idx+1] - x_lane_interp[idx], 
                y_lane_interp[idx+1] - y_lane_interp[idx]])
            # unit normalize
            vec = vec / np.linalg.norm(vec)

            # set PAFs for all points within distance delta of line
            cols, rows = find_nhood_pts(x_lane_interp[idx], y_lane_interp[idx], label, vec, delta)
            for row, col in zip(rows, cols):
                PAF[row, col, 0] = vec[0]
                PAF[row, col, 1] = vec[1]

    if viz:
        down_rate = 4 # subsample visualization by this factor
        fig, ax = plt.subplots()
        q = ax.quiver(np.arange(0, label.shape[1], down_rate), -np.arange(0, label.shape[0], down_rate), 
            PAF[::down_rate, ::down_rate, 0], -PAF[::down_rate, ::down_rate, 1], scale=120)
        plt.show()

    return PAF


label = cv2.imread('test.png')
generatedPAFs = generatePAFs(label[:, :, 0], delta=1.0, viz=True)