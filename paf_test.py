import cv2
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def assign_vecs(PAF, fg_pts_x, fg_pts_y, x_lane, y_lane, vec_lane):
    fg_pts = np.stack((fg_pts_x, fg_pts_y), axis=1).astype(np.float32)
    lane_pts = np.stack((x_lane, y_lane), axis=1).astype(np.float32)
    # calculate pairwise distance between lane pts and fg pts
    D = cdist(fg_pts, lane_pts)
    # find which lane pt is closest to each fg pt
    min_idx = np.argmin(D, axis=1)
    # assign direction vector of closest lane pt to each fg pt
    for i, idx in enumerate(min_idx):
        PAF[fg_pts_y[i], fg_pts_x[i], 0] = vec_lane[idx, 0]
        PAF[fg_pts_y[i], fg_pts_x[i], 1] = vec_lane[idx, 1]
    return PAF

def generatePAFs(label, viz=False):
    # creating PAF array
    num_lanes = np.amax(label)
    PAF = np.zeros((label.shape[0], label.shape[1], 2))

    # loop over each lane
    for l in range(1, num_lanes+1):
        fg_pts_y, fg_pts_x = np.where(label == l) # foreground pt locations
        if len(fg_pts_y) == 0:
            continue
        x_lane, y_lane, vec_lane = [], [], [] # mean lane locations and direction vectors

        # lane thinning and preprocessing
        for row in range(label.shape[0]-1, -1, -1):
            cols = np.where(label[row, :] == l)[0] # get fg columns
            if cols.size == 0:
                continue
            col = int(round(np.mean(cols))) # calculate mean if multiple fg columns

            if len(x_lane) != 0:
                # calculate location direction vector
                vec = np.array([col - x_lane[-1], row - y_lane[-1]], dtype=np.float32)
                # unit normalize
                vec = vec / np.linalg.norm(vec)
                vec_lane.append(vec)
            x_lane.append(col)
            y_lane.append(row)
        vec_lane.append(vec)
        x_lane, y_lane, vec_lane = np.array(x_lane), np.array(y_lane), np.array(vec_lane)

        # update PAF with direction vector for each fg pt based on nearest lane point
        PAF = assign_vecs(PAF, fg_pts_x, fg_pts_y, x_lane, y_lane, vec_lane)

    if viz: # visualization
        down_rate = 8 # subsample visualization by this factor
        fig, ax = plt.subplots()
        q = ax.quiver(np.arange(0, label.shape[1], down_rate), -np.arange(0, label.shape[0], down_rate), 
            PAF[::down_rate, ::down_rate, 0], -PAF[::down_rate, ::down_rate, 1], scale=120)
        plt.show()

    return PAF

#label = cv2.imread('test.png')
#PAF = generatePAFs(label[:, :, 0], viz=True)