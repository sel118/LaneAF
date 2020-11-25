import cv2
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def generateAFs(label, viz=False):
    # creating AF arrays
    num_lanes = np.amax(label)
    VAF = np.zeros((label.shape[0], label.shape[1], 2))
    HAF = np.zeros((label.shape[0], label.shape[1], 2))

    # loop over each lane
    for l in range(1, num_lanes+1):
        # initialize previous row/cols
        prev_cols = np.array([], dtype=np.int64)
        prev_row = label.shape[0]

        # parse row by row, from second last to first
        for row in range(label.shape[0]-1, -1, -1):
            cols = np.where(label[row, :] == l)[0] # get fg cols

            # get horizontal vector
            for c in cols:
                if c <= np.mean(cols):
                    HAF[row, c, 0] = 1.0 # point to right
                else:
                    HAF[row, c, 0] = -1.0 # point to left

            # check if both previous cols and current cols are non-empty
            if prev_cols.size == 0: # if no previous row/cols, update and continue
                prev_cols = cols
                prev_row = row
                continue
            if cols.size == 0: # if no current cols, continue
                continue
            col = int(round(np.mean(cols))) # calculate mean

            # get vertical vector
            for c in prev_cols:
                # calculate location direction vector
                vec = np.array([col - c, row - prev_row], dtype=np.float32)
                # unit normalize
                vec = vec / np.linalg.norm(vec)
                VAF[prev_row, c, 0] = vec[0]
                VAF[prev_row, c, 1] = vec[1]

            # update previous row/cols with current row/cols
            prev_cols = cols
            prev_row = row

    if viz: # visualization
        down_rate = 5 # downsample visualization by this factor
        fig, (ax1, ax2) = plt.subplots(1, 2)
        # visualize VAF
        q = ax1.quiver(np.arange(0, label.shape[1], down_rate), -np.arange(0, label.shape[0], down_rate), 
            VAF[::down_rate, ::down_rate, 0], -VAF[::down_rate, ::down_rate, 1], scale=120)
        # visualize HAF
        q = ax2.quiver(np.arange(0, label.shape[1], down_rate), -np.arange(0, label.shape[0], down_rate), 
            HAF[::down_rate, ::down_rate, 0], -HAF[::down_rate, ::down_rate, 1], scale=120)
        plt.show()

    return VAF, HAF

def decodeAFs(BW, VAF, HAF, threshold=0.5, viz=False):
    output = np.zeros_like(BW, dtype=np.uint8) # initialize output array
    lane_end_pts = [] # keep track of latest lane points
    next_lane_id = 1 # next available lane ID

    if viz:
        im_color = cv2.applyColorMap(255*BW, cv2.COLORMAP_JET)
        cv2.imshow('BW', im_color)
        ret = cv2.waitKey(0)

    # start decoding from last row to first
    for row in range(BW.shape[0]-1, -1, -1):
        cols = np.where(BW[row, :] > 0)[0] # get fg cols
        clusters = [[]]
        if cols.size > 0:
            prev_col = cols[0]

        # parse horizontally
        for col in cols:
            if HAF[row, prev_col] > 0 and HAF[row, col] > 0: # keep moving to the right
                clusters[-1].append(col)
                prev_col = col
                continue
            elif HAF[row, prev_col] > 0 and HAF[row, col] < 0: # found lane center, process VAF
                clusters[-1].append(col)
                prev_col = col
            elif HAF[row, prev_col] < 0 and HAF[row, col] > 0: # found lane end, spawn new lane
                clusters.append([])
                clusters[-1].append(col)
                prev_col = col
                continue
            elif HAF[row, prev_col] < 0 and HAF[row, col] < 0: # keep moving to the right
                clusters[-1].append(col)
                prev_col = col
                continue

        # parse vertically
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            max_score = 0
            max_lane_id = 0
            cluster_mean = np.array([[np.mean(cluster), row]], dtype=np.float32)

            for idx, pts in enumerate(lane_end_pts): # for each end point in an active lane
                # get unit vector in direction of offset
                vecs = cluster_mean - pts
                # unit normalize
                vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

                # update line integral with current estimate
                vafs = np.array([VAF[int(round(x[1])), int(round(x[0])), :] for x in pts], dtype=np.float32)
                scores = np.sum(vafs * vecs, axis=1)
                score = np.mean(scores)

                # update highest score for current pixel
                if score > max_score:
                    max_score = score
                    max_lane_id = idx+1

            # if no match is found for current pixel
            # spawn a new line
            if max_score < threshold:
                output[row, cluster] = next_lane_id
                lane_end_pts.append(np.stack((np.array(cluster, dtype=np.float32), row*np.ones_like(cluster)), axis=1))
                next_lane_id += 1
                continue

            # update best lane match with current pixel
            output[row, cluster] = max_lane_id
            lane_end_pts[max_lane_id-1] = np.stack((np.array(cluster, dtype=np.float32), row*np.ones_like(cluster)), axis=1)

    if viz:
        im_color = cv2.applyColorMap(40*output, cv2.COLORMAP_JET)
        cv2.imshow('Output', im_color)
        ret = cv2.waitKey(0)

    return output