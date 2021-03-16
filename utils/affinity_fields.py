import cv2
import numpy as np
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
                if c < np.mean(cols):
                    HAF[row, c, 0] = 1.0 # point to right
                elif c > np.mean(cols):
                    HAF[row, c, 0] = -1.0 # point to left
                else:
                    HAF[row, c, 0] = 0.0 # point to left

            # check if both previous cols and current cols are non-empty
            if prev_cols.size == 0: # if no previous row/cols, update and continue
                prev_cols = cols
                prev_row = row
                continue
            if cols.size == 0: # if no current cols, continue
                continue
            col = np.mean(cols) # calculate mean

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
        down_rate = 1 # downsample visualization by this factor
        fig, (ax1, ax2) = plt.subplots(1, 2)
        # visualize VAF
        q = ax1.quiver(np.arange(0, label.shape[1], down_rate), -np.arange(0, label.shape[0], down_rate), 
            VAF[::down_rate, ::down_rate, 0], -VAF[::down_rate, ::down_rate, 1], scale=120)
        # visualize HAF
        q = ax2.quiver(np.arange(0, label.shape[1], down_rate), -np.arange(0, label.shape[0], down_rate), 
            HAF[::down_rate, ::down_rate, 0], -HAF[::down_rate, ::down_rate, 1], scale=120)
        plt.show()

    return VAF, HAF

def decodeAFs(BW, VAF, HAF, fg_thresh=128, err_thresh=5, viz=False):
    output = np.zeros_like(BW, dtype=np.uint8) # initialize output array
    lane_end_pts = [] # keep track of latest lane points
    next_lane_id = 1 # next available lane ID

    if viz:
        im_color = cv2.applyColorMap(BW, cv2.COLORMAP_JET)
        cv2.imshow('BW', im_color)
        ret = cv2.waitKey(0)

    # start decoding from last row to first
    for row in range(BW.shape[0]-1, -1, -1):
        cols = np.where(BW[row, :] > fg_thresh)[0] # get fg cols
        clusters = [[]]
        if cols.size > 0:
            prev_col = cols[0]

        # parse horizontally
        for col in cols:
            if col - prev_col > err_thresh: # if too far away from last point
                clusters.append([])
                clusters[-1].append(col)
                prev_col = col
                continue
            if HAF[row, prev_col] >= 0 and HAF[row, col] >= 0: # keep moving to the right
                clusters[-1].append(col)
                prev_col = col
                continue
            elif HAF[row, prev_col] >= 0 and HAF[row, col] < 0: # found lane center, process VAF
                clusters[-1].append(col)
                prev_col = col
            elif HAF[row, prev_col] < 0 and HAF[row, col] >= 0: # found lane end, spawn new lane
                clusters.append([])
                clusters[-1].append(col)
                prev_col = col
                continue
            elif HAF[row, prev_col] < 0 and HAF[row, col] < 0: # keep moving to the right
                clusters[-1].append(col)
                prev_col = col
                continue

        # parse vertically
        # assign existing lanes
        assigned = [False for _ in clusters]
        C = np.Inf*np.ones((len(lane_end_pts), len(clusters)), dtype=np.float64)
        for r, pts in enumerate(lane_end_pts): # for each end point in an active lane
            for c, cluster in enumerate(clusters):
                if len(cluster) == 0:
                    continue
                # mean of current cluster
                cluster_mean = np.array([[np.mean(cluster), row]], dtype=np.float32)
                # get vafs from lane end points
                vafs = np.array([VAF[int(round(x[1])), int(round(x[0])), :] for x in pts], dtype=np.float32)
                vafs = vafs / np.linalg.norm(vafs, axis=1, keepdims=True)
                # get predicted cluster center by adding vafs
                pred_points = pts + vafs*np.linalg.norm(pts - cluster_mean, axis=1, keepdims=True)
                # get error between prediceted cluster center and actual cluster center
                error = np.mean(np.linalg.norm(pred_points - cluster_mean, axis=1))
                C[r, c] = error
        # assign clusters to lane (in acsending order of error)
        row_ind, col_ind = np.unravel_index(np.argsort(C, axis=None), C.shape)
        for r, c in zip(row_ind, col_ind):
            if C[r, c] >= err_thresh:
                break
            if assigned[c]:
                continue
            assigned[c] = True
            # update best lane match with current pixel
            output[row, clusters[c]] = r+1
            lane_end_pts[r] = np.stack((np.array(clusters[c], dtype=np.float32), row*np.ones_like(clusters[c])), axis=1)
        # initialize unassigned clusters to new lanes
        for c, cluster in enumerate(clusters):
            if len(cluster) == 0:
                continue
            if not assigned[c]:
                output[row, cluster] = next_lane_id
                lane_end_pts.append(np.stack((np.array(cluster, dtype=np.float32), row*np.ones_like(cluster)), axis=1))
                next_lane_id += 1

    if viz:
        im_color = cv2.applyColorMap(40*output, cv2.COLORMAP_JET)
        cv2.imshow('Output', im_color)
        ret = cv2.waitKey(0)

    return output
