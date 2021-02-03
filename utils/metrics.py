import numpy as np
import cv2
import matplotlib.pyplot as plt


def match_multi_class(pred, target):
    pred_ids = np.unique(pred[pred > 0]) # find unique pred ids
    target_ids = np.unique(target[target > 0]) # find unique target ids
    pred_out = np.zeros_like(pred) # initialize output array

    sizes = np.array([pred[pred == idx].size for idx in pred_ids]) # get sizes of each predicted class
    order = np.argsort(sizes)[::-1] # descending order of size
    pred_ids = pred_ids[order] # sort prediceted classes in descending order of size
    assigned = [False for _ in range(np.amax(target_ids) + 1)] # keep track of which target ids have been asssigned

    next_id = np.amax(target_ids) + 1 # next available class id
    for idx in pred_ids:
        # get target id with max overlap
        max_id = np.argmax(np.bincount(target[pred == idx]))
        if max_id == 0:
            pred_out[pred == idx] = next_id
            next_id += 1
            continue
        if assigned[max_id]:
            pred_out[pred == idx] = next_id
            next_id += 1
        else:
            pred_out[pred == idx] = max_id
            assigned[max_id] = True

    return pred_out
