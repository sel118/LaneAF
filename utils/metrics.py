import json

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import linear_sum_assignment


# borrowed code from offficial TuSimple benchmark
# https://github.com/TuSimple/tusimple-benchmark/blob/master/evaluate/lane.py
class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        if running_time > 200 or len(gt) + 2 < len(pred):
            raise Exception('Format of lanes error.')
            return 0., 0., 1.
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        for x_gts, thresh in zip(gt, threshs):
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.) , 1.)

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        try:
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}
        accuracy, fp, fn = 0., 0., 0.
        for pred in json_pred:
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception('raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time = pred['run_time']
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']
            try:
                a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        # the first return parameter is the default ranking parameter
        return json.dumps([
            {'name': 'Accuracy', 'value': accuracy / num, 'order': 'desc'},
            {'name': 'FP', 'value': fp / num, 'order': 'asc'},
            {'name': 'FN', 'value': fn / num, 'order': 'asc'}
        ])

def match_multi_class(pred, target):
    pred_ids = np.unique(pred[pred > 0]) # find unique pred ids
    target_ids = np.unique(target[target > 0]) # find unique target ids
    pred_out = np.zeros_like(pred) # initialize output array

    # return input array if no lane points in prediction/target
    if pred_ids.size == 0:
        return pred
    if target_ids.size == 0:
        return pred

    assigned = [False for _ in range(pred_ids.size)] # keep track of which ids have been asssigned

    # create cost matrix for matching predicted with target lanes
    C = np.zeros((target_ids.size, pred_ids.size))
    for i, t_id in enumerate(target_ids):
        for j, p_id in enumerate(pred_ids):
            C[i, j] = -np.sum(target[pred == p_id] == t_id)

    # optimal linear assignment (Hungarian)
    row_ind, col_ind = linear_sum_assignment(C)
    for r, c in zip(row_ind, col_ind):
        pred_out[pred == pred_ids[c]] = target_ids[r]
        assigned[c] = True

    # get next available ID to assign
    if target_ids.size > 0:
        max_target_id = np.amax(target_ids)
    else:
        max_target_id = 0
    next_id = max_target_id + 1 # next available class id
    # assign IDs to unassigned fg pixels
    for i, p_id in enumerate(pred_ids):
        if assigned[i]:
            pass
        else:
            pred_out[pred == p_id] = next_id
            next_id += 1
    assert np.unique(pred[pred > 0]).size == np.unique(pred_out[pred_out > 0]).size, "Number of output lanes altered!"

    return pred_out
