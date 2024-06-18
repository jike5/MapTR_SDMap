import os.path as osp
import pickle
import shutil
import tempfile
import time
import json
import os
import tqdm
import cv2
import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results

import datetime
import mmcv
import numpy as np
import pycocotools.mask as mask_util

from lane_evaluator import LaneEvaluator, line_classes, polygon_classes


WIDTH = 240
HEIGHT = 480

PC_RANGE = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
# PC_RANGE = [-30.0, -60.0, -2.0, 30.0, 60.0, 2.0]
# PC_RANGE = [-30.0, -120.0, -2.0, 30.0, 120.0, 2.0]

INST_THRESHOLD_LINE = 0.05
INST_THRESHOLD_POLYGON = 0.05
IOUTHRS_LINE = np.linspace(0.25, 0.50, int(np.round((0.50 - 0.25) / 0.05)) + 1)
IOUTHRS_POLYGON = np.linspace(0.50, 0.75, int(np.round((0.75 - 0.50) / 0.05)) + 1)
MAXDETS_LINE = [1, 10, 100]
MAXDETS_POLYGON = [1, 10, 100]
QUERY_LINE = [
    ("AP", "all", "all", 100),
    ("AP", "all", "divider", 100),
    ("AP", "all", "boundary", 100),
    ("AP", "all", "ped_crossing", 100),
    ("AP", 0.25, "all", 100),
    ("AP", 0.25, "divider", 100),
    ("AP", 0.25, "boundary", 100),
    ("AP", 0.50, "all", 100),
    ("AP", 0.50, "divider", 100),
    ("AP", 0.50, "boundary", 100),
    ("AR", "all", "all", 1),
    ("AR", "all", "all", 10),
    ("AR", "all", "all", 100),
    ("AR", "all", "divider", 100),
    ("AR", "all", "boundary", 100),
]
QUERY_PED = [
    ("AP", "all", "all", 100),
    ("AP", "all", "ped_crossing", 100),

]
QUERY_POLYGON = [
    ("AP", "all", "ped_crossing", 100),
    ("AP", 0.50, "ped_crossing", 100),
    ("AP", 0.75, "ped_crossing", 100),
    ("AR", "all", "ped_crossing", 1),
    ("AR", "all", "ped_crossing", 10),
    ("AR", "all", "ped_crossing", 100),
]
DILATION_LINE = 5
DILATION_POLYGON = 5


def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]


def post_process_instance_polygon(data, preds):
    def instance_to_segm_polygon(params):
        normalized_params = np.copy(params)
        normalized_params[:, 0] = (normalized_params[:, 0] - PC_RANGE[0]) / (PC_RANGE[3] - PC_RANGE[0]) * WIDTH
        normalized_params[:, 1] = (normalized_params[:, 1] - PC_RANGE[1]) / (PC_RANGE[4] - PC_RANGE[1]) * HEIGHT
        points = np.around(normalized_params).astype(int)
        mask = np.zeros([HEIGHT, WIDTH])
        cv2.fillPoly(mask, [points], 1)
        return (mask > 0).astype(float)

    def instance_to_segm_line(params):
        normalized_params = np.copy(params)
        normalized_params[:, 0] = (normalized_params[:, 0] - PC_RANGE[0]) / (PC_RANGE[3] - PC_RANGE[0]) * WIDTH
        normalized_params[:, 1] = (normalized_params[:, 1] - PC_RANGE[1]) / (PC_RANGE[4] - PC_RANGE[1]) * HEIGHT
        points = np.around(normalized_params).astype(int)
        mask = np.zeros([HEIGHT, WIDTH])
        cv2.polylines(mask, [points], False, 1, 1)
        return (mask > 0).astype(float)

    gts = []
    dts = []

    gt_vectors = data
    for vector in gt_vectors:
        if vector["type"] in polygon_classes.values():
            gts.append({
                "class": vector["type"],
                "segmentation": instance_to_segm_polygon(vector["pts"]),
            })

    pred_instances = preds
    for vector in pred_instances:
        score = vector['confidence_level']
        pts = vector['pts']
        label = vector['type']
        if score < INST_THRESHOLD_POLYGON or len(pts) < 2:
            continue
        if label in polygon_classes.values():
            dts.append({
                "class": label,
                "segmentation": instance_to_segm_polygon(pts),
                "score": score,
            })

    return gts, dts


def post_process_instance_line(data, preds):
    def instance_to_segm_line(params):
        normalized_params = np.copy(params)
        normalized_params[:, 0] = (normalized_params[:, 0] - PC_RANGE[0]) / (PC_RANGE[3] - PC_RANGE[0]) * WIDTH
        normalized_params[:, 1] = (normalized_params[:, 1] - PC_RANGE[1]) / (PC_RANGE[4] - PC_RANGE[1]) * HEIGHT
        points = np.around(normalized_params).astype(int)
        mask = np.zeros([HEIGHT, WIDTH])
        cv2.polylines(mask, [points], False, 1, 1)
        return (mask > 0).astype(float)

    gts = []
    dts = []

    gt_vectors = data
    for vector in gt_vectors:
        if vector["type"] in line_classes.values():
            gts.append({
                "class": vector["type"],
                "segmentation": instance_to_segm_line(vector["pts"]),
            })

    pred_instances = preds
    for vector in pred_instances:
        score = vector['confidence_level']
        pts = vector['pts']
        label = vector['type']
        if score < INST_THRESHOLD_LINE or len(pts) < 2:
            continue
        if label in line_classes.values():
            dts.append({
                "class": label,
                "segmentation": instance_to_segm_line(pts),
                "score": score,
            })

    return gts, dts


def custom_single_gpu_test(map_ann_file, map_pred_file):
    with open(map_ann_file, "r") as f:
        gt_data = json.load(f)['results']

    with open(map_pred_file, "r") as f:
        dt_data = json.load(f)['results']

    assert len(gt_data) == len(dt_data)
    coco_results_line = []
    coco_results_ped= []
    coco_results_polygon = []

    evaluator_line = LaneEvaluator(classes=line_classes,
                                   parameterization="instanceseg",
                                   post_process_func=post_process_instance_line,
                                   iouThrs=IOUTHRS_LINE,
                                   maxDets=MAXDETS_LINE,
                                   query=QUERY_LINE,
                                   dilation=DILATION_LINE,
                                   to_thin=False,
                                   width=WIDTH,
                                   height=HEIGHT)
    evaluator_ped = LaneEvaluator(classes=line_classes,
                                   parameterization="instanceseg",
                                   post_process_func=post_process_instance_line,
                                   iouThrs=IOUTHRS_POLYGON,
                                   maxDets=MAXDETS_LINE,
                                   query=QUERY_PED,
                                   dilation=DILATION_LINE,
                                   to_thin=False,
                                   width=WIDTH,
                                   height=HEIGHT)
    evaluator_polygon = LaneEvaluator(classes=polygon_classes,
                                      parameterization="instanceseg",
                                      post_process_func=post_process_instance_polygon,
                                      iouThrs=IOUTHRS_POLYGON,
                                      maxDets=MAXDETS_POLYGON,
                                      query=QUERY_POLYGON,
                                      dilation=DILATION_POLYGON,
                                      to_thin=False,
                                      width=WIDTH,
                                      height=HEIGHT)

    for token, gt_list in tqdm.tqdm(gt_data.items()):
        coco_results_line.append(evaluator_line.evaluate(gt_list, dt_data[token]))
        coco_results_ped.append(evaluator_ped.evaluate(gt_list, dt_data[token]))
        coco_results_polygon.append(evaluator_polygon.evaluate(gt_list, dt_data[token]))


    return [{"type": "line", "evaluator": evaluator_line, "results": coco_results_line}, 
            {"type": "line", "evaluator": evaluator_ped, "results": coco_results_ped}, 
            {"type": "polygon", "evaluator": evaluator_polygon, "results": coco_results_polygon}]


def main_pmapnet_vector():
    map_ann_file = "/DATA_EDS2/zhuzx/Work/P-MapNet/json_file/gt60.json"
    map_pred_file = "/DATA_EDS2/zhuzx/Work/P-MapNet/json_file/pred60.json"
    outputs_coco = custom_single_gpu_test(map_ann_file, map_pred_file)
    eval_result = {}
    for output in outputs_coco:
        eval_result[output["type"]] = output["evaluator"].summarize(output["results"])
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    output_path = os.path.join('work_dirs/mapvr', f"results_{date}.json")
    with open(output_path, "w") as f:
        json.dump(eval_result, f)
    print(f"Results are written to {output_path}")

main_pmapnet_vector()