# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np
from xtcocotools.coco import COCO

from mmpose.apis import (inference_interhand_3d_model, init_pose_model,
                         process_mmdet_results, vis_3d_pose_result)

from mmdet.apis import inference_detector, init_detector
from mmpose.datasets import DatasetInfo
from mmpose.core import SimpleCamera

def _transform_interhand_camera_param(interhand_camera_param):
    """Transform the camera parameters in interhand2.6m dataset to the format
    of SimpleCamera.

    Args:
        interhand_camera_param (dict): camera parameters including:
            - camrot: 3x3, camera rotation matrix (world-to-camera)
            - campos: 3x1, camera location in world space
            - focal: 2x1, camera focal length
            - princpt: 2x1, camera center

    Returns:
        param (dict): camera parameters including:
            - R: 3x3, camera rotation matrix (camera-to-world)
            - T: 3x1, camera translation (camera-to-world)
            - f: 2x1, camera focal length
            - c: 2x1, camera center
    """
    camera_param = {}
    camera_param['R'] = np.array(interhand_camera_param['camrot']).T
    camera_param['T'] = np.array(interhand_camera_param['campos'])[:, None]
    camera_param['f'] = np.array(interhand_camera_param['focal'])[:, None]
    camera_param['c'] = np.array(interhand_camera_param['princpt'])[:, None]
    return camera_param

def parse_args():

    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for det network')
    parser.add_argument('det_checkpoint', help='Checkpoint file')
    parser.add_argument('pose_config', help='Config file for pose network')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--camera-id',
        type=int,
        default=0,
        help='Camera device id')
    parser.add_argument(
        '--camera-param-file',
        type=str,
        default=None,
        help='Camera parameter file for converting 3D pose predictions from '
        ' the pixel space to camera space. If None, keypoints in pixel space'
        'will be visualized')
    parser.add_argument(
        '--rebase-keypoint-height',
        action='store_true',
        help='Rebase the predicted 3D pose so its lowest keypoint has a '
        'height of 0 (landing on the ground). This is useful for '
        'visualization when the model do not predict the global position '
        'of the 3D pose.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--video-path',
        type=str,
        help='Video path')
    parser.add_argument(
        '--device', default='cuda:0', help='Device for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--bbox-thr', type=float, default=0.5, help='Bbox score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=6,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Link thickness for visualization')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    assert args.det_config is not None
    assert args.det_checkpoint is not None

    # build the det model from a config file and a checkpoint file
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())
    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # load camera parameters
    camera_params = None
    if args.camera_param_file is not None:
        camera_params = mmcv.load(args.camera_param_file)

    #cap = cv2.VideoCapture(args.camera_id)
    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f'Failed to load video file'

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True
    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        #size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        #        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        size = (800, 400)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc, fps, size)
    cnt = 0
    while(cap.isOpened()):
        hand_results_list = []
        flag, img = cap.read()
        if not flag:
            break
        # Hand detection
        mmdet_results = inference_detector(det_model, img)

        hand_results = process_mmdet_results(mmdet_results)
        if len(hand_results) == 0:
            rimg = cv2.resize(img, (800, 400))
            videoWriter.write(rimg)
            continue
        else:
            hand_results_list.append(hand_results[0])
            # Hand pose
            pose_results = inference_interhand_3d_model(
                pose_model, img, hand_results_list, format='xyxy', dataset=dataset)

            # Post processing
            pose_results = pose_results[0]
            keypoints_3d = pose_results['keypoints_3d']
            # normalize kpt score
            if keypoints_3d[:, 3].max() > 1:
                keypoints_3d[:, 3] /= 255
            # get 2D keypoints in pixel space
            pose_results['keypoints'] = keypoints_3d[:, [0, 1, 3]]

            # rotate the keypoint to make z-axis correspondent to height
            # for better visualization
            vis_R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            keypoints_3d[:, :3] = keypoints_3d[:, :3] @ vis_R

            # rebase height (z-axis)
            if args.rebase_keypoint_height:
                valid = keypoints_3d[..., 3] > 0
                keypoints_3d[..., 2] -= np.min(
                    keypoints_3d[valid, 2], axis=-1, keepdims=True)
            pose_results['keypoints_3d'] = keypoints_3d

            # Add title
            pose_results['title'] = f'keypoints_3d'

            # Visualization
            pose_results_vis = []
            pose_results_vis.append(pose_results)
            vis_img = vis_3d_pose_result(
                pose_model,
                result=pose_results_vis,
                img=img,
                dataset=dataset,
                show=args.show,
                kpt_score_thr=args.kpt_thr,
                radius=args.radius,
                thickness=args.thickness,
            )
            cnt += 1
            if cnt % 100 == 0:
                print(cnt)
            #cv2.imwrite('./vis_results/'+str(cnt)+'.jpg', vis_img)

            if args.show:
                cv2.imshow('Image', vis_img)

            if save_out_video:
                videoWriter.write(vis_img)

            if args.show and cv2.waitKet(1) & 0xFF == ord('q'):
                break
    cap.release()
    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destoryAllWindows()




if __name__ == '__main__':
    main()
