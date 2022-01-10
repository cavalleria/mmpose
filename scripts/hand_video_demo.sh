
python demo/interhand3d_video_demo.py \
    demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth \
    configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py \
    https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3d_all_256x256-b9c1cf4c_20210506.pth \
    --camera-param-file tests/data/interhand2.6m/test_interhand2.6m_camera.json \
    --video-path resources/VID_20210610_195641.mp4 \
    --out-video-root vis_results \
    --rebase-keypoint-height
