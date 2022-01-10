
python demo/interhand3d_img_demo.py \
    configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py \
    https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3d_all_256x256-b9c1cf4c_20210506.pth \
    --json-file tests/data/interhand2.6m/test_interhand2.6m_data.json \
    --img-root tests/data/interhand2.6m \
    --camera-param-file tests/data/interhand2.6m/test_interhand2.6m_camera.json \
    --gt-joints-file tests/data/interhand2.6m/test_interhand2.6m_joint_3d.json \
    --out-img-root vis_results \
    --rebase-keypoint-height \
    --show-ground-truth
