# HRC-Pose
Pytorch implementation of HRC-Pose: Learning Point Cloud Representations with Pose Continuity for Category-Level 6D Object Pose Estimation.

## Table of Contents  

- HRC-Pose
  - Table of Content
  - Installation
  - Code Structure
  - Datasets
  - Training
  - Evaluation
    

## Installation - From Pip
```shell
cd HRC-Pose
pip install -r requirements.txt
```
## Code Structure
<details>
  <summary>[Click to expand]</summary>

- **HRC-Pose**
  - **HRC-Pose/backbone**: Some backbone networks used in training.
  - **HRC-Pose/config**
    - **HRC-Pose/config/common.py**: Some network and datasets settings for experiments.  
  - **HRC-Pose/contrast**
    - **HRC-Pose/contrast/Cont_split_rot.py**: Contrast learning codes for rotation.
    - **HRC-Pose/contrast/Cont_split_trans.py**: Contrast learning codes for translation.
    - **HRC-Pose/contrast/rnc_loss.py**: Contrast learning loss functions.
    - **HRC-Pose/contrast/Rot_3DGC.py**: Backbone networks used for rotation.
    - **HRC-Pose/contrast/utils.py**: Some utilities functions.
  - **HRC-Pose/datasets**
    - **HRC-Pose/datasets/data_augmentation.py**: Data augmentation functions.
    - **HRC-Pose/datasets/load_data.py**： Data loading functions.
  - **HRC-Pose/engine**
    - **HRC-Pose/engine/organize_loss.py**: Loss terms for training phase.
    - **HRC-Pose/engine/train_estimator.py**: The training phase.
  - **HRC-Pose/evaluation**
    - **HRC-Pose/evaluation/eval_utils_v1.py**: basic function for evaluation.
    - **HRC-Pose/evaluation/evaluate.py**: evaluation codes to evaluate our model's performance.
    - **HRC-Pose/evaluation/load_data_eval.py**: Data loading functions for the evaluation.
  - **HRC-Pose/losses**
      - **HRC-Pose/losses/fs_net_loss.py**: Loss functions from the FS-Net.
      - **HRC-Pose/losses/geometry_loss.py**: Loss functions from the GPV-Pose.
      - **HRC-Pose/losses/prop_loss.py**: Loss functions from the GPV-Pose.
      - **HRC-Pose/losses/recon_loss.py**: Loss functions from the GPV-Pose.
  - **HRC-Pose/mmcv**: MMCV packages.
  - **HRC-Pose/network**
    - **HRC-Pose/network/fs_net_repo**
        - **HRC-Pose/network/fs_net_repo/FaceRecon.py**: The reconstruction codes from the HS-Pose.
        - **HRC-Pose/network/fs_net_repo/gcn3d.py**: The 3DGCN codes from the HS-Pose.
        - **HRC-Pose/network/fs_net_repo/PoseNet9D.py**: The pose estimation codes.
        - **HRC-Pose/network/fs_net_repo/PoseR.py**: The rotation head codes.
        - **HRC-Pose/network/fs_net_repo/PoseTs.py**: The translation and size heads codes.
        - **HRC-Pose/network/fs_net_repo/PoseTs.py**: The translation and size heads codes.
    - **HRC-Pose/network/Pose_Estimator.py**: The training codes.
  - **HRC-Pose/tools**: Some neccessary functions for point cloud processing. 
</details>

## Dataset.
The datasets we used for training and testing are provided from the NOCS: Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation. 



## The Training Process.

```shell
python -m engine.train_estimator 
```
Detailed configurations are in `config/config.py` and `script.sh`

## Evaluation
```shell
python -m evaluation.evaluate 
```
Detailed configurations are in `config/config.py` and `script.sh`
