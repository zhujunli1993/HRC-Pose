import os
import torch
import random
from network.Pose_Estimator import Pose_Estimator  
from contrast.Cont_split_trans import Model_Trans_all as Model_trans
from contrast.Cont_split_rot import Model_Rot_all as Model_rot
from tools.geom_utils import generate_RT
from config.config import *
from absl import app

FLAGS = flags.FLAGS
from evaluation.load_data_eval_ros import PoseDataImg
import numpy as np
import time
import cv2
# from creating log
import tensorflow as tf
from evaluation.eval_utils_v1 import setup_logger
from evaluation.eval_utils_v1 import compute_degree_cm_mAP, draw_bbox
from tqdm import tqdm
def draw_all(opt, result, save_dir, save_id):
    """ Load data and draw visualization results.
    """
    intrinsics = result['K']
    img_pth = result['image_path']
    
    image = cv2.imread(img_pth)
    
    # Get the correct RTs for Class_ids. If the target is missing we will return np.eye(). If multi-target is matched, we only keep the first.
    all_gt_RTs = []
    all_pred_RTs = []
    all_gt_scales = []
    all_pred_scales = []
    misses = []
    import pdb;pdb.set_trace()
    for i in tqdm(range(len(result["cat_id"]))):
        
        
        pred_RTs = result["pred_RTs"][i]
        pred_scales = result["pred_scales"][i]
        
        gt_RTs = pred_RTs
        gt_scales = pred_scales
        miss = False
        if len(pred_RTs) <= 0:
            
            
            pred_RTs = np.eye(4)
            pred_RTs = np.broadcast_to(pred_RTs, gt_RTs.shape)
            pred_scales = np.zeros(gt_scales.shape)
        
        # elif len(pred_RTs) > 1 and len(gt_RTs)==1:
        #     pred_RTs = pred_RTs[0]
        #     pred_scales = pred_scales[0]
        
        misses.append(miss)
        all_pred_RTs.append(pred_RTs)
        all_gt_RTs.append(gt_RTs)
        all_pred_scales.append(pred_scales)
        all_gt_scales.append(gt_scales)
    
    (h, w) = image.shape[:2]
    center = (w/2, h/2)

    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    
    draw_bbox(image, all_gt_RTs, all_pred_RTs, all_gt_scales, all_pred_scales, class_ids=result["cat_id"], misses=misses, 
              intrinsics=intrinsics, save_path=os.path.join(save_dir, 'ROS'+"_"+str(save_id)+"_bbox.png"))
def seed_init_fn(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

device = 'cuda'

def evaluate(argv):
    if FLAGS.eval_seed == -1:
        seed = int(time.time())
    else:
        seed = FLAGS.eval_seed
    seed_init_fn(seed)
    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)
    tf.compat.v1.disable_eager_execution()

    resume_model_num = FLAGS.resume_model.split('/')[-1].split('.')[0].split('_')[1]
    logger = setup_logger('eval_log', os.path.join(FLAGS.model_save, 'log_eval_'+resume_model_num+'.txt'))
    Train_stage = 'PoseNet_only'
    FLAGS.train = False

    model_name = os.path.basename(FLAGS.resume_model).split('.')[0]
    # build dataset annd dataloader
    
    
    output_path = os.path.join(FLAGS.model_save, f'eval_result_{model_name}')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    import pickle

    t_inference = 0.0
    img_count = 0
    pred_result_save_path = os.path.join(output_path, 'pred_result.pkl')
    
    network = Pose_Estimator(Train_stage)
    network = network.to(device)
    ################ Load CLIP Models #########################
    ''' Load pretrained CLIP trianing agent'''
    clip_model_rot = Model_rot().to(FLAGS.device)
    if FLAGS.pretrained_clip_rot_model_path:
        clip_model_rot.load_state_dict(torch.load(FLAGS.pretrained_clip_rot_model_path))
    else:
        print("No Pretrained Rotation CLIP Model !!!")
        sys.exit()
        
    clip_model_t = Model_trans().to(FLAGS.device)
    if FLAGS.pretrained_clip_t_model_path:
        clip_model_t.load_state_dict(torch.load(FLAGS.pretrained_clip_t_model_path))
    else:
        print("No Pretrained Translation CLIP Model !!!")
        sys.exit()
    # Freeze clip model parameters
    for param in clip_model_t.parameters():
        param.requires_grad = False     
    for param in clip_model_rot.parameters():
        param.requires_grad = False   
    ################ Finish Loading CLIP Models #########################  
    
    #################### Loading Pose Estimator ############################
    if FLAGS.resume:
        state_dict = torch.load(FLAGS.resume_model)['posenet_state_dict']
        unnecessary_nets = ['posenet.face_recon.conv1d_block', 'posenet.face_recon.face_head', 'posenet.face_recon.recon_head']
        for key in list(state_dict.keys()):
            for net_to_delete in unnecessary_nets:
                if key.startswith(net_to_delete):
                    state_dict.pop(key)
            # Adapt weight name to match old code version. 
            # Not necessary for weights trained using newest code. 
            # Dose not change any function. 
            if 'resconv' in key:
                state_dict[key.replace("resconv", "STE_layer")] = state_dict.pop(key)
        network.load_state_dict(state_dict, strict=True) 
    else:
        raise NotImplementedError
    
    # start to test
    network = network.eval()
    clip_model_rot = clip_model_rot.eval()
    clip_model_t = clip_model_t.eval()
    pred_results = []
    depth_pth = '/workspace/clip/ContrastPose/maskrcnn_data/2_depth.npy'
    # depth_pth = '/workspace/clip/CLIPDATA/Real/train/scene_1/0001_depth.png'
    pc_pth = '/workspace/clip/ContrastPose/maskrcnn_data/2_points.npy'
    det_pth = '/workspace/clip/ContrastPose/maskrcnn_data/2_masks.npy'
    label_pth = '/workspace/clip/ContrastPose/maskrcnn_data/2_labels.npy'
    bbox_pth = '/workspace/clip/ContrastPose/maskrcnn_data/2_boxes.npy'
    c_X = 324.8378935141304
    c_Y = 235.4308556036733
    f_X = 527.8716747158403
    f_Y = 520.6781086489601
    intrinsic = np.array(([f_X, 0.0, c_X], [0.0, f_Y, c_Y], [0.0, 0.0, 1.0]), dtype=float)
    data = PoseDataImg(depth_pth,det_pth,label_pth,bbox_pth,intrinsic,pc_pth=None)
    mean_shape = data['mean_shape'].to(device)
    sym = data['sym_info'].to(device)
    t_start = time.time()
    output_dict \
        = network(clip_r_func=clip_model_rot.to(device),
                    clip_t_func=clip_model_t.to(device),
                    PC=data['pcl_in'].to(device), 
                    obj_id=data['cat_id_0base'].to(device), 
                    mean_shape=mean_shape,
                    sym=sym,
                #   def_mask=data['roi_mask'].to(device)
                    )
    
    p_green_R_vec = output_dict['p_green_R'].detach()
    p_red_R_vec = output_dict['p_red_R'].detach()
    p_T = output_dict['Pred_T'].detach()
    p_s = output_dict['Pred_s'].detach()
    f_green_R = output_dict['f_green_R'].detach()
    f_red_R = output_dict['f_red_R'].detach()
    pred_s = p_s + mean_shape
    pred_RT = generate_RT([p_green_R_vec, p_red_R_vec], [f_green_R, f_red_R], p_T, mode='vec', sym=sym)

    t_inference += time.time() - t_start
    img_count += 1
    
    if pred_RT is not None:
        pred_RT = pred_RT.detach().cpu().numpy()
        pred_s = pred_s.detach().cpu().numpy()
        data['pred_RTs'] = pred_RT
        data['pred_scales'] = pred_s
        data['image_path'] = '/workspace/clip/ContrastPose/maskrcnn_data/2.jpg'
        data['K'] = intrinsic
    else:
        assert NotImplementedError
    pred_results.append(data)
    
    torch.cuda.empty_cache()
    with open(pred_result_save_path, 'wb') as file:
        pickle.dump(pred_results, file)
    print('inference time:', t_inference / img_count)
    # if FLAGS.eval_inference_only:
    #     import sys
    #     sys.exit()
        
    ################### Draw Visualization ##############################    
    
    os.makedirs(os.path.join(output_path, "vis_new"), exist_ok=True)
    save_dir = os.path.join(output_path, "vis_new")
    
    draw_all(FLAGS, pred_results[0], save_dir, save_id=2)
    print("Drawing Done!")
    

if __name__ == "__main__":
    app.run(evaluate)
