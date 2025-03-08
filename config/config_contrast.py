import torch
import argparse
def get_config():
    parser = argparse.ArgumentParser()
    
    
    '''CLIP settings'''

    parser.add_argument('--pix_embedding', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--debug', default=True, action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--factor', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--wandb_proj', type=str, default='clip_2')
    parser.add_argument('--wandb_name', type=str, default='clip_v2')
    parser.add_argument('--pts_embedding', type=int, default=1024)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--trainable', default=False, action='store_true')
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--num_projection_layers', type=int, default=1)
    parser.add_argument('--projection_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--k1', type=float, default=0.400)
    parser.add_argument('--k2', type=float, default=0.400)
    parser.add_argument('--use_clip', type=float, default=1.0)
    parser.add_argument('--resume_model', type=str, default='')
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--use_clip_global', type=float, default=1.0)
    parser.add_argument('--use_clip_nonLinear',type=float, default=0.0)
    parser.add_argument('--use_clip_atte',type=float, default=0.0)
    parser.add_argument('--save_info',type=float, default=0.0)
    parser.add_argument('--eval_inference_only',type=int, default=0)
    parser.add_argument('--heads',type=int, default=2)
    parser.add_argument('--total_epoch',type=int, default=150)
    """ dataset """
    parser.add_argument('--dataset_dir', type=str, default='/workspace/DATA/NOCS')
    parser.add_argument('--detection_dir', type=str, default='')
    parser.add_argument('--synset_names', nargs='+', default=['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug'])
    parser.add_argument('--selected_classes', nargs='+')
    parser.add_argument('--data_path', type=str, default='/workspace/DATA/NOCS')
    parser.add_argument('--o2c_pose', default=True, action='store_true')
    parser.add_argument('--max_batch_size', type=int, default=192)
    parser.add_argument('--mini_bs', type=int, default=192)
    parser.add_argument('--pose_mode', type=str, default='original')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--percentage_data_for_train', type=float, default=1.0) 
    parser.add_argument('--percentage_data_for_val', type=float, default=1.0) 
    parser.add_argument('--percentage_data_for_test', type=float, default=1.0) 
    parser.add_argument('--train_source', type=str, default='CAMERA+Real')
    parser.add_argument('--val_source', type=str, default='CAMERA')
    parser.add_argument('--test_source', type=str, default='Real')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--per_obj', type=str, default='')
    
    
    """ model """
    parser.add_argument('--train_steps', type=int, default=200)
    parser.add_argument('--posenet_mode',  type=str, default='score')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--sampler_mode', nargs='+')
    parser.add_argument('--sampling_steps', type=int)
    parser.add_argument('--sde_mode', type=str, default='ve')
    parser.add_argument('--sigma', type=float, default=25) # base-sigma for SDE
    parser.add_argument('--likelihood_weighting', default=False, action='store_true')
    parser.add_argument('--regression_head', type=str, default='Rx_Ry_and_T')
    parser.add_argument('--pointnet2_params', type=str, default='lighter')
    parser.add_argument('--pts_encoder', type=str, default='pointnet2') 
    parser.add_argument('--energy_mode', type=str, default='IP') 
    parser.add_argument('--s_theta_mode', type=str, default='score') 
    parser.add_argument('--norm_energy', type=str, default='identical') 
    parser.add_argument('--theta_lambda', type=float, default=1.0)
    parser.add_argument('--shift_lambda', type=float, default=1.0)
    parser.add_argument('--smooth_l1_beta', type=float, default=1.0)
    """ training """
    parser.add_argument('--agent_type', type=str, default='score', help='one of the [score, energy, energy_with_ranking]')
    parser.add_argument('--pretrained_clip_model_path', type=str)
    parser.add_argument('--pretrained_clip_rot_model_path', type=str)
    parser.add_argument('--pretrained_clip_t_model_path', type=str)
    parser.add_argument('--pretrained_clip_model_green_path', type=str)
    parser.add_argument('--pretrained_clip_model_red_path', type=str)
    parser.add_argument('--pretrained_decoder_rot_model_path', type=str)
    parser.add_argument('--pretrained_decoder_t_model_path', type=str)
    parser.add_argument('--model_save', type=str)
    parser.add_argument('--pretrained_decoder_model_path', type=str)
    parser.add_argument('--distillation', default=False, action='store_true')
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--log_dir', type=str, default='debug')
    parser.add_argument('--optimizer',  type=str, default='Adam')
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--repeat_num', type=int, default=20)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--ema_rate', type=float, default=0.999)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--lr_decay', type=float, default=0.98)
    parser.add_argument('--use_pretrain', default=False, action='store_true')
    parser.add_argument('--parallel', default=False, action='store_true')   
    parser.add_argument('--num_gpu', type=int, default=4)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--train', default=True, action='store_true')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    """ testing """
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--pred', default=False, action='store_true')
    parser.add_argument('--eval_repeat_num', type=int, default=50)
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--max_eval_num', type=int, default=10000000)
    parser.add_argument('--results_path', type=str, default='')
    parser.add_argument('--T0', type=float, default=1.0)
    
    """ GPV setting """
    # pose network
    parser.add_argument('--rgb_backbone', type=str, default='vit')
    parser.add_argument('--feat_c_R', type=int, default=1024, help='input channel of rotation, default=1283')
    parser.add_argument('--R_c', type=int, default=4, help='output channel of rotation, here confidence(1)+ rot(3)')
    parser.add_argument('--feat_c_ts', type=int, default=1286, help='input channel of translation and size')
    parser.add_argument('--Ts_c', type=int, default=3,  help='output channel of translation (3)')
    parser.add_argument('--feat_face',type=int, default=1280, help='input channel of the face recon')

    parser.add_argument('--face_recon_c', type=int, default=6 * 5, help='for every point, we predict its distance and normal to each face')
    #  the storage form is 6*3 normal, then the following 6 parametes distance, the last 6 parameters confidence
    parser.add_argument('--gcn_sup_num', type=int, default=7, help='support number for gcn')
    parser.add_argument('--gcn_n_num', type=int, default=10, help='neighbor number for gcn')
    parser.add_argument('--obj_c', type=int, default=6, help='nnumber of categories')
    
    parser.add_argument('--fsnet_loss_type', type=str, default='l1', help='l1 or smoothl1')

    parser.add_argument('--rot_1_w', type=float, default=8.0, help='')
    parser.add_argument('--rot_2_w', type=float, default=8.0, help='')
    parser.add_argument('--rot_regular', type=float, default=4.0, help='')
    parser.add_argument('--tran_w', type=float, default=8.0, help='')
    parser.add_argument('--size_w', type=float, default=8.0, help='')
    parser.add_argument('--recon_w', type=float, default=8.0, help='')
    parser.add_argument('--r_con_w', type=float, default=1.0, help='')

    parser.add_argument('--recon_n_w', type=float, default=3.0, help='normal estimation loss')
    parser.add_argument('--recon_d_w', type=float, default=3.0, help='dis estimation loss')
    parser.add_argument('--recon_v_w', type=float, default=1.0, help='voting loss weight')
    parser.add_argument('--recon_s_w', type=float, default=0.3, help='point sampling loss weight, important')
    parser.add_argument('--recon_f_w', type=float, default=1.0, help='confidence loss')
    parser.add_argument('--recon_bb_r_w', type=float, default=1.0, help='bbox r loss')
    parser.add_argument('--recon_bb_t_w', type=float, default=1.0, help='bbox t loss')
    parser.add_argument('--recon_bb_s_w', type=float, default=1.0, help='bbox s loss')
    parser.add_argument('--recon_bb_self_w', type=float, default=1.0, help='bb self')


    parser.add_argument('--mask_w', type=float, default=1.0, help='obj_mask_loss')

    parser.add_argument('--geo_p_w', type=float, default=1.0, help='geo point mathcing loss')
    parser.add_argument('--geo_s_w', type=float, default=10.0, help='geo symmetry loss')
    parser.add_argument('--geo_f_w', type=float, default=0.1, help='geo face loss, face must be consistent with the point cloud')

    parser.add_argument('--prop_pm_w', type=float, default=2.0, help='')
    parser.add_argument('--prop_sym_w', type=float, default=1.0, help='importtannt for symmetric objects, can do point aug along reflection plane')
    parser.add_argument('--prop_r_reg_w', type=float, default=1.0, help='rot confidence must be sum to 1')
    
    
    """ nocs_mrcnn testing"""
    parser.add_argument('--img_size', type=int, default=224, help='cropped image size')
    parser.add_argument('--result_dir', type=str, default='', help='result directory')
    parser.add_argument('--model_dir_list', nargs='+')
    parser.add_argument('--energy_model_dir', type=str, default='', help='energy network ckpt directory')
    parser.add_argument('--score_model_dir', type=str, default='', help='score network ckpt directory')
    parser.add_argument('--ranker', type=str, default='energy_ranker', help='energy_ranker, gt_ranker or random')
    parser.add_argument('--pooling_mode', type=str, default='nearest', help='nearest or average')

    cfg = parser.parse_args()
    # dynamic zoom in parameters
    cfg.DYNAMIC_ZOOM_IN_PARAMS = {
        'DZI_PAD_SCALE': 1.5,
        'DZI_TYPE': 'uniform',
        'DZI_SCALE_RATIO': 0.25,
        'DZI_SHIFT_RATIO': 0.25
    }
    
    # pts aug parameters
    # cfg.PTS_AUG_PARAMS = {
    #     'aug_pc_pro': 0.2,
    #     'aug_pc_r': 0.2,
    #     'aug_rt_pro': 0.3,
    #     'aug_bb_pro': 0.3,
    #     'aug_bc_pro': 0.3
    # }
    
    # To solve overfitting
    cfg.PTS_AUG_PARAMS = {
        'aug_pc_pro': 0.4,
        'aug_pc_r': 0.4,
        'aug_rt_pro': 0.6,
        'aug_bb_pro': 0.6,
        'aug_bc_pro': 0.6
    }
    # 2D aug parameters
    cfg.DEFORM_2D_PARAMS = {
        'roi_mask_r': 3,
        'roi_mask_pro': 0.5
    }



    
    return cfg