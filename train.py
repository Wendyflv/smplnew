import os
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio
import tqdm
import glob
import json
from datetime import datetime, timedelta

from torch.utils.data import DataLoader
from models.transformer_basics import TranformerConfig

from models.whole_model import model0

from utils.geometry import batch_rodrigues_v2, compute_weak_perspective_cam
from utils.eval_metrics import mean_per_joint_position_error, mean_per_vertex_error, reconstruction_error, pytorch_reconstruction_error


import config
import constants
from utils import misc
from models.SMPL_handler import SMPLHandler
from data_functions.human_mesh_tsv import MeshTSVYamlDataset

class Trainer:
    def __init__(self, args):
        self.args = args
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_only = args.eval_only
        

        # Set random seed
        misc.set_seed(self.args.seed, False, True)

        if not self.eval_only:
            self.train_dataset = MeshTSVYamlDataset(config.PW3D_train_yaml, True, False, 1)
        self.val_dataset = MeshTSVYamlDataset(config.PW3D_val_yaml, False, False,1)

        if not self.eval_only:
            print('Dataset finished: train-{}, test-{}'.format(len(self.train_dataset), len(self.val_dataset)))
        else:
            print('Dataset finished: test-{}'.format(len(self.val_dataset)))

        # model
        trans_cfg = TranformerConfig()
        trans_cfg.raw_feat_dim = config.hrnet_dict[args.hrnet_type][2]

        self.model = model0(args, trans_cfg)
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params: {:.1f}M'.format(n_parameters/(1024**2)))

        self.model.to(self.device)

        
        if not self.eval_only:
            self.global_iter = 0
            self.start_epoch = 0
            self.optimizer = self.prepare_optimizer()
            #self.auto_load()

            self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size)

        self.val_dataloader = DataLoader(self.val_dataset, batch_size=args.val_batch_size)

        # joint loss criterion
        if self.args.joint_criterion == 'l1':
            self.joint_criterion_func = F.l1_loss
        else:
            self.joint_criterion_func = F.mse_loss

        if self.args.data_mode == 'h36m':
            self.gt_smpl_handler = SMPLHandler(path_to_regressor=config.JOINT_REGRESSOR_H36M_correct).to(self.device)
        else:
            self.gt_smpl_handler = SMPLHandler(path_to_regressor=config.JOINT_REGRESSOR_3DPW).to(self.device)

        self.vis_loss_list = ['loss_vertices', 'loss_2d_joints', 'loss_3d_joints', 'loss_theta', 'loss_combine']

        if self.args.resume:
            checkpoint_path = os.path.join(self.args.checkpoint_dir, 'latest_checkpoint.pth')
            if os.path.exists(checkpoint_path):
                print(f'Loading model from {checkpoint_path}')
                self.model.load_state_dict(torch.load(checkpoint_path))
                self.model.to(self.device)
            else:
                print('No checkpoint found, starting training from scratch.')


       


    def auto_load(self):
        checkpoint_paths = sorted(glob.glob(os.path.join(self.checkpoint_dir, '*.pt')))
        if self.args.auto_load > 0 and len(checkpoint_paths) > 0:
            checkpoint_path = checkpoint_paths[-1]
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.start_epoch = state_dict['epoch'] + 1
            self.global_iter = state_dict['global_iter']
            print(f'{checkpoint_path} is loaded!')

    def prepare_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=0)
        return optimizer

    @staticmethod
    def load_checkpoint(model, checkpoint_path, edit_state_dict=0):
        old_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'optimizer' in old_dict:
            old_dict = old_dict['model']
        if edit_state_dict > 0:
            new_dict = {}
            for k, v in old_dict.items():
                if "J_regressor_h36m_correct" not in k:
                    new_dict[k] = v
            model.load_state_dict(new_dict, strict=False)
        else:
            model.load_state_dict(old_dict, strict=False)
        print(f'{checkpoint_path} is loaded!')

    def compute_loss_2djoint(self, pred, gt, has_gt=None):
        if len(gt) > 0:
            conf = gt[:, :, -1].unsqueeze(-1).clone()
            return (conf * self.joint_criterion_func(pred, gt[:, :, :-1], reduction='none')).mean()
        else:
            return torch.tensor(0.0, device=self.device)

    def compute_loss_3djoint(self, pred, gt, has_gt, midpoint_as_origin=True):
        gt = gt[has_gt == 1]
        pred = pred[has_gt == 1]
        if len(gt) > 0:
            conf = gt[:, :, -1].unsqueeze(-1).clone()
            gt = gt[:, :, :-1].clone()
            if midpoint_as_origin:
                gt_pelvis = (gt[:, 2, :] + gt[:, 3, :]) / 2
                gt = gt - gt_pelvis[:, None, :]
                pred_pelvis = (pred[:, 2, :] + pred[:, 3, :]) / 2
                pred = pred - pred_pelvis[:, None, :]
            return (conf * self.joint_criterion_func(pred, gt, reduction='none')).mean()
        else:
            return torch.tensor(0.0, device=self.device)
    
    def compute_loss_3djoint_PA(self, pred, gt, has_gt):
        gt = gt[has_gt == 1]
        pred = pred[has_gt == 1]
        if len(gt) > 0:
            conf = gt[:, :, -1].clone()
            gt = gt[:, :, :-1].clone()
            return (conf * pytorch_reconstruction_error(pred, gt, reduction='none')).mean()
        else:
            return torch.tensor(0.0, device=self.device)
    
    def compute_loss_J_Tpose(self, pred, gt, has_gt):
        gt = gt[has_gt == 1]
        pred = pred[has_gt == 1]
        if len(gt) > 0:
            return F.l1_loss(pred, gt, reduction='mean')
        else:
            return torch.tensor(0.0, device=self.device)

    def compute_loss_vertices(self, pred, gt, has_gt=None):
        if len(gt) > 0:
            return F.mse_loss(pred[has_gt == 1], gt[has_gt == 1], reduction='mean')
        else:
            return torch.tensor(0.0, device=self.device)
        
    def compute_loss_theta_beta(self, pred, gt, has_gt):
        pred = pred[has_gt == 1]
        gt = gt[has_gt == 1]
        if len(gt) > 0:
            return F.l1_loss(pred, gt) 
        else:
            return torch.tensor(0.0, device=self.device)
    

    def load_batch(self, batch):
        img_paths, images, annotations = batch
        images = images.to(self.device)
        ori_img = annotations['ori_img'].to(self.device)

        # GT 2d keypoint
        gt_2d_joints = annotations['joints_2d'].to(self.device)
        gt_2d_joints = gt_2d_joints[:, constants.J24_TO_J14, :]
        has_2d_joints = annotations['has_2d_joints'].to(self.device)

        # GT 3d keypoint
        gt_3d_joints = annotations['joints_3d'].to(self.device)
        gt_3d_pelvis = gt_3d_joints[:,constants.J24_NAME.index('Pelvis'),:3]
        gt_3d_joints = gt_3d_joints[:,constants.J24_TO_J14,:] 
        gt_3d_joints_minus_pelvis = gt_3d_joints.clone()
        gt_3d_joints_minus_pelvis[:,:,:3] = gt_3d_joints[:,:,:3] - gt_3d_pelvis[:, None, :]
        has_3d_joints = annotations['has_3d_joints'].to(self.device)

        # GT smpl
        gt_pose = annotations['pose'].to(self.device)
        gt_betas = annotations['betas'].to(self.device)
        has_smpl = annotations['has_smpl'].to(self.device)

        gt_smpl_dict = self.gt_smpl_handler(gt_pose, gt_betas, 'axis-angle')
        gt_vertices = gt_smpl_dict['vertices']
        gt_vertices_minus_pelvis = gt_smpl_dict['vertices_minus_pelvis'] 

        # GT cam
        gt_3d_joints_from_smpl = gt_smpl_dict['joints']
        has_cam = torch.logical_and(has_smpl==1, has_2d_joints==1)
        gt_cam = compute_weak_perspective_cam(gt_3d_joints_from_smpl[has_cam], gt_2d_joints[has_cam, :, 0:-1], gt_2d_joints[has_cam, :, -1])
        
        return {'images': images, 'img_paths': img_paths, 'ori_img': ori_img, 
                'gt_2d_joints': gt_2d_joints, 'has_2d_joints': has_2d_joints, 'gt_cam': gt_cam, 'has_gt_cam': has_cam,
                'gt_3d_joints': gt_3d_joints, 'gt_3d_joints_minus_pelvis': gt_3d_joints_minus_pelvis, 'has_3d_joints': has_3d_joints, 
                'gt_pose': gt_pose, 'gt_betas': gt_betas, 'has_smpl': has_smpl, 'gt_vertices': gt_vertices, 'gt_vertices_minus_pelvis': gt_vertices_minus_pelvis}

    # Simplified
    def forward_step(self, batch, phase=True, visualize=True):
        # --- Load batch data ---
        batch_dict = self.load_batch(batch)

        # --- Run model ---
        model = self.model
        
        pred_smpl_dicts = model(batch_dict['images'])
        
        pred_smpl_dict = pred_smpl_dicts[-1]
        pred_rotmat = pred_smpl_dict['theta']  
        pred_vertices_minus_pelvis = pred_smpl_dict['vertices_minus_pelvis']
        pred_3d_joints_from_smpl_minus_pelvis = pred_smpl_dict['joints_minus_pelvis']
        pred_2d_joints_from_smpl = pred_smpl_dict['joints2d']

        # losses
        if phase == 'train':
            self.loss_vertices = self.compute_loss_vertices(pred_vertices_minus_pelvis, batch_dict['gt_vertices_minus_pelvis'], batch_dict['has_smpl'])
            self.loss_2d_joints = self.compute_loss_2djoint(pred_2d_joints_from_smpl, batch_dict['gt_2d_joints'], batch_dict['has_2d_joints'])
            self.loss_3d_joints = self.compute_loss_3djoint(pred_3d_joints_from_smpl_minus_pelvis, batch_dict['gt_3d_joints_minus_pelvis'], batch_dict['has_3d_joints'], midpoint_as_origin=True)
            self.loss_theta = self.compute_loss_theta_beta(pred_rotmat, batch_rodrigues_v2(batch_dict['gt_pose']), batch_dict['has_smpl'])
            
            self.loss_combine = self.args.w_vert * self.loss_vertices + self.args.w_2dj * self.loss_2d_joints + \
                                self.args.w_3dj * self.loss_3d_joints + self.args.w_theta * self.loss_theta  
            
        else:
            error_vertices = mean_per_vertex_error(pred_vertices_minus_pelvis.detach(), batch_dict['gt_vertices_minus_pelvis'], batch_dict['has_smpl'])
            error_joints = mean_per_joint_position_error(pred_3d_joints_from_smpl_minus_pelvis.detach(), batch_dict['gt_3d_joints_minus_pelvis'], batch_dict['has_3d_joints'])
            error_joints_pa = reconstruction_error(pred_3d_joints_from_smpl_minus_pelvis.detach().cpu().numpy(), batch_dict['gt_3d_joints_minus_pelvis'][:,:,:3].cpu().numpy(), reduction=None)
            self.mpve_sum += np.sum(error_vertices)    # mean per-vertex error
            self.mpve_count += torch.sum(batch_dict['has_smpl']).item()
            self.mpjpe_sum += np.sum(error_joints)   # mean per-joint position error
            self.mpjpepa_sum += np.sum(error_joints_pa)
            self.mpjpe_count += torch.sum(batch_dict['has_3d_joints']).item()

        # for visualization
        self.images = batch_dict['ori_img']
        self.pred_vertices = pred_smpl_dict['vertices']
        self.pred_2d_joints_from_smpl = pred_smpl_dict['joints2d']
        self.pred_cam = pred_smpl_dict['cam']
        self.gt_vertices = batch_dict['gt_vertices']
        self.gt_2d_joints = batch_dict['gt_2d_joints']
        self.gt_cam = self.pred_cam.clone() 

    def run_training(self):
        print('start training...')
        if not os.path.exists(self.args.checkpoint_dir):
            os.makedirs(self.args.checkpoint_dir)
        
        for epoch in tqdm.tqdm(range(self.start_epoch, self.num_epochs)):
            self.model.train()
            epoch_loss = 0
            for (iter , batch) in tqdm.tqdm(enumerate(self.train_dataloader), total = len(self.train_dataloader), desc=f'Train-{epoch:03}'):
                self.global_iter +=1 
                self.forward_step(batch, phase='train')

                self.optimizer.zero_grad()
                self.loss_combine.backward()

                if self.args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)

                self.optimizer.step()

                print("loss:", self.loss_combine)

                
            if self.args.save_interval > 0 and (epoch + 1) % self.args.save_interval == 0:
                checkpoint_path = os.path.join(self.args.checkpoint_dir, f'checkpoint_{epoch + 1}.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f'Saved checkpoint to {checkpoint_path}')

        

            if epoch % self.args.eval_epochs == 0:
                self.run_evaluation(epoch)

    @torch.no_grad()
    def run_evaluation(self, epoch=0, do_tb=False, do_render=True):
        self.model.eval()
        # 初始化评价指标
        self.mpve_sum = 0
        self.mpve_count = 0
        self.mpjpe_sum = 0
        self.mpjpepa_sum = 0
        self.mpjpe_count = 0

        # 遍历验证数据集
        for batch in tqdm.tqdm(self.val_dataloader, total=len(self.val_dataloader), desc=f'Eval-{epoch:03}'):
            self.forward_step(batch, phase='eval')

        # 计算评价指标
        mpve = self.mpve_sum / (self.mpve_count + 1e-8)
        mpjpe = self.mpjpe_sum / self.mpjpe_count
        mpjpepa = self.mpjpepa_sum / self.mpjpe_count

        # 输出结果和可视化
        if do_render:
            rendered = self.visualizer.draw_skeleton_and_mesh(
                self.images, self.gt_2d_joints, self.pred_2d_joints_from_smpl, 
                self.gt_vertices, self.pred_vertices, self.gt_cam, self.pred_cam, num_draws=6
            )

        if do_tb:
            self.writer.add_scalar('mpve', mpve, epoch)
            self.writer.add_scalar('mpjpe', mpjpe, epoch)
            self.writer.add_scalar('mpjpe-pa', mpjpepa, epoch)
            if do_render:
                self.writer.add_image('eval-vis', rendered, epoch, dataformats='HWC')
        else:
            print(f'mpve:         {mpve*1000:.3f}')
            print(f'mpjpe:        {mpjpe*1000:.3f}')
            print(f'mpjpe-pa:     {mpjpepa*1000:.3f}')
            if do_render:
                save_path = self.args.load_checkpoint[:-3] + '.png'
                imageio.imwrite(save_path, (rendered*255).astype(np.uint8))

        # 恢复模型为训练模式
        self.model.train()

    def run(self):
        if self.eval_only:
            self.run_evaluation(do_tb=False, do_render=False)
        else:
            self.run_training()


if __name__ == '__main__':
    from utils.argument_manager import ArgManager
    arg_manager = ArgManager()
    
    trainer = Trainer(arg_manager.args)
    trainer.run()

     


            







''