import torch
import tqdm
import imageio
import constants
import config
from models.transformer_basics import TranformerConfig
from argparse import ArgumentParser
from torchvision import transforms
from torch.utils.data import DataLoader
from data_functions.human_mesh_tsv import MeshTSVYamlDataset
from utils.geometry import batch_rodrigues_v2, compute_weak_perspective_cam
from models.SMPL_handler import SMPLHandler
from utils.graph_utils import adj_mx_from_skeleton
from models.whole_model import model0
from torchsummary import summary



yaml_file = "../SMPLer/data/3dpw/train.yaml"
is_train = True  # 或 False，取决于你是否在训练
cv2_output = False  # 根据需要设置
scale_factor = 1  # 根据需要设置


# 2. 创建 DataLoader 实例
batch_size = 32  # 根据需要设置
num_workers = 4  # 根据需要设置并行加载的工作线程数



class MyDataProcessor:
    def __init__(self, device):
        self.device = device
        self.gt_smpl_handler = SMPLHandler(path_to_regressor="meta_data/J_regressor_3dpw.npy").to(self.device)  
    
    def gt_smpl_handler(self, gt_pose, gt_betas, pose_format):
        # Dummy implementation for demonstration
        return {
            'vertices': torch.randn_like(gt_pose), 
            'vertices_minus_pelvis': torch.randn_like(gt_pose),
            'joints': torch.randn_like(gt_pose)
        }
    
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
        gt_3d_pelvis = gt_3d_joints[:, constants.J24_NAME.index('Pelvis'), :3]
        gt_3d_joints = gt_3d_joints[:, constants.J24_TO_J14, :]
        gt_3d_joints_minus_pelvis = gt_3d_joints.clone()
        gt_3d_joints_minus_pelvis[:, :, :3] = gt_3d_joints[:, :, :3] - gt_3d_pelvis[:, None, :]
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
        has_cam = torch.logical_and(has_smpl == 1, has_2d_joints == 1)
        gt_cam = compute_weak_perspective_cam(gt_3d_joints_from_smpl[has_cam], gt_2d_joints[has_cam, :, 0:-1], gt_2d_joints[has_cam, :, -1])
        
        return {'images': images, 'img_paths': img_paths, 'ori_img': ori_img, 
                'gt_2d_joints': gt_2d_joints, 'has_2d_joints': has_2d_joints, 'gt_cam': gt_cam, 'has_gt_cam': has_cam,
                'gt_3d_joints': gt_3d_joints, 'gt_3d_joints_minus_pelvis': gt_3d_joints_minus_pelvis, 'has_3d_joints': has_3d_joints, 
                'gt_pose': gt_pose, 'gt_betas': gt_betas, 'has_smpl': has_smpl, 'gt_vertices': gt_vertices, 'gt_vertices_minus_pelvis': gt_vertices_minus_pelvis}

if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data_processor = MyDataProcessor(device)

    # # 创建 MeshTSVYamlDataset 实例
    # dataset = MeshTSVYamlDataset(yaml_file, is_train=is_train, cv2_output=cv2_output, scale_factor=scale_factor)
    # # 创建 DataLoader 实例
    # dataloader = DataLoader(
    # dataset,
    # batch_size=batch_size,
    # shuffle=True if is_train else False,
   
    #  # 如果使用 GPU，建议将 pin_memory 设置为 True
    # )
    # for iter, batch in tqdm.tqdm(enumerate(dataloader)):
    #     batch_dict = data_processor.load_batch(batch)
    #     print(batch_dict["images"])
    #     break
    # Basic arguments
    parser = ArgumentParser()
    # video
    # parser.add_argument('--video_path', required=True, type=str, help='Input video path') 
    parser.add_argument('--img_path', default='./samples/im01.png', type=str, help='Input image path')
    parser.add_argument('--device', default='cpu', type=str, help='Device to run the model')
    # parser.add_argument('--udp_host', default='127.0.0.1', type=str, help='UDP server host address')
    # parser.add_argument('--udp_port', default=12345, type=int, help='UDP server port number')
    args = parser.parse_args()

    args.data_mode = '3dpw'
    args.model_type = 'smpler'
    args.hrnet_type = 'w32'
    args.num_transformers = 3
    args.load_checkpoint = 'pretrained/SMPLer_h36m.pt'

    trans_cfg = TranformerConfig()
    trans_cfg.raw_feat_dim = config.hrnet_dict[args.hrnet_type][2]

    model = model0(args, trans_cfg)
    print(model)

    
    model.eval()
    model.to(args.device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


    # img = imageio.imread(args.img_path)
    # img = transform(img)[None].to(args.device)
    # img_vis = img * torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(args.device) + torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(args.device)
    # print(img.shape)

    # pred_smpl_dict = model(img)[-1]
    
    

