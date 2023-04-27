# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from mmdet3d.core import AssignResult
from mmdet3d.core.bbox import bbox3d2result, bbox3d2roi
from mmdet.core import build_assigner, build_sampler
from ..builder import HEADS, build_head
from mmdet.models.builder import build_roi_extractor
from .base_3droi_head import Base3DRoIHead
from  ...core.bbox.structures.utils import rotation_3d_in_axis
from ..model_utils.fusion_rcnn_utils import MLP, Transformer

from mmdet3d.core.bbox.structures import (get_proj_mat_by_coord_type, points_cam2img)

import numpy as np
from numpy import *

@HEADS.register_module()
class FusionRCNNHead(Base3DRoIHead):
    def __init__(self,
                 num_classes=3,
                 rpn_expand_ratio = 2,
                 num_samples = 256,
                 num_queries=1,
                 output_dim=256,
                 image_roi_extractor=None,
                 transformer=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FusionRCNNHead, self).__init__(
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        
        self.up_dimension = MLP(input_dim = 28, hidden_dim = 64, output_dim = 256, num_layers = 3)
        self.image_roi_extractor = build_roi_extractor(image_roi_extractor)
        
        self.query_embed = nn.Embedding(num_queries, output_dim)
        self.transformer = Transformer(**transformer)



        self.rpn_expand_ratio = rpn_expand_ratio
        self.num_samples = num_samples

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            if isinstance(self.train_cfg.assigner, dict):
                self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            elif isinstance(self.train_cfg.assigner, list):
                self.bbox_assigner = [
                    build_assigner(res) for res in self.train_cfg.assigner
                ]
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)

    def expand_rpn(self, original_rpn_tensor, rpn_expand_ratio):
        device = original_rpn_tensor.device
        expand_tensor = torch.FloatTensor([1.0, 1.0, 1.0, 
                                        rpn_expand_ratio, 
                                        rpn_expand_ratio,
                                        rpn_expand_ratio, 
                                        1]).to(device)
        height_shift = original_rpn_tensor[:, 5] / 2 * (rpn_expand_ratio - 1) # (N, 1)
        original_rpn_tensor[:, 2] = original_rpn_tensor[:, 2] - height_shift
        expanded_rpn_tensor = original_rpn_tensor*expand_tensor
        return expanded_rpn_tensor
    
    def points_inside_rpn(self, points, rpn_tensor, rpn_centers, num_samples, yaw_axis):
        num_rois = rpn_tensor.shape[0]
        points_rep = points[:,:3]
        points_rep = points_rep.repeat(num_rois, 1, 1) # (N, M, 3)

        points_rep_shifted = points_rep - rpn_centers.view(-1, 1, 3)
        points_relative = rotation_3d_in_axis(points_rep_shifted, -rpn_tensor[:, 6], axis=yaw_axis)
        points_mask = torch.all(torch.abs(points_relative) <= (rpn_tensor[:, 3:6].view(-1, 1, 3) / 2), dim=-1)

        src = rpn_tensor.new_zeros(num_rois, num_samples, 4)

        for rpn_idx in range(num_rois):
            cur_points = points[points_mask[rpn_idx]]
            if cur_points.shape[0] >= num_samples:
                random.seed(0)
                index = np.random.randint(cur_points.shape[0], size=num_samples)
                cur_roi_points_sample = cur_points[index]

            elif cur_points.shape[0] == 0:
                cur_roi_points_sample = cur_points.new_zeros(num_samples, 4)

            else:
                empty_num = num_samples - cur_points.shape[0]
                add_zeros = cur_points.new_zeros(empty_num, 4)
                add_zeros = cur_points[0].repeat(empty_num, 1)
                cur_roi_points_sample = torch.cat([cur_points, add_zeros], dim = 0)
            src[rpn_idx, : , : ] = cur_roi_points_sample 
        
        return src  # (N, num_samples, 4)
    
    def spherical_coordinate(self, src, diag_dist):
        assert (src.shape[-1] == 27)
        device = src.device
        indices_x = torch.LongTensor([0,3,6,9,12,15,18,21,24]).to(device)  
        indices_y = torch.LongTensor([1,4,7,10,13,16,19,22,25]).to(device) 
        indices_z = torch.LongTensor([2,5,8,11,14,17,20,23,26]).to(device) 
        src_x = torch.index_select(src, -1, indices_x)
        src_y = torch.index_select(src, -1, indices_y)
        src_z = torch.index_select(src, -1, indices_z)
        dis = (src_x ** 2 + src_y ** 2 + src_z ** 2) ** 0.5
        phi = torch.atan(src_y / (src_x + 1e-5))
        the = torch.acos(src_z / (dis + 1e-5))
        dis = dis / diag_dist
        src = torch.cat([dis, phi, the], dim = -1)
        return src

    def point_feats_extractor(self, src, rpn_tensor, rpn_centers, rpn_corners, num_samples):
        num_rois = rpn_tensor.shape[0] 
        corner_points = rpn_corners.view(num_rois, -1) # (N, 24)
        corner_add_center_points = torch.cat([corner_points, rpn_centers], dim = -1) # (N, 27)

        # vector embedding of distance to each corner and the corner of the RPN 
        # (N, num_samples, 27)
        pos_fea = src[:,:,:3].repeat(1,1,9) - corner_add_center_points.unsqueeze(1).repeat(1,num_samples,1) 
        lwh = rpn_tensor[:,3:6].unsqueeze(1).repeat(1,num_samples,1) # (N, num_samples, 3)
        diag_dist = (lwh[:,:,0]**2 + lwh[:,:,1]**2 + lwh[:,:,2]**2) ** 0.5 # (N, num_samples)
        pos_fea = self.spherical_coordinate(pos_fea, diag_dist = diag_dist.unsqueeze(-1)) # (N, num_samples, 27)
        src = torch.cat([pos_fea, src[:,:,-1].unsqueeze(-1)], dim = -1) # add intensity features, (N, num_samples, 28)
        src = self.up_dimension(src)  # (N, num_samples, 256)
        return src

    def corners(self, rpn_tensor, yaw_axis):
        """torch.Tensor: Coordinates of corners of all the boxes
        in shape (N, 8, 3).

        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

        .. code-block:: none

                                           up z
                            front x           ^
                                 /            |
                                /             |
                  (x1, y0, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / origin    | /
            left y<-------- + ----------- + (x0, y1, z0)
                (x0, y0, z0)
        """
        if rpn_tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=rpn_tensor.device)

        dims = rpn_tensor[:, 3:6]
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin [0.5, 0.5, 0]
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        corners = rotation_3d_in_axis(corners, rpn_tensor[:, 6], axis=yaw_axis)
        corners += rpn_tensor[:, :3].view(-1, 1, 3)
        return corners
    
    def bbox3dtobbox2d(self, corners, img_meta, batch_idx):
        img_scale_factor = (
            corners.new_tensor(img_meta['scale_factor'][:2])
            if 'scale_factor' in img_meta.keys() else 1)
        img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
        img_crop_offset = (
            corners.new_tensor(img_meta['img_crop_offset'])
            if 'img_crop_offset' in img_meta.keys() else 0)
        
        proj_mat = get_proj_mat_by_coord_type(img_meta, 'LIDAR')
        N = corners.shape[0]
        
        bboxes_2d = corners.new_full((N, 5), batch_idx)

        for roi_idx in range(corners):
            curr_corners_2d = points_cam2img(corners[roi_idx], proj_mat) # (8, 2)

            curr_corners_2d = curr_corners_2d * img_scale_factor # (8, 2)
            curr_corners_2d = curr_corners_2d - img_crop_offset # (8, 2)

            if img_flip:
                img_shape = img_meta['img_shape'][:2]
                coor_x, coor_y = torch.split(curr_corners_2d, 1, dim=1)
                # by default we take it as horizontal flip
                # use img_shape before padding for flip
                orig_h, orig_w = img_shape
                coor_x = orig_w - coor_x
                curr_corners_2d = torch.cat([coor_x, coor_y], dim=1)

            br = torch.max(curr_corners_2d, 0).values
            tl = torch.min(curr_corners_2d, 0).values
            roi = torch.cat([tl, br], dim=-1)
            bboxes_2d[roi_idx, 1: ] = roi
        return bboxes_2d

    def forward_train(self, points, img_feats, img_metas, proposal_list,
                      gt_bboxes_3d, gt_labels_3d):
        batch_size = len(proposal_list)
        batch_point_feats = [] # list[(N, num_samples, 256)]
        batch_bboxes_2d = []

        for batch_idx in range(batch_size):
            original_rpn_object = proposal_list[batch_idx][0] 
            rpn_centers = original_rpn_object.gravity_center
            curr_points = points[batch_idx]
            yaw_axis = original_rpn_object.YAW_AXIS
            expanded_rpn_tensor = self.expand_rpn(original_rpn_object.tensor, 
                                                  self.rpn_expand_ratio) # (N, 7)
            expanded_rpn_corners = self.corners(expanded_rpn_tensor, 
                                                yaw_axis) # (N, 8, 3)
            point_feats = self.points_inside_rpn(curr_points, expanded_rpn_tensor, rpn_centers, self.num_samples, yaw_axis)
            point_feats = self.point_feats_extractor(point_feats, 
                                    expanded_rpn_tensor, 
                                    rpn_centers, 
                                    expanded_rpn_corners, 
                                    self.num_samples) # (N, num_samples, 256)
            bboxes_2d = self.bbox3dtobbox2d(expanded_rpn_corners, img_metas[batch_idx], batch_idx)
            batch_point_feats.append(point_feats)
            batch_bboxes_2d.append(bboxes_2d)
        
        batch_bboxes_2d = torch.cat(batch_bboxes_2d, dim=0)
        batch_img_feats = self.image_roi_extractor(img_feats[:self.image_roi_extractor.num_inputs], batch_bboxes_2d) # (B*N, 256, 7, 7)
        batch_img_feats = batch_img_feats.view(batch_img_feats.size(0), batch_img_feats.size(1), -1).permute(0,2,1) # (B*N, 49, 256)
        batch_point_feats = torch.concat(batch_point_feats, dim = 0) # (B*N, num_samples, 256)

        
        hs = self.transformer(batch_point_feats, batch_img_feats, self.query_embed.weight) # (bs, 1, c)
        

        # assign and sample
        sample_results = self._assign_and_sample(proposal_list, gt_bboxes_3d,
                                                 gt_labels_3d)
        # expand 
        rois = proposal_list['boxes_3d']

        # 

        return losses
        
    @torch.no_grad()
    def _assign_and_sample(self, proposal_list, gt_bboxes_3d, gt_labels_3d):
        """Assign and sample proposals for training.

        Args:
            proposal_list (list[dict]): Proposals produced by RPN.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels

        Returns:
            list[:obj:`SamplingResult`]: Sampled results of each training
                sample.
        """
        sampling_results = []
        # bbox assign
        for batch_idx in range(len(proposal_list)):
            cur_proposal_list = proposal_list[batch_idx]
            cur_boxes = cur_proposal_list[0]
            cur_labels_3d = cur_proposal_list[2]
            cur_gt_bboxes = gt_bboxes_3d[batch_idx].to(cur_boxes.device)
            cur_gt_labels = gt_labels_3d[batch_idx]
            batch_num_gts = 0
            # 0 is bg
            batch_gt_indis = cur_gt_labels.new_full((len(cur_boxes), ), 0)
            batch_max_overlaps = cur_boxes.tensor.new_zeros(len(cur_boxes))
            # -1 is bg
            batch_gt_labels = cur_gt_labels.new_full((len(cur_boxes), ), -1)

            # each class may have its own assigner
            if isinstance(self.bbox_assigner, list):
                for i, assigner in enumerate(self.bbox_assigner):
                    gt_per_cls = (cur_gt_labels == i)
                    pred_per_cls = (cur_labels_3d == i)
                    cur_assign_res = assigner.assign(
                        cur_boxes.tensor[pred_per_cls],
                        cur_gt_bboxes.tensor[gt_per_cls],
                        gt_labels=cur_gt_labels[gt_per_cls])
                    # gather assign_results in different class into one result
                    batch_num_gts += cur_assign_res.num_gts
                    # gt inds (1-based)
                    gt_inds_arange_pad = gt_per_cls.nonzero(
                        as_tuple=False).view(-1) + 1
                    # pad 0 for indice unassigned
                    gt_inds_arange_pad = F.pad(
                        gt_inds_arange_pad, (1, 0), mode='constant', value=0)
                    # pad -1 for indice ignore
                    gt_inds_arange_pad = F.pad(
                        gt_inds_arange_pad, (1, 0), mode='constant', value=-1)
                    # convert to 0~gt_num+2 for indices
                    gt_inds_arange_pad += 1
                    # now 0 is bg, >1 is fg in batch_gt_indis
                    batch_gt_indis[pred_per_cls] = gt_inds_arange_pad[
                        cur_assign_res.gt_inds + 1] - 1
                    batch_max_overlaps[
                        pred_per_cls] = cur_assign_res.max_overlaps
                    batch_gt_labels[pred_per_cls] = cur_assign_res.labels

                assign_result = AssignResult(batch_num_gts, batch_gt_indis,
                                             batch_max_overlaps,
                                             batch_gt_labels)
            else:  # for single class
                assign_result = self.bbox_assigner.assign(
                    cur_boxes.tensor,
                    cur_gt_bboxes.tensor,
                    gt_labels=cur_gt_labels)

            # sample boxes
            sampling_result = self.bbox_sampler.sample(assign_result,
                                                       cur_boxes.tensor,
                                                       cur_gt_bboxes.tensor,
                                                       cur_gt_labels)
            sampling_results.append(sampling_result)
        return sampling_results