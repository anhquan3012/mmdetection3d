import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from .. import builder
from .two_stage import TwoStage3DDetector


@DETECTORS.register_module()
class FusionRCNN(TwoStage3DDetector):
    """
    FusionRCNN detector
    https://arxiv.org/abs/2209.10733 
    """

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 img_backbone = None,
                 img_neck = None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FusionRCNN, self).__init__(backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        
        # self.freeze_img = kwargs.get('freeze_img', True)
        # self.init_weights(pretrained=kwargs.get('pretrained', None))

    # def init_weights(self, pretrained=None):
    #     """Initialize model weights."""
    #     super(FusionRCNN, self).init_weights(pretrained)

    #     if self.freeze_img:
    #         if self.with_img_backbone:
    #             for param in self.img_backbone.parameters():
    #                 param.requires_grad = False
    #         if self.with_img_neck:
    #             for param in self.img_neck.parameters():
    #                 param.requires_grad = False

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_(0)
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)
        img_feats = self.img_backbone(img.float())

        img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_pts_feat(self, pts, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        img_feats = self.extract_img_feat(img, img_metas=img_metas)
        pts_feats = self.extract_pts_feat(points, img_metas=img_metas)

        losses = dict()
        if self.with_rpn:
            rpn_outs = self.rpn_head(pts_feats)
            rpn_loss_inputs = rpn_outs + (gt_bboxes_3d, gt_labels_3d,
                                          img_metas)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(points, img_feats,
                                                 img_metas, proposal_list,
                                                 gt_bboxes_3d, gt_labels_3d)
        losses.update(roi_losses)
        return losses

    # def simple_test_pts(self, x, x_img, img_metas, rescale=False):
    #     """Test function of point cloud branch."""
    #     outs = self.pts_bbox_head(x, x_img, img_metas)
    #     bbox_list = self.pts_bbox_head.get_bboxes(
    #         outs, img_metas, rescale=rescale)
    #     bbox_results = [
    #         bbox3d2result(bboxes, scores, labels)
    #         for bboxes, scores, labels in bbox_list
    #     ]
    #     return bbox_results

    def simple_test(self, points, img_metas, img=None, proposals=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_img_feat(img, img_metas=img_metas)
        pts_feats = self.extract_pts_feat(points, img_metas=img_metas)

        if self.with_rpn:
            rpn_outs = self.rpn_head(pts_feats)
            proposal_cfg = self.test_cfg.rpn
            bbox_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*bbox_inputs)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(points, img_feats, img_metas,
                                         proposal_list)
