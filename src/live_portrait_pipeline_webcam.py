import torch
torch.backends.cudnn.benchmark = True  # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2
import numpy as np
import os.path as osp

from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.utils.cropper import Cropper
from src.utils.camera import get_rotation_matrix
from src.utils.io import load_image_rgb, resize_to_limit
from src.utils.helper import dct2device
from src.live_portrait_wrapper import LivePortraitWrapper

def dummy_crop(frame):
    return frame

class LivePortraitPipeline(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg)
        self.inference_cfg = inference_cfg

        # Load and preprocess the source image
        img_rgb = load_image_rgb(inference_cfg.source_image)
        img_rgb = resize_to_limit(img_rgb, self.inference_cfg.source_max_dim, self.inference_cfg.source_division)

        crop_info = self.cropper.crop_source_image(img_rgb, self.cropper.crop_cfg)
        if crop_info is None:
            raise Exception("No face detected in the source image!")
        source_lmk = crop_info['lmk_crop']
        img_crop, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']

        if self.inference_cfg.flag_do_crop:
            self.I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
        else:
            img_crop_256x256 = cv2.resize(img_rgb, (256, 256))
            self.I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)

        self.x_s_info = self.live_portrait_wrapper.get_kp_info(self.I_s)
        self.x_c_s = self.x_s_info['kp']
        self.R_s = get_rotation_matrix(self.x_s_info['pitch'], self.x_s_info['yaw'], self.x_s_info['roll'])
        self.f_s = self.live_portrait_wrapper.extract_feature_3d(self.I_s)
        self.x_s = self.live_portrait_wrapper.transform_keypoint(self.x_s_info)

    def process_frame(self, frame):
        if self.inference_cfg.flag_crop_driving_video:
            crop_info = self.cropper.crop_source_image(frame, self.cropper.crop_cfg)
            if crop_info is not None:
                frame = crop_info['img_crop']

        frame_resized = cv2.resize(frame, (512, 512))
        cv2.imshow('Webcam frame', frame_resized)
        I_d = self.live_portrait_wrapper.prepare_driving_videos([frame_resized])[0]
        x_d_info = self.live_portrait_wrapper.get_kp_info(I_d)
        R_d = get_rotation_matrix(x_d_info['pitch'], x_d_info['yaw'], x_d_info['roll'])

        R_new = (R_d @ self.R_s.permute(0, 2, 1)) @ self.R_s
        delta_new = self.x_s_info['exp'] + (x_d_info['exp'] - self.x_s_info['exp'])
        scale_new = self.x_s_info['scale'] * (x_d_info['scale'] / self.x_s_info['scale'])
        t_new = self.x_s_info['t'] + (x_d_info['t'] - self.x_s_info['t'])

        t_new[..., 2].fill_(0)
        x_d_new = scale_new * (self.x_c_s @ R_new + delta_new) + t_new

        # Eye and lip retargeting
        eyes_delta, lip_delta = None, None
        if self.inference_cfg.flag_eye_retargeting:
            combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(x_d_info['eye_ratio'], self.x_s_info['eye_ratio'])
            eyes_delta = self.live_portrait_wrapper.retarget_eye(self.x_s, combined_eye_ratio_tensor)
        if self.inference_cfg.flag_lip_retargeting:
            combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio(x_d_info['lip_ratio'], self.x_s_info['lip_ratio'])
            lip_delta = self.live_portrait_wrapper.retarget_lip(self.x_s, combined_lip_ratio_tensor)

        if self.inference_cfg.flag_relative_motion:
            x_d_new = self.x_s + \
                (eyes_delta.reshape(-1, self.x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                (lip_delta.reshape(-1, self.x_s.shape[1], 3) if lip_delta is not None else 0)
        else:
            x_d_new = x_d_new + \
                (eyes_delta.reshape(-1, self.x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                (lip_delta.reshape(-1, self.x_s.shape[1], 3) if lip_delta is not None else 0)

        if self.inference_cfg.flag_stitching:
            x_d_new = self.live_portrait_wrapper.stitching(self.x_s, x_d_new)

        out = self.live_portrait_wrapper.warp_decode(self.f_s, self.x_s, x_d_new)
        I_p = self.live_portrait_wrapper.parse_output(out['out'])[0]
        I_p = cv2.resize(I_p, (1024, 1024))
        # Convert from RGB to BGR
        I_p_bgr = cv2.cvtColor(I_p, cv2.COLOR_RGB2BGR)

        # Show the result in a window
        cv2.imshow('Live Portrait', I_p_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True
