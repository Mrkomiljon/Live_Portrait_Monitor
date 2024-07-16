# inference.py

import os.path as osp
import tyro
import subprocess
import cv2
import mss
import numpy as np
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline_monitor import LivePortraitPipeline

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source_image):
        raise FileNotFoundError(f"source image not found: {args.source_image}")


def main():
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    if not fast_check_ffmpeg():
        raise ImportError("FFmpeg is not installed. Please install FFmpeg before running this script. https://ffmpeg.org/download.html")

    fast_check_args(args)

    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    inference_cfg.source_image = args.source_image

    live_portrait_pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while True:
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if not live_portrait_pipeline.process_frame(frame):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
