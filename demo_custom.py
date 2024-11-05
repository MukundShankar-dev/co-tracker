import os
import torch
import argparse
import numpy as np

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="yt_samples/switch_leap.mp4",
        help="path to a video"
    )
    parser.add_argument(
        "--mask_path",
        default=None,
        help="path to a segmentation mask"
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/scaled_offline.pth",
        # default=None,
        help="CoTracker model parameters",
    )
    args = parser.parse_args()

    video = read_video_from_path(args.video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    segm_mask = None

    model = CoTrackerPredictor(
        checkpoint=args.checkpoint,
        v2=False,
        offline=True,
        window_len=60
    )
    model = model.to("cuda")
    video = video.to("cuda")

    # [(200, 280), (241, 268), (177, 192), (214, 183), (182, 121),
    #  (208, 117), (181, 65), (208, 64), (210, 348), (257, 341)]
    
    # adelle_speck.mp4
    # queries = torch.tensor([
    #     [0., 200., 280.],
    #     [0., 241., 268.],
    #     [0., 177., 192.],
    #     [0., 214., 183.],
    #     [0., 182., 121.],
    #     [0., 208., 117.],
    #     [0., 181., 65.],
    #     [0., 208., 64.],
    #     [0., 210., 348.],
    #     [0., 257., 341.]
    # ])
    # kaliya_lincoln.mp4
    # queries = torch.tensor([
    #     [0., 177., 185.],
    #     [0., 226., 185.],
    #     [0., 181., 258.],
    #     [0., 231., 251.],
    #     [0., 184., 258.],
    #     [0., 222., 353.],
    #     [0., 170., 414.],
    #     [0., 212., 422.],
    #     [0., 148., 488.],
    #     [0., 214., 514.],
    #     [0., 186., 154.],
    #     [0., 219., 147.]
    # ])
    # kaliya_lincoln_2.mp4
    # queries = torch.tensor([
    #     [0., 139., 270.],
    #     [0., 98., 303.],
    #     [0., 30., 327.],
    #     [0., 219., 358.],
    #     [0., 162., 370.],
    #     [0., 176., 427.],
    #     [0., 151., 427.],
    #     [0., 196., 528.],
    #     [0., 148., 524.]
    # ])
    # kaliya_lincoln_3.mp4
    # queries = torch.tensor([
    #     [0., 116., 228.],
    #     [0., 139., 223.],
    #     [0., 111., 259.],
    #     [0., 169., 255.],
    #     [0., 96., 307.],
    #     [0., 196., 270.],
    #     [0., 111., 294.],
    #     [0., 143., 296.],
    #     [0., 125., 328.],
    #     [0., 172., 328.],
    #     [0., 122., 392.],
    #     [0., 135., 364.]
    # ])
    # switch_leap.mp4
    queries = torch.tensor([
        [0., 271., 275.],
        [0., 354., 227.],
        [0., 271., 452.],
        [0., 198., 554.],
        [0., 279., 592.],
        [0., 215., 285.]
    ])
    queries = queries.cuda()

    pred_tracks, pred_visibility = model(
        video,
        queries=queries[None]
    )

    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video,
        pred_tracks,
        # pred_visibility,
        query_frame=0
    )