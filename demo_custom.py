import os
import torch
import argparse
import numpy as np
import imageio.v3 as iio

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from read_video import read_video
from cotracker.predictor import CoTrackerPredictor
from cotracker.predictor import CoTrackerOnlinePredictor

if __name__ == '__main__':

    DEFAULT_DEVICE = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="yt_samples/adelle_speck.mp4",
        help="path to a video"
    )
    parser.add_argument(
        "--mask_path",
        default=None,
        help="path to a segmentation mask"
    )

    parser.add_argument(
        "--online",
        action="store_true",
    )

    parser.add_argument(
        "--grid_size",
        default=10
    )

    # parser.add_argument(
    #     "--checkpoint",
    #     default="./checkpoints/scaled_online.pth",
    #     help="CoTracker model parameters",
    # )
    args = parser.parse_args()

    checkpoint = "./checkpoints/scaled_offline.pth"
    if args.online:
        checkpoint = "./checkpoints/scaled_online.pth"


    # video = read_video_from_path(args.video_path)
    video, _ = read_video(args.video_path, fps=10)
    segm_mask = None

    if args.online == False:
        model = CoTrackerPredictor(
            checkpoint=checkpoint,
            v2=False,
            offline=True,
            window_len=60
        )
    else:
        model = CoTrackerOnlinePredictor(
            checkpoint=checkpoint
        )

    model = model.to("cuda")

    vid_name = args.video_path[11:-4]
    # adelle_speck.mp4
    if vid_name == "adelle_speck":
        queries = torch.tensor([
            [0., 200., 280.],
            [0., 241., 268.],
            [0., 177., 192.],
            [0., 214., 183.],
            [0., 182., 121.],
            [0., 208., 117.],
            [0., 181., 65.],
            [0., 208., 64.],
            [0., 210., 348.],
            [0., 257., 341.]
        ])
    elif vid_name == "kaliya_lincoln":
    # kaliya_lincoln.mp4
        queries = torch.tensor([
            [0., 177., 185.],
            [0., 226., 185.],
            [0., 181., 258.],
            [0., 231., 251.],
            [0., 184., 258.],
            [0., 222., 353.],
            [0., 170., 414.],
            [0., 212., 422.],
            [0., 148., 488.],
            [0., 214., 514.],
            [0., 186., 154.],
            [0., 219., 147.]
        ])
    elif vid_name == "kaliya_lincoln_2":
    # kaliya_lincoln_2.mp4
        queries = torch.tensor([
            [0., 139., 270.],
            [0., 98., 303.],
            [0., 30., 327.],
            [0., 219., 358.],
            [0., 162., 370.],
            [0., 176., 427.],
            [0., 151., 427.],
            [0., 196., 528.],
            [0., 148., 524.]
        ])
    elif vid_name == "kaliya_lincoln_3":
    # kaliya_lincoln_3.mp4
        queries = torch.tensor([
            [0., 116., 228.],
            [0., 139., 223.],
            [0., 111., 259.],
            [0., 169., 255.],
            [0., 96., 307.],
            [0., 196., 270.],
            [0., 111., 294.],
            [0., 143., 296.],
            [0., 125., 328.],
            [0., 172., 328.],
            [0., 122., 392.],
            [0., 135., 364.]
        ])
    elif vid_name == "switch_leap":
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

    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame):
                video_chunk = (
                    torch.tensor(
                        np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
                    )
                    .float()
                    .permute(0, 3, 1, 2)[None]
                )  # (1, T, 3, H, W)
                return model(
                    video_chunk,
                    is_first_step=is_first_step,
                    queries=queries,
                    grid_size=grid_size,
                    grid_query_frame=grid_query_frame,
                )

    if args.online == False:
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        video = video.to("cuda")
        pred_tracks, pred_visibility = model(
            video,
            queries=queries[None]
        )

        vis = Visualizer(save_dir="./saved_videos_offline", pad_value=120, linewidth=3)
        vis.visualize(
            video,
            pred_tracks,
            query_frame=0
        )
    
    else:        
        queries = queries.unsqueeze(0)  # Adds a dimension, making shape (1, 6, 3)
        is_first_step = True
        window_frames = []

        for i, frame in enumerate(
            video
            # iio.imiter(
            #     args.video_path,
            #     plugin="FFMPEG",
            # )
        ):
            if i % model.step == 0 and i != 0:
                pred_tracks, pred_visibility = _process_step(
                    window_frames,
                    is_first_step,
                    grid_size=args.grid_size,
                    grid_query_frame=0,
                )
                is_first_step = False
            window_frames.append(frame)

        pred_tracks, pred_visibility = _process_step(
            window_frames[-(i % model.step) - model.step - 1 :],
            is_first_step,
            grid_size=args.grid_size,
            grid_query_frame=0,
        )

        seq_name = args.video_path.split("/")[-1]
        video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
            0, 3, 1, 2
        )[None]
        vis = Visualizer(save_dir="./saved_videos_online", pad_value=120, linewidth=3)
        vis.visualize(
            video, pred_tracks, query_frame=0
        )

        torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()