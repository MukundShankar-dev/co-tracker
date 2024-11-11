import numpy as np
from decord import VideoReader


def read_video(video_path, fps=None):
    """Read video using decord VideoReader at specified fps
    
    Args:
        video_path (str): Path to video file
        fps (int): Target frames per second to extract
        
    Returns:
        frames (np.ndarray): Video frames array of shape (num_frames, H, W, C)
        actual_fps (int): Actual fps used for extraction
    """
    vr = VideoReader(video_path, num_threads=1)
    if fps is None:
        fps = vr.get_avg_fps()
    num_frames = len(vr)
    duration = vr.get_frame_timestamp(num_frames-1)[-1]
    
    # Calculate frames to extract
    frames_to_extract = int(duration * fps)
    frame_indices = np.linspace(0, num_frames-1, frames_to_extract).astype(int)
    
    # Read frames
    frames = [vr[i].asnumpy()[None] for i in frame_indices]
    frames = np.concatenate(frames, axis=0)
    
    return frames, fps