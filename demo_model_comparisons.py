# import os
# import ffmpeg

# # Define file paths for both folders
# folder_online = "saved_videos_online"
# folder_offline = "saved_videos_offline"
# output_folder = "model_comparisons"
# os.makedirs(output_folder, exist_ok=True)

# # Create lists of input files
# online_files = os.listdir(folder_online)
# offline_files = os.listdir(folder_offline)

# # Process each pair of files individually
# for online, offline in zip(online_files, offline_files):
#     # Full paths to the video files
#     online_path = os.path.join(folder_online, online)
#     offline_path = os.path.join(folder_offline, offline)
    
#     # Output path for each combined video
#     output_video = os.path.join(output_folder, f"combined_{online}")

#     # Load the videos
#     video_online = ffmpeg.input(online_path)
#     video_offline = ffmpeg.input(offline_path)
    
#     # Combine them side by side without titles
#     side_by_side = ffmpeg.filter(
#         [video_online, video_offline],
#         'hstack'
#     )
    
#     # Save the output for each pair
#     ffmpeg.output(side_by_side, output_video).run()

# print("Processing complete. All videos saved in 'model_comparisons' folder.")

import os
from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip

# Define file paths for both folders
folder_online = "saved_videos_online_fps"
folder_offline = "saved_videos_offline_fps"
output_folder = "model_comparisons_fps"
os.makedirs(output_folder, exist_ok=True)

# Create lists of input files
online_files = os.listdir(folder_online)
offline_files = os.listdir(folder_offline)

# Process each pair of files individually
for online, offline in zip(online_files, offline_files):
    # Full paths to the video files
    online_path = os.path.join(folder_online, online)
    offline_path = os.path.join(folder_offline, offline)
    
    # Load the videos
    video_online = VideoFileClip(online_path)
    video_offline = VideoFileClip(offline_path)
    
    # Create titles as text clips
    title_online = TextClip("Online Model", fontsize=24, color='black').set_position('center').set_duration(video_online.duration)
    title_offline = TextClip("Offline Model", fontsize=24, color='black').set_position('center').set_duration(video_offline.duration)
    
    # Combine titles with videos
    video_online_with_title = CompositeVideoClip([video_online, title_online.set_position(("center", "top"))])
    video_offline_with_title = CompositeVideoClip([video_offline, title_offline.set_position(("center", "top"))])
    
    # Stack videos side by side
    side_by_side = clips_array([[video_online_with_title, video_offline_with_title]])
    
    # Output path for each combined video
    output_video = os.path.join(output_folder, f"combined_{online}")
    
    # Write the final video
    side_by_side.write_videofile(output_video, codec="libx264")

print("Processing complete. All videos saved in 'model_comparisons' folder.")
