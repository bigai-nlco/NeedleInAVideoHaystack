import os
import logging
import tempfile
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

import time
start = time.time()

# Suppress MoviePy logs
logging.getLogger('moviepy').setLevel(logging.CRITICAL)

# Function to estimate the font size based on the text length
def adjust_font_size(text, max_width, font='Arial', min_font_size=20, max_font_size=70):
    # Estimate the font size without creating a TextClip each time
    # This is a naive approach and may need adjustment based on the actual font metrics
    font_size = max_font_size
    text_clip = TextClip(text, fontsize=font_size, font=font)
    text_width = text_clip.size[0]
    text_clip.close()
    font_size = int(font_size * max_width / text_width)
    return max(min(font_size, max_font_size), min_font_size)

def process_video(input_video_filename, insert_needle, modality, insert_point, output_video_filename=None, tmp_dir='.tmp/'):
    # Load the main video
    main_video_clip = VideoFileClip(input_video_filename)
    fps = main_video_clip.fps

    if modality == 'text':
        # Calculate the maximum width for the text based on the video size
        max_text_width = main_video_clip.size[0] * 0.9  # 90% of the video width

        # Adjust the font size based on the text length
        font_size = adjust_font_size(insert_needle, max_text_width)

        # Create a text clip with the adjusted font size
        text_clip = TextClip(insert_needle, fontsize=font_size, color='white', font="Arial").set_position("bottom").set_duration(1).set_start(insert_point)

        # Create a composite clip with the text overlaid on the original video
        final_clip = CompositeVideoClip([main_video_clip, text_clip])
    elif modality == 'video':
        # Load the video to insert
        insert_video_clip = VideoFileClip(insert_needle).set_start(insert_point).set_position("center")

        # Create a composite clip with the insert video overlaid on the original video
        final_clip = CompositeVideoClip([main_video_clip, insert_video_clip])
    else:
        main_video_clip.close()
        raise ValueError("Invalid modality. Choose 'text' or 'video'.")

    # Use a NamedTemporaryFile to automatically delete the file after use
    os.makedirs(tmp_dir, exist_ok=True)
    output_video_filename = output_video_filename or os.path.join(tmp_dir, 'output.mp4')

    # Write the result to the file using FFMPEG_VideoWriter for better performance
    writer = FFMPEG_VideoWriter(output_video_filename, main_video_clip.size, fps, codec="libx264")
    for frame in final_clip.iter_frames(fps=fps, dtype="uint8"):
        writer.write_frame(frame)
    writer.close()

    # Close the clips to release their resources
    main_video_clip.close()
    if modality == 'text':
        text_clip.close()
    elif modality == 'video':
        insert_video_clip.close()
    final_clip.close()

    # Return the path to the output file
    return output_video_filename

# Example usage:
# To insert a video
output_video_path = process_video("test.mp4", "insert.mp4", "video", 5)

print(output_video_path)

print(time.time()-start)