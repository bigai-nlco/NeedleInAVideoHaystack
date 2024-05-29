import cv2
import numpy as np

def put_text(frame, text, position, font_scale, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def insert_video(main_video, insert_video, insert_point, position):
    # Read the frame from the insert video
    success, insert_frame = insert_video.read()
    if not success:
        return main_video

    # Resize the insert frame to fit the position if necessary
    insert_frame = cv2.resize(insert_frame, (position[2], position[3]))

    # Insert the frame into the main video
    main_video[position[1]:position[1]+position[3], position[0]:position[0]+position[2]] = insert_frame
    return main_video

def process_video(input_video_filename, insert_needle, modality, insert_point, output_video_filename):
    # Open the main video
    main_video = cv2.VideoCapture(input_video_filename)
    fps = main_video.get(cv2.CAP_PROP_FPS)
    width = int(main_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(main_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(main_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_filename, fourcc, fps, (width, height))

    # Prepare the insert video or text
    if modality == 'video':
        insert_video_clip = cv2.VideoCapture(insert_needle)
        insert_frame_count = int(insert_video_clip.get(cv2.CAP_PROP_FRAME_COUNT))
        insert_video_clip.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame

    # Process the video
    frame_id = 0
    while True:
        ret, frame = main_video.read()
        if not ret:
            break

        if modality == 'text' and insert_point <= frame_id / fps < insert_point + 1:
            put_text(frame, insert_needle, (50, height - 50), 1, (255, 255, 255))

        elif modality == 'video' and insert_point <= frame_id / fps < insert_point + insert_frame_count / fps:
            frame = insert_video(frame, insert_video_clip, insert_point, (50, 50, 200, 200))  # Example position and size

        out.write(frame)
        frame_id += 1

    # Release everything
    main_video.release()
    out.release()
    if modality == 'video':
        insert_video_clip.release()

    return output_video_filename

# Example usage:
output_video_path = process_video("test.mp4", "The secret word is 'needle'", "text", 5, "output.mp4")
# output_video_path = process_video("test.mp4", "insert.mp4", "video", 5, "output.mp4")

print(output_video_path)
