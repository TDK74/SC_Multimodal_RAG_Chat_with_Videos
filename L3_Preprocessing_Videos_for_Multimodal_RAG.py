import base64
import json
import os
import cv2
import webvtt
import whisper

from os import path as osp
from pathlib import Path
from urllib.request import urlretrieve
from moviepy.editor import VideoFileClip
from PIL import Image
from utils import (download_video, encode_image, get_transcript_vtt, getSubs, lvlm_inference,
                    maintain_aspect_ratio_resize, str2time)


## ------------------------------------------------------ ##
vid1_url = "https://www.youtube.com/watch?v=7Hcg-rLYwdM"

vid1_dir = "./shared_data/videos/video1"
vid1_filepath = download_video(vid1_url, vid1_dir)

vid1_transcript_filepath = get_transcript_vtt(vid1_url, vid1_dir)

## ------------------------------------------------------ ##
print(vid1_filepath)
print(vid1_transcript_filepath)

## ------------------------------------------------------ ##
vid2_url=(
    "https://multimedia-commons.s3-us-west-2.amazonaws.com/"
    "data/videos/mp4/010/a07/010a074acb1975c4d6d6e43c1faeb8.mp4"
)
vid2_dir = "./shared_data/videos/video2"
vid2_name = "toddler_in_playground.mp4"

Path(vid2_dir).mkdir(parents=True, exist_ok=True)
vid2_filepath = urlretrieve(
                        vid2_url,
                        osp.join(vid2_dir, vid2_name)
                    )[0]

## ------------------------------------------------------ ##
def extract_and_save_frames_and_metadata(
        path_to_video,
        path_to_transcript,
        path_to_save_extracted_frames,
        path_to_save_metadatas):
    metadatas = []

    video = cv2.VideoCapture(path_to_video)

    trans = webvtt.read(path_to_transcript)

    for idx, transcript in enumerate(trans):
        start_time_ms = str2time(transcript.start)
        end_time_ms = str2time(transcript.end)

        mid_time_ms = (end_time_ms + start_time_ms) / 2

        text = transcript.text.replace("\n", ' ')

        video.set(cv2.CAP_PROP_POS_MSEC, mid_time_ms)
        success, frame = video.read()
        if success:
            image = maintain_aspect_ratio_resize(frame, height=350)

            img_fname = f'frame_{idx}.jpg'
            img_fpath = osp.join(
                path_to_save_extracted_frames, img_fname
            )
            cv2.imwrite(img_fpath, image)

            metadata = {
                'extracted_frame_path': img_fpath,
                'transcript': text,
                'video_segment_id': idx,
                'video_path': path_to_video,
                'mid_time_ms': mid_time_ms,
            }
            metadatas.append(metadata)

        else:
            print(f"ERROR! Cannot extract frame: idx = {idx}")

    fn = osp.join(path_to_save_metadatas, 'metadatas.json')
    with open(fn, 'w') as outfile:
        json.dump(metadatas, outfile)
    return metadatas

## ------------------------------------------------------ ##
extracted_frames_path = osp.join(vid1_dir, 'extracted_frame')
metadatas_path = vid1_dir

Path(extracted_frames_path).mkdir(parents=True, exist_ok=True)
Path(metadatas_path).mkdir(parents=True, exist_ok=True)

metadatas = extract_and_save_frames_and_metadata(
                vid1_filepath,
                vid1_transcript_filepath,
                extracted_frames_path,
                metadatas_path,
            )

## ------------------------------------------------------ ##
print(metadatas[:4])

## ------------------------------------------------------ ##
path_to_video_no_transcript = vid1_filepath

path_to_extracted_audio_file = os.path.join(vid1_dir, 'audio.mp3')

clip = VideoFileClip(path_to_video_no_transcript)
clip.audio.write_audiofile(path_to_extracted_audio_file)

## ------------------------------------------------------ ##
model = whisper.load_model("small")
options = dict(task="translate", best_of=1, language='en')
results = model.transcribe(path_to_extracted_audio_file, **options)

## ------------------------------------------------------ ##
vtt = getSubs(results["segments"], "vtt")

path_to_generated_trans = osp.join(vid1_dir, 'generated_video1.vtt')

with open(path_to_generated_trans, 'w') as f:
    f.write(vtt)

## ------------------------------------------------------ ##
lvlm_prompt = "Can you describe the image?"

## ------------------------------------------------------ ##
path_to_frame = osp.join(vid1_dir, "extracted_frame", "frame_5.jpg")
frame = Image.open(path_to_frame)
display(frame)

## ------------------------------------------------------ ##
image = encode_image(path_to_frame)
caption = lvlm_inference(lvlm_prompt, image)
print(caption)

## ------------------------------------------------------ ##
def extract_and_save_frames_and_metadata_with_fps(
        path_to_video,
        path_to_save_extracted_frames,
        path_to_save_metadatas,
        num_of_extracted_frames_per_second=1):
    metadatas = []

    video = cv2.VideoCapture(path_to_video)

    fps = video.get(cv2.CAP_PROP_FPS)

    hop = round(fps / num_of_extracted_frames_per_second)
    curr_frame = 0
    idx = -1
    while(True):
        ret, frame = video.read()
        if not ret:
            break
        if curr_frame % hop == 0:
            idx = idx + 1

            image = maintain_aspect_ratio_resize(frame, height=350)

            img_fname = f'frame_{idx}.jpg'
            img_fpath = osp.join(
                            path_to_save_extracted_frames,
                            img_fname
                        )
            cv2.imwrite(img_fpath, image)

            b64_image = encode_image(img_fpath)
            caption = lvlm_inference(lvlm_prompt, b64_image)

            metadata = {
                'extracted_frame_path': img_fpath,
                'transcript': caption,
                'video_segment_id': idx,
                'video_path': path_to_video,
            }
            metadatas.append(metadata)
        curr_frame += 1

    metadatas_path = osp.join(path_to_save_metadatas,'metadatas.json')
    with open(metadatas_path, 'w') as outfile:
        json.dump(metadatas, outfile)
    return metadatas

## ------------------------------------------------------ ##
extracted_frames_path = osp.join(vid2_dir, 'extracted_frame')
metadatas_path = vid2_dir

Path(extracted_frames_path).mkdir(parents=True, exist_ok=True)
Path(metadatas_path).mkdir(parents=True, exist_ok=True)

metadatas = extract_and_save_frames_and_metadata_with_fps(
                vid2_filepath,
                extracted_frames_path,
                metadatas_path,
                num_of_extracted_frames_per_second=0.1
            )

## ------------------------------------------------------ ##
data = metadatas[1]
caption = data['transcript']
print(f'Generated caption is: "{caption}"')
frame = Image.open(data['extracted_frame_path'])
display(frame)
