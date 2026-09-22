from pathlib import Path
from urllib.request import urlretrieve
from IPython.display import display
from PIL import Image
from utils import (encode_image, lvlm_inference_with_conversation, prediction_guard_llava_conv)


## ------------------------------------------------------ ##
url1 = ('https://farm4.staticflickr.com/3300/3497460990_11dfb95dd1_z.jpg')

img1_metadata = {"link" : url1,
                "transcript" : ("Wow, this trick is amazing!"),
                "path_to_file" : "./shared_data/skateboard.jpg"}

img2_metadata = {"transcript" : ("As I look back on the the mission that we've had here on "
                                "the International Space Station, I'm proud to have been a part of "
                                "much of the science activities that happened over the last "
                                "two months."),
                "path_to_file" : "./shared_data/videos/video1/extracted_frame/frame_1.jpg"}

## ------------------------------------------------------ ##
img3_metadata = {"transcript" : ("the science activities that happened over the last two months. "
                                "The view is always amazing I didn't think I would do another "
                                "spacewalk and to now have the chance to have done four more was "
                                "just icing on the cake for a"),
                "path_to_file" : "./shared_data/videos/video1/extracted_frame/frame_5.jpg"}

if not Path(img1_metadata['path_to_file']).exists():
    _ = urlretrieve(img1_metadata['link'], img1_metadata['path_to_file'])

## ------------------------------------------------------ ##
prompt = "Please describe the image in detail"
image_path = img2_metadata['path_to_file']
b64_img = encode_image(image_path)

img_captioning_conv = prediction_guard_llava_conv.copy()
img_captioning_conv.append_message('user', [prompt, b64_img])

caption = lvlm_inference_with_conversation(img_captioning_conv)

## ------------------------------------------------------ ##
display(Image.open(image_path))
print(caption)

## ------------------------------------------------------ ##
prompt = "What is likely going to happen next?"
image_path = img1_metadata['path_to_file']
b64_img = encode_image(image_path)

qna_visual_cues_conv = prediction_guard_llava_conv.copy()
qna_visual_cues_conv.append_message('user', [prompt, b64_img])

answer = lvlm_inference_with_conversation(qna_visual_cues_conv)

## ------------------------------------------------------ ##
display(Image.open(image_path))
print(answer)

## ------------------------------------------------------ ##
prompt = 'What is the name of one of the astronauts?'
image_path = img2_metadata['path_to_file']
b64_img = encode_image(image_path)

qna_textual_cues_conv = prediction_guard_llava_conv.copy()
qna_textual_cues_conv.append_message('user', [prompt, b64_img])

answer = lvlm_inference_with_conversation(qna_textual_cues_conv)

display(Image.open(image_path))
print(answer)

## ------------------------------------------------------ ##
prompt_template = ("The transcript associated with the image is '{transcript}'."
                    "What do the astronauts feel about their work?")

prompt = prompt_template.format(transcript= img2_metadata["transcript"])
image_path = img2_metadata['path_to_file']
b64_img = encode_image(image_path)

qna_transcript_conv = prediction_guard_llava_conv.copy()
qna_transcript_conv.append_message('user', [prompt, b64_img])

answer = lvlm_inference_with_conversation(qna_transcript_conv, temperature= 0.95, top_k= 2)

## ------------------------------------------------------ ##
display(Image.open(image_path))
print(f"Prompt: {prompt}")
print("Answer: ")
print(answer)

## ------------------------------------------------------ ##
qna_transcript_conv.append_message('assistant', [answer])

follow_up_query = "Where did the astronauts return from?"
qna_transcript_conv.append_message('user', [follow_up_query])

follow_up_ans = lvlm_inference_with_conversation(qna_transcript_conv)

## ------------------------------------------------------ ##
print("Answer to the follow-up query: ")
print(follow_up_ans)
