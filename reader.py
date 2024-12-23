import os
import cv2
import json
import torch
import ollama
import random
import logging
import numpy as np
from tqdm import tqdm
from typing import Any
from datetime import datetime
from collections import Counter
from PIL import Image, ImageDraw
from numpy.typing import NDArray
from argparse import ArgumentParser
from PIL.Image import Image as ImageType
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoProcessor, AutoModelForCausalLM 


CONFIG = {
    # Minimum number of pixels required to change between frames
    'pixel_threshold': 10000,
    # Minimal time-lapse between different frames (in seconds)
    'fps': 0.5,
    # For fuzzy-matching when stitching two frames
    'levenshtein_threshold': 20,
    # Parallel computation of Florence
    'max_parallel_threads': 3
}

NOW = datetime.now().strftime('%Y%m%d%H%M%S')
FLORENCE_TASK = '<OCR_WITH_REGION>'


logger = logging.getLogger('ScreenReader')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.propagate = False


def torch_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"  
    elif torch.cuda.is_available():
        device = "cuda"
    else: 
        device = "cpu"
    return device


device = torch_device()
torch_dtype = torch.float16 if device != 'cpu' else torch.float32
logger.info(f'Running on device: {device}')
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)


def get_blur_radius(width: int, height: int) -> tuple[int, int]:
    width_radius = width // 100  
    height_radius = height // 100 
    # Ensure both radii are odd
    if width_radius % 2 == 0: width_radius += 1
    if height_radius % 2 == 0: height_radius += 1
    return width_radius, height_radius


def most_frequent(lst: list, *, k: int = 1, threshold: float = 0) -> list[int]:
    # Sort the list to handle threshold grouping
    lst_sorted = sorted(lst)
    grouped_counts = Counter()

    # Group numbers within the threshold and sum their counts
    current_group_leader = lst_sorted[0]
    for num in lst_sorted:
        if abs(num - current_group_leader) <= threshold:
            grouped_counts[current_group_leader] += 1
        else:
            current_group_leader = num
            grouped_counts[current_group_leader] += 1

    # Get the top X elements by count, sorted in descending order
    top_elements = grouped_counts.most_common(k)
    return [t[0] for t in top_elements]


def levenshtein_distance(str1: str, str2: str) -> int:
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1

    # Create a 2D array to store the distances
    dp = [[0 for _ in range(len_str2)] for _ in range(len_str1)]

    # Initialize the first column and row
    for i in range(len_str1):
        dp[i][0] = i
    for j in range(len_str2):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(dp[i - 1][j] + 1,         # Deletion
                           dp[i][j - 1] + 1,         # Insertion
                           dp[i - 1][j - 1] + cost)  # Substitution

    return dp[-1][-1]


def draw_ocr_bboxes(image: ImageType, prediction: dict[str, Any], output_path: str) -> None:
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = np.array(box).tolist()
        draw.polygon(new_box, width=3, outline=color)   # pyright: ignore[reportArgumentType]
        draw.text((new_box[0]+8, new_box[1]+2),         # pyright: ignore[reportOperatorIssue,reportIndexIssue]
                    "{}".format(label),
                    align="right",
                    fill=color)
    image.save(output_path)
        

def analyze_single_frame(image_array: NDArray, index: int) -> dict[str, Any]:
    image = Image.fromarray(image_array)
    inputs = processor(text=FLORENCE_TASK, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=4096,
        num_beams=3,
        do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=FLORENCE_TASK, image_size=(image.width, image.height))
    prediction: dict[str, Any] = parsed_answer[FLORENCE_TASK]
    draw_ocr_bboxes(image, prediction, os.path.join(os.getcwd(), 'cache/{now}_OCR_{i}.png'.format(now=NOW, i=index)))
    with open('cache/{now}_OCR_{i}.json'.format(now=NOW, i=index), 'w') as f:
        json.dump(prediction, f)
    return prediction


def find_title(image_path: str) -> str:
    logger.info(f'Finding title from image: {image_path}')
    prompt = \
"""
What is the title of the article which appears in this screenshot?
Return only the title, formatted as a JSON in the following way:
```json
{
    "title": "..."
}
```
""".strip()
    response = ollama.chat(
        model='llama3.2-vision',
        messages=[{'role': 'user', 'content': prompt, 'images': [image_path]}],
        format={
                'properties': {'title': {'title': 'Title', 'type': 'string'}},
                'required': ['title'],
                'title': 'Response',
                'type': 'object'
            }
    )['message']['content']
    return json.loads(response)['title']


def stitch_lines(predictions: list[dict[str, Any]]) -> str:
    all_bboxes: list[list[float]] = [] 
    for p in predictions:
        all_bboxes.extend(p['quad_boxes'])  

    # Here we look for the most common left-x coordinates of the texts found by the model.
    # The underlying assumption is that most of the text in the video is of the article itself,
    # and it's all aligned in the same way. This means we can assume that all (and only) text boxes
    # which are aligned in this way are part of the article text
    x_threshold = 5
    x = [int(b[0]) for b in all_bboxes] + [int(b[6]) for b in all_bboxes]   # [x1, y1, x2, y2, x3, y3, x4, y4]
    article_text_x = most_frequent(x, threshold=x_threshold)[0]
    
    # Next, we'll find the spacing between lines of the article text.
    # The assumption here is that the most common spacing will be between lines of the same paragraph,
    # and the second-most common will be the spacing between different paragraphs.
    spaces = []
    for p in predictions:
        bboxes = p['quad_boxes']
        last_bbox: None | int = None
        for i, bbox in enumerate(bboxes):
            if i == 0: continue
            if article_text_x - x_threshold <= int(bbox[0]) <= article_text_x + x_threshold:
                if last_bbox is not None:
                    spaces.append(int(bbox[1]) - int(bboxes[last_bbox][-1]))
                last_bbox = i
    
    y_threshold = 8
    most_frequent_spaces = most_frequent(spaces, k=2, threshold=y_threshold)
    in_paragraph_space, paragraph_space = tuple(most_frequent_spaces)
    print(in_paragraph_space, paragraph_space)
    print(spaces)

    # Here we extract all lines of the article itself, based on the metadata computed so far
    # A paragraph-separation is added where needed, based on the line spacing.
    # We also keep track of the index of the first frame with article text, as it will be later
    # used to try and extract the article title from.
    first_frame_with_text = -1
    all_lines: list[list[str]] = []
    for frame_index, p in enumerate(predictions):
        current_lines: list[str] = []
        labels: list[str] = p['labels']
        bboxes: list[list[float]] = p['quad_boxes']
        last_bbox = None
        for i, (label, bbox) in enumerate(zip(labels, bboxes)):
            if article_text_x - x_threshold <= int(bbox[0]) <= article_text_x + x_threshold:
                if last_bbox is not None and paragraph_space - y_threshold <= int(bbox[1]) - int(bboxes[last_bbox][-1]) <= paragraph_space + y_threshold: 
                    current_lines.append('\n\n')
                current_lines.append(label.replace('</s>','').strip())  # </s> is an artifact of Florence.
                last_bbox = i
        if current_lines:
            all_lines.append(current_lines)
            if first_frame_with_text == -1:
                first_frame_with_text = frame_index

    # Stitching lines
    text = '\n'.join(all_lines[0])
    last_line = [line for line in all_lines[0] if line.strip()][-1]
    for i, frame_lines in enumerate(all_lines):
        if i==0: continue
        try:
            matching_index = frame_lines.index(last_line)
        except ValueError:
            below_threshold = [levenshtein_distance(last_line, line) <= CONFIG['levenshtein_threshold'] for line in frame_lines]
            if any(below_threshold):
                for i, b in enumerate(below_threshold):
                    if b:
                        matching_index = i
                        break
            else:
                text += ' _<POSSIBLE MISSING/DUPLICATE TEXT>_'
                matching_index = -1
        if matching_index + 1 < len(frame_lines):
            text += (' ' + ' '.join(frame_lines[matching_index+1:]))
            last_line = [line for line in frame_lines if line.strip()][-1]

    title = find_title(os.path.join(os.getcwd(), 'cache/{now}_{i}.png'.format(now=NOW, i=first_frame_with_text)))
    if title:
        text = f'# {title}\n\n{text}'
    text = text.replace('\n ', '\n')
    return text


def frames_from_video(video_path: str) -> list[NDArray]:
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Resolution: {width}x{height}, duration: {fps * frame_count} seconds (FPS: {fps})")

    frame_interval = int(fps * CONFIG['fps'])  
    change_threshold = CONFIG['pixel_threshold']
    width_blur_radius, height_blur_radius = get_blur_radius(width, height)
    
    output_frames_paths: list[NDArray] = []
    frame_count = 0
    previous_frame = None
    output_path_template = os.path.join(os.getcwd(), 'cache/{now}_{i}.png')
    while True:
        ret, frame = video.read()  # Read the frame
        if not ret:
            break  # Break if no more frames are available        

        if frame_count % frame_interval == 0:
            current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert current frame to grayscale

            if previous_frame is not None:
                # Calculate absolute difference between current and previous frame
                diff = cv2.absdiff(previous_frame, current_frame_gray)

                # Apply Gaussian blur with the corrected radius
                diff_blurred = cv2.GaussianBlur(diff, (width_blur_radius, height_blur_radius), 0)

                # Count the number of pixels with significant change after the blur
                diff_thresholded = cv2.threshold(diff_blurred, 25, 255, cv2.THRESH_BINARY)[1]
                non_zero_count = np.count_nonzero(diff_thresholded)

                # Save the frame only if the change is significant
                if non_zero_count > change_threshold:
                    name = output_path_template.format(now=NOW, i=len(output_frames_paths))
                    cv2.imwrite(name, frame)
                    output_frames_paths.append(frame)
                    previous_frame = current_frame_gray

            else:
                # Save the first frame unconditionally
                previous_frame = current_frame_gray
                name = output_path_template.format(now=NOW, i=len(output_frames_paths))
                cv2.imwrite(name, frame)
                output_frames_paths.append(frame)

        frame_count += 1

    video.release()
    logger.info(f'Extracted {len(output_frames_paths)} frames')
    return output_frames_paths


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('video_filename', nargs=1, type=str, help='Path to video')
    args = parser.parse_args()
    video_filename: str = args.video_filename[0] if isinstance(args.video_filename, list) else args.video_filename
    logger.info(f'Loading: {video_filename}')

    frames = frames_from_video(video_filename)
    analyze_and_return_with_index = lambda t: (t[0], analyze_single_frame(t[1], t[0]))
    with ThreadPoolExecutor(max_workers=CONFIG['max_parallel_threads']) as executor:
        results: list[tuple[int, dict[str, Any]]] = list(tqdm(executor.map(analyze_and_return_with_index, enumerate(frames)), 
                                                              desc='Extracting text from images', total=len(frames))) 
    predictions = [t[1] for t in sorted(results, key=lambda t: t[0])]
    article = stitch_lines(predictions)

    output_filename = f'{video_filename}.md'
    with open(output_filename, 'w') as f:
        f.write(article)
    logger.info(f'Saved to: {output_filename}')
