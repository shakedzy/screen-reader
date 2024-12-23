# Screen Reader

This repo contains a demo of reading a webpage text directly from a screen-capture video, using Florence-2-large and some logic 
(and a little help from Llama-3.2 Vision to extract the article title). The code is designed to handle the fact that there's always
only a part of the article that is shown, and it requires no additional metadata.

## Installation
1. Install [Ollama](https://ollama.com/) and pull the required models:
```bash
ollama pull llama-3.2-vision
```
2. cline this repo, and from the its root directory run:
```bash
pip install -r requirements.txt
```
3. Create a directory named `cache` under the root directory of the repo.

## Running
From the root directory of the repo, run:
```bash
python reader.py VIDEO_FILENAME
```
where `VIDEO_FILENAME` is the path to the screen-recording. 
The output article will be saved as Markdown text under `VIDEO_FILENAME.md`

## Examples
See the `examples` directory for an example of input and output.

## How does it work?
The code uses Florence-2-large `<OCR_WITH_REGION>` prompt to get both texts on the screen and their locations.
I then applies some common-sense logic regarding the location and orientation of the article in the web-page
to separate it from the rest of the text. 
Most logic is documented under the `stitch_lines` function, see its comments for more info.