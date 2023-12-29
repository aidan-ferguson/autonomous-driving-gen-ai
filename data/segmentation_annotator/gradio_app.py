# Code credit: [FastSAM Demo](https://huggingface.co/spaces/An-619/FastSAM).
# Modified by Aidan Ferguson for University of Glasgow Level 4 Dissertation

import torch
import gradio as gr
import numpy as np
from edge_sam import sam_model_registry, SamPredictor
from edge_sam.onnx import SamPredictorONNX
from PIL import ImageDraw
from utils.tools_gradio import fast_process
import copy
import argparse
import os

# Class ids for annotation - the id corresponds to the position in the array
classes = [
    "sky",
    "ground",
    "car_body",
    "blue_cone",
    "yellow_cone",
    "orange_cone",
    "large_orange_cone",
    "barrier",
    "tire_wall",
    "tree",
    "building",
    "person"
]

parser = argparse.ArgumentParser(
    description="Host EdgeSAM as a local web service."
)
parser.add_argument(
    "--checkpoint",
    default="weights/edge_sam_3x.pth",
    type=str,
    help="The path to the PyTorch checkpoint of EdgeSAM."
)
parser.add_argument(
    "--encoder-onnx-path",
    default="weights/edge_sam_3x_encoder.onnx",
    type=str,
    help="The path to the ONNX model of EdgeSAM's encoder."
)
parser.add_argument(
    "--decoder-onnx-path",
    default="weights/edge_sam_3x_decoder.onnx",
    type=str,
    help="The path to the ONNX model of EdgeSAM's decoder."
)
parser.add_argument(
    "--enable-onnx",
    action="store_true",
    help="Use ONNX to speed up the inference.",
)
parser.add_argument(
    "--server-name",
    default="0.0.0.0",
    type=str,
    help="The server address that this demo will be hosted on."
)
parser.add_argument(
    "--port",
    default=8080,
    type=int,
    help="The port that this demo will be hosted on."
)
parser.add_argument(
    '--input_image_folder',
    required=True,
    type=str,
    help="A folder containing all the images to be annotated"
)
parser.add_argument(
    '--output_folder',
    required=True,
    type=str,
    help="A folder in which the annotated data will be written to"
)
args = parser.parse_args()

if not os.path.exists(args.input_image_folder) or not os.path.isdir(args.input_image_folder):
    raise Exception("You need to provide a valid image folder")

if not os.path.exists(args.output_folder):
    raise Exception("The output folder must be a valid folder and exist")

examples = [os.path.join(args.input_image_folder, elem) for elem in os.listdir(args.input_image_folder) if os.path.isfile(os.path.join(args.input_image_folder, elem))]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.enable_onnx:
    # device = "cpu"
    predictor = SamPredictorONNX(args.encoder_onnx_path, args.decoder_onnx_path)
else:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["edge_sam"](checkpoint=args.checkpoint, upsample_mode="bicubic")
    sam = sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

# Description
title = "<center><strong><font size='8'>EdgeSAM<font></strong> <a href='https://github.com/chongzhou96/EdgeSAM'><font size='6'>[GitHub]</font></a> </center>"

description_p = """ # Instructions for point mode

                1. Upload an image.
                2. Select the point type.
                3. Click once or multiple times on the image to indicate the object of interest.
                4. The Clear button clears all the points.
                5. The Reset button resets both points and the image.

              """


css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

# Some custom JS to get the filename of the currently selected image
# js = """<html>
#   <body>

#       <script type = "text/JavaScript">
#         function test() {
#           document.getElementById('demo').innerHTML = "Hello"
#           }
#       </script>

#     <h1>My First JavaScript</h1>
#     <button type="testButton" onclick="test()"> Start </button>

#   </body>
# </html>
# """

global_points = []
global_point_label = []
global_box = []
global_image = None
global_image_with_prompt = None
seg_mask = None
current_image_filename = None
current_semantic_class = None

def reset():
    global global_points
    global global_point_label
    global global_box
    global global_image
    global global_image_with_prompt
    global_points = []
    global_point_label = []
    global_box = []
    global_image = None
    global_image_with_prompt = None
    return None


def reset_all():
    global global_points
    global global_point_label
    global global_box
    global global_image
    global global_image_with_prompt
    global_points = []
    global_point_label = []
    global_box = []
    global_image = None
    global_image_with_prompt = None
    return None, None


def clear():
    global global_points
    global global_point_label
    global global_box
    global global_image
    global global_image_with_prompt
    global_points = []
    global_point_label = []
    global_box = []
    global_image_with_prompt = copy.deepcopy(global_image)
    return global_image


def on_image_upload(image, input_size=1024):
    global global_points
    global global_point_label
    global global_box
    global global_image
    global global_image_with_prompt
    global_points = []
    global_point_label = []
    global_box = []

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))
    global_image = copy.deepcopy(image)
    global_image_with_prompt = copy.deepcopy(image)
    nd_image = np.array(global_image)
    predictor.set_image(nd_image)

    return image

def class_dropdown_change(dropdown):
    global current_semantic_class
    current_semantic_class = classes[dropdown]
    print(classes[dropdown])

def save_segment(image):
    if current_semantic_class is None or current_image_filename is None:
        print("Semantic class or image filename not set by frontend")

    if not os.path.exists(f"{args.output_folder}/{current_image_filename}"):
        os.mkdir(f"{args.output_folder}/{current_image_filename}")

    filename_prefix = f"{args.output_folder}/{current_image_filename}/{current_semantic_class}_"
    filename_number = 0
    filename_suffix = ".png"
    while (os.path.exists(filename_prefix + str(filename_number) + filename_suffix)):
        filename_number += 1
    
    seg_mask.save((filename_prefix + str(filename_number) + filename_suffix))

    return clear()

def filename_input_change(fname):
    global current_image_filename
    print(fname)
    current_image_filename = fname

def segment_with_points(
        label,
        evt: gr.SelectData,
        input_size=1024,
        better_quality=False,
        withContours=True,
        use_retina=True,
        mask_random_color=False,
):
    global global_points
    global global_point_label
    global global_image_with_prompt
    global seg_mask

    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 5, (97, 217, 54) if label == "Positive" else (237, 34, 13)
    global_points.append([x, y])
    global_point_label.append(1 if label == "Positive" else 0)

    print(f'global_points: {global_points}')
    print(f'global_point_label: {global_point_label}')

    draw = ImageDraw.Draw(global_image_with_prompt)
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    image = global_image_with_prompt

    if args.enable_onnx:
        global_points_np = np.array(global_points)[None]
        global_point_label_np = np.array(global_point_label)[None]
        masks, scores, _ = predictor.predict(
            point_coords=global_points_np,
            point_labels=global_point_label_np,
        )
        masks = masks.squeeze(0)
        scores = scores.squeeze(0)
    else:
        global_points_np = np.array(global_points)
        global_point_label_np = np.array(global_point_label)
        masks, scores, logits = predictor.predict(
            point_coords=global_points_np,
            point_labels=global_point_label_np,
            num_multimask_outputs=4,
            use_stability_score=True
        )

    print(f'scores: {scores}')
    area = masks.sum(axis=(1, 2))
    print(f'area: {area}')

    annotations = np.expand_dims(masks[scores.argmax()], axis=0)

    seg, seg_mask = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )

    return seg


script = """
function test(){
    let script = document.createElement('script');
    script.innerHTML = `
        // Attach a callback to the 'examples' component
        const button = document.querySelector("#gallery_component");
        button.addEventListener("click", e => {
        
            // When an image is highlighted, it will have the class 'selected' (in the Gradio version I am using)
            //   we use this to determine the filename of the image and set it in the filename entry automatically
            const collection = document.getElementsByClassName("selected");

            for(const element of collection)
            {
                if (element.classList.contains("gallery"))
                {
                    // Recover image filename from src URI
                    const src = element.children[0].getAttribute("src");
                    const uri_segments = src.split("/");
                    const filename = uri_segments[uri_segments.length-1];
                    
                    // Set the filename in the text box
                    const filename_entry = document.getElementById("filename_entry").children[0].children[1];
                    filename_entry.value = filename;

                    // Dispatch an event so the Gradio backend gets the value
                    const event = new Event('input');
                    filename_entry.dispatchEvent(event);
                }
            }
        });
    `;
    document.head.appendChild(script);
}       
    
"""

img_p = gr.Image(type="pil", height=720, width=1280)
with gr.Blocks(css=css, js=script, title="EdgeSAM") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)

    with gr.Tab("Point mode") as tab_p:
        filename_textbox = gr.Textbox(placeholder="Image Filename", elem_id="filename_entry")
        filename_textbox.change(filename_input_change, [filename_textbox])

        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                img_p.render()
        with gr.Row(variant="panel"):
            with gr.Row():
                add_or_remove = gr.Radio(
                    ["Positive", "Negative"],
                    value="Positive",
                    label="Point Type"
                )

                with gr.Column():
                    class_dropdown = gr.Dropdown(classes, type="index", label="Class")
                    save_segment_p = gr.Button("Save Segment", variant="secondary")
                    clear_btn_p = gr.Button("Clear", variant="secondary")
                    reset_btn_p = gr.Button("Reset", variant="secondary")
            with gr.Row():
                gr.Markdown(description_p)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("Please annotate images from the dataset below")
                examples_elem = gr.Examples(
                    examples=examples,
                    inputs=[img_p],
                    outputs=[img_p],
                    examples_per_page=8,
                    fn=on_image_upload,
                    run_on_click=True,
                    elem_id="gallery_component"
                )

    class_dropdown.input(class_dropdown_change, [class_dropdown])

    img_p.upload(on_image_upload, img_p, [img_p])
    img_p.select(segment_with_points, [add_or_remove], img_p)

    save_segment_p.click(save_segment, outputs=[img_p])
    clear_btn_p.click(clear, outputs=[img_p])
    reset_btn_p.click(reset, outputs=[img_p])
    tab_p.select(fn=reset_all, outputs=[img_p])

demo.queue()
demo.launch(server_name=args.server_name, server_port=args.port)