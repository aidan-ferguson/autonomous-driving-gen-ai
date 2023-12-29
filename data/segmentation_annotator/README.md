This script hosts a Gradio web server which enables semantic segmentation labelling of images from real world datasets

## Setup

To run this first clone [EdgeSAM](https://github.com/chongzhou96/EdgeSAM) and follow the setup instructions, then, run the following command:

```console
python3 gradio_app.py --checkpoint <path to pre-trained weights> --input_image_folder <path to folder containing images> --output_folder <path to the folder which should hold the output annotations>
```

## Usage

TODO