## StabledGroundingSAM

StabledGroundingSAM is a package that leverages stable diffusion, Grounding Dino, and Segment Anything to generate new images with object segmentation. It takes an input image and a text file containing the names of objects of interest. The package then applies stable diffusion to the input image, followed by object detection using Grounding Dino to draw bounding boxes around the detected objects. Finally, Segment Anything generates segmentation masks for the objects in the images. The generated dataset is saved in the YOLO format.

### 💻 Installation

To install StabledGroundingSAM, make sure you have Python 3.7 or later installed. Then, use pip to install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

### 🌀 Results

Here are the results obtained using StabledGroundingSAM with the input prompt "apple":

**Input image:**

<img src=".asset/input_apple.png" alt="input_apple" width="300"/>

**Stable Diffusion image:**

<img src=".asset/stabled_apple.png" alt="stabled_apple" width="300"/>

**Segmented image:**

<img src=".asset/segmented_apple.png" alt="segmented_apple" width="300"/>

### 🔥 Quickstart

To get started with StabledGroundingSAM, follow these steps:

1. Clone the StabledGroundingSAM repository from GitHub:

```bash
git clone https://github.com/Marco2929/StabledGroundingSAM.git
```

2. Change the current directory to the StabledGroundingSAM folder:

```bash
cd StabledGroundingSAM/
```

3. Clone the GroundingDINO repository from GitHub and follow the instructions provided there:

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
```

4. Download the weights for Segment Anything and GroundingDINO (if not already done) and move them to the "weights" folder:

```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

5. Create a "classes.txt" file that contains the objects you want to label.

6. Run the program by providing the location of the "classes.txt" file, the initial image location, and the number of pictures you want the model to generate:

```bash
python -m main <class.txt location> <initial image location> <number of pictures>
```

### 🔧 Adjustments

StabledGroundingSAM provides several optional arguments that allow you to adjust the output of the models:

#### Stable Diffusion:

- `--diffusion_prompt`: Provide a different prompt to stable diffusion (by default, uses the "classes.txt" file).

- `--guidance_scale`: Adjust the guidance scale.

- `--strength`: Adjust the strength.

Refer to the stable diffusion documentation for more information.

#### Grounding-Dino:

- `--box_threshold`: Adjust the box threshold for object detection in Grounding Dino.

- `--text_threshold`: Adjust the text threshold for object detection in Grounding Dino.

Refer to the Grounding Dino documentation for more information.

#### Segment Anything (only available in the YOLO dataset):

- `--min_image_area_percentage`: Specify the minimum image area percentage for the segmentation mask.

- `--max_image_area_percentage`: Specify the maximum image area percentage for the segmentation mask.

- `--approximation_percentage`: Adjust the sharpness of the segmentation mask.

These adjustable parameters allow you to fine-tune the output according to your requirements.

Feel free to experiment with these parameters to achieve the desired results.

## Conclusion

StabledGroundingSAM is a powerful package that combines stable diffusion, Grounding Dino, and Segment Anything to generate new images with object segmentation. By following the installation steps and using the provided adjustable parameters, you can customize the output according to your needs. Have fun exploring the capabilities of StabledGroundingSAM and creating your own segmented image datasets!