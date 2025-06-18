
# 1. Import the required libraries and packages
import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont # could also use torch utilities for drawing

from transformers import AutoImageProcessor
from transformers import AutoModelForObjectDetection

### 2. Setup preprocessing and helper functions ###

# Setup target model path to load
# Note: Can load from Hugging Face or can load from local 
model_save_path = "mrdbourke/rt_detrv2_finetuned_trashify_box_detector_v1"

# Load the model and preprocessor
# Because this app.py file is running directly on Hugging Face Spaces, the model will be loaded from the Hugging Face Hub
image_processor = AutoImageProcessor.from_pretrained(model_save_path)
model = AutoModelForObjectDetection.from_pretrained(model_save_path)

# Set the target device (use CUDA/GPU if it is available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Get the id2label dictionary from the model
id2label = model.config.id2label

# Set up a colour dictionary for plotting boxes with different colours
color_dict = {   
    "bin": "green",
    "trash": "blue",
    "hand": "purple",
    "trash_arm": "yellow",
    "not_trash": "red",
    "not_bin": "red",
    "not_hand": "red",
}

# Create helper functions for seeing if items from one list are in another 
def any_in_list(list_a, list_b):
    "Returns True if *any* item from list_a is in list_b, otherwise False."
    return any(item in list_b for item in list_a)

def all_in_list(list_a, list_b):
    "Returns True if *all* items from list_a are in list_b, otherwise False."
    return all(item in list_b for item in list_a)

### 3. Create function to predict on a given image with a given confidence threshold ###
def predict_on_image(image, conf_threshold):
    # Make sure model is in eval mode
    model.eval()

    # Make a prediction on target image 
    with torch.no_grad():
        inputs = image_processor(images=[image], return_tensors="pt")
        model_outputs = model(**inputs.to(device))

        target_sizes = torch.tensor([[image.size[1], image.size[0]]]) # -> [batch_size, height, width] 
        
        # Post process the raw outputs from the model 
        results = image_processor.post_process_object_detection(model_outputs,
                                                                threshold=conf_threshold,
                                                                target_sizes=target_sizes)[0]

    # Return all items in results to CPU (we'll want this for displaying outputs with matplotlib)
    for key, value in results.items():
        try:
            results[key] = value.item().cpu() # can't get scalar as .item() so add try/except block
        except:
            results[key] = value.cpu()

    ### 4. Draw the predictions on the target image ###

    # Can return results as plotted on a PIL image (then display the image)
    draw = ImageDraw.Draw(image)

    # Get a font from ImageFont
    font = ImageFont.load_default(size=20)

    # Get class names as text for print out
    class_name_text_labels = []

    # Iterate through the predictions of the model and draw them on the target image
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        # Create coordinates
        x, y, x2, y2 = tuple(box.tolist())

        # Get label_name
        label_name = id2label[label.item()]
        targ_color = color_dict[label_name]
        class_name_text_labels.append(label_name)

        # Draw the rectangle
        draw.rectangle(xy=(x, y, x2, y2), 
                       outline=targ_color,
                       width=3)
        
        # Create a text string to display
        text_string_to_show = f"{label_name} ({round(score.item(), 3)})"

        # Draw the text on the image
        draw.text(xy=(x, y),
                  text=text_string_to_show,
                  fill="white",
                  font=font)
    
    # Remove the draw each time
    del draw

    # Setup blank string to print out
    return_string = ""

    # Setup list of target items to discover
    target_items = ["trash", "bin", "hand"]

    ### 5. Create logic for outputting information message ### 

    # If no items detected or trash, bin, hand not in list, return notification 
    if (len(class_name_text_labels) == 0) or not (any_in_list(list_a=target_items, list_b=class_name_text_labels)):
        return_string = f"No trash, bin or hand detected at confidence threshold {conf_threshold}. Try another image or lowering the confidence threshold."
        return image, return_string

    # If there are some missing, print the ones which are missing
    elif not all_in_list(list_a=target_items, list_b=class_name_text_labels):
        missing_items = []
        for item in target_items:
            if item not in class_name_text_labels:
                missing_items.append(item)
        return_string = f"Detected the following items: {class_name_text_labels}. But missing the following in order to get +1: {missing_items}. If this is an error, try another image or altering the confidence threshold. Otherwise, the model may need to be updated with better data."
        
    # If all 3 trash, bin, hand occur = + 1
    if all_in_list(list_a=target_items, list_b=class_name_text_labels):
        return_string = f"+1! Found the following items: {class_name_text_labels}, thank you for cleaning up the area!"

    print(return_string)
    
    return image, return_string

### 6. Setup the demo application to take in image, make a prediction with our model, return the image with drawn predicitons ### 

# Write description for our demo application
description = """
Help clean up your local area! Upload an image and get +1 if there is all of the following items detected: trash, bin, hand.

Model is a fine-tuned version of [RT-DETRv2](https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config) on the [Trashify dataset](https://huggingface.co/datasets/mrdbourke/trashify_manual_labelled_images).

See the full data loading and training code on [learnhuggingface.com](https://www.learnhuggingface.com/notebooks/hugging_face_object_detection_tutorial).

This version is v4 because the first three versions were using a different model and did not perform as well, see the [README](https://huggingface.co/spaces/mrdbourke/trashify_demo_v4/blob/main/README.md) for more.
"""

# Create the Gradio interface to accept an image and confidence threshold and return an image with drawn prediction boxes
demo = gr.Interface(
    fn=predict_on_image,
    inputs=[
        gr.Image(type="pil", label="Target Image"),
        gr.Slider(minimum=0, maximum=1, value=0.3, label="Confidence Threshold")
    ],
    outputs=[
        gr.Image(type="pil", label="Image Output"),
        gr.Text(label="Text Output")
    ],
    title="🚮 Trashify Object Detection Demo V4",
    description=description,
    # Examples come in the form of a list of lists, where each inner list contains elements to prefill the `inputs` parameter with
    # See where the examples originate from here: https://huggingface.co/datasets/mrdbourke/trashify_examples/
    examples=[
        ["trashify_examples/trashify_example_1.jpeg", 0.3],
        ["trashify_examples/trashify_example_2.jpeg", 0.3], 
        ["trashify_examples/trashify_example_3.jpeg", 0.3],
    ],
    cache_examples=True
)

# Launch the demo
demo.launch()
