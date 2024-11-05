import streamlit as st
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
from ultralytics import YOLO
from torchvision import transforms as torchtrans


# Load the YOLOv8 model (you can replace the path with your trained model path)
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Change this to your model's path if needed
    return model


# Plot bounding boxes on the image
def plot_img_bbox(img, boxes):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    ax.imshow(img)

    # boxes should be in (x1, y1, x2, y2) format
    for box in boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
    return fig, ax


# Core image processing function
def processing_image(model):
    image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg', 'tif'])

    if image_file is not None:
        img = Image.open(image_file).convert('RGB')
        file_details = {"FileName": image_file.name, "FileType": image_file.type}
        st.write(file_details)

        # Display uploaded image
        st.image(img, caption='Uploaded Image.')

        # Save the image
        with open(image_file.name, mode="wb") as f:
            f.write(image_file.getbuffer())

        st.success("Saved File")

        # Process the image
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torchtrans.ToTensor()(img_np).unsqueeze(0)

        # Run the model inference (YOLOv8)
        results = model(img_tensor)  # Perform inference with YOLOv8

        # Get the first image's result (YOLOv8 can return results for multiple images)
        result = results[0]  # The result for the first image

        # Extract bounding boxes (xywh format)
        boxes = result.boxes.xywh.cpu().numpy()  # (x, y, w, h) format

        # Convert bounding boxes to (x1, y1, x2, y2) format for plotting
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes_xyxy = np.column_stack((x1, y1, x2, y2))

        # Plot bounding boxes on the image
        fig, ax = plot_img_bbox(img_np, boxes_xyxy)

        buf = BytesIO()
        fig.savefig(buf, format="png")

        # Display processed image with bounding boxes
        st.image(buf, caption='Processed Image with Bounding Boxes.')


def processing_video(x):
    st.write("Coming soon!")


# Main function for Streamlit app
def main():
    st.set_page_config(page_title="YOLOv8 Instance Segmentation App", page_icon=":camera:", layout="wide",
                       initial_sidebar_state="expanded")
    with st.expander("About the App"):
        st.markdown('<p style="font-size: 30px;"><strong>Welcome to YOLOv8 Instance Segmentation App!</strong></p>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size: 20px;">This app demonstrates <strong>YOLOv8 Instance Segmentation</strong> for both images and videos.</p>',
            unsafe_allow_html=True)

    option = st.selectbox('What Type of File do you want to work with?', ('Images', 'Videos'))

    if option == "Images":
        st.title('Instance Segmentation for Images')
        st.subheader("Upload an image and the model will provide the segmented bounding boxes.")

        # Load the YOLOv8 model
        model = load_model()
        processing_image(model)
    else:
        st.title('Instance Segmentation for Videos')
        st.subheader("Video processing will be available soon.")
        processing_video()


if __name__ == '__main__':
    main()
