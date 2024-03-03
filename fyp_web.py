import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import base64
# from streamlit_drawable_canvas import st_canvas
from streamlit_image_zoom import image_zoom  # Import the image_zoom function

def morphological_processing_with_canny(image, threshold1, threshold2):
    umat_image = cv2.UMat(image)
    kernel = np.ones((2, 2), np.uint8)
    canny_edges = cv2.Canny(umat_image, threshold1=threshold1, threshold2=threshold2)
    canny_edges = np.asarray(canny_edges.get())
    gradient = cv2.morphologyEx(canny_edges, cv2.MORPH_GRADIENT, kernel)
    return gradient

def morphological_processing(image):
    kernel = np.ones((2, 2), np.uint8)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return gradient

def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_brightness_contrast(image, alpha, beta):
    return cv2.addWeighted(image, alpha, image, 0, beta)

def modify_image(image, t1, t2, option, draw_bbox = False, bounding_box = ((0,0), (0, 0))):
    if option == "No":
        enhanced_contrast = adjust_contrast(image, 1.5)
        enhanced_contrast = np.array(enhanced_contrast)

        x = morphological_processing(enhanced_contrast)
        kernel = np.ones((3, 3), np.uint8)
        gradient = cv2.morphologyEx(x, cv2.MORPH_GRADIENT, kernel)

        t1 = 1 + (t1 * 0.01)
        enhanced_image = adjust_brightness_contrast(gradient, t1, t2)

        enhanced_image = np.array(enhanced_image)
        if len(enhanced_image.shape) > 2:  # Check if the image has multiple channels
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        # enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

        if draw_bbox:
            enhanced_image = draw_bounding_box(enhanced_image, bounding_box)

        return enhanced_image

    if option == "Yes":
        image = np.array(image)
        modified_image = morphological_processing_with_canny(image, t1, t2)
        if draw_bbox:
            modified_image = draw_bounding_box(modified_image, bounding_box)

        return modified_image
    
def draw_bounding_box(image, bounding_box):
    drawn_image = image.copy()
    cv2.rectangle(drawn_image, tuple(bounding_box[0]), tuple(bounding_box[1]), (255, 255, 0), 2)
    return drawn_image


def download_image(image):
    pil_image = Image.fromarray(image)
    temp_file_path = "modified_image.png"
    pil_image.save(temp_file_path, "PNG")

    try:
        with open(temp_file_path, "rb") as file:
            contents = file.read()

        encoded_file = base64.b64encode(contents).decode()
        href = f'<a href="data:file/png;base64,{encoded_file}" download="modified_image.png">Download modified image</a>'
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred during download: {e}")

def main():
    st.set_page_config(layout='wide')
    st.title("Image Uploader, Modifier, and Downloader")

    col1, col2 = st.columns(2)

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "JPG", "JPEG"])

    option = st.sidebar.selectbox(
        "Would you like to use edge detection?",
        ("Choose either Yes or No", "Yes", "No"),
        index=0
    )

    if option is not None and option == "Choose either Yes or No":
        st.write("Please choose a valid option - Yes or No")
        st.stop()

    st.sidebar.subheader("Parameters")
    if option == "No":
        threshold = st.sidebar.slider("Contrast increase by : ", min_value=0, max_value=100, value=40)
        threshold2 = st.sidebar.slider("Brightness increase by adding : ", min_value=0, max_value=100, value=5)

    if option == "Yes":
        threshold = st.sidebar.slider("Threshold 1 : ", min_value=0, max_value=100, value=12)
        threshold2 = st.sidebar.slider("Threshold 2 : ", min_value=0, max_value=400, value=76)

    zoom_factor = st.sidebar.slider("Zoom Factor : ", min_value=1.0, max_value=5.0, value=2.0)

    # Initialize draw_bbox with default value
    draw_bbox = st.sidebar.checkbox("Draw bounding box on the image", value=False)
    if draw_bbox == True:
        st.sidebar.subheader("Bounding Box Parameters")
        x1 = st.sidebar.slider("X1", min_value=0, max_value=1000, value=50)
        y1 = st.sidebar.slider("Y1", min_value=0, max_value=1000, value=50)
        x2 = st.sidebar.slider("X2", min_value=0, max_value=1000, value=150)
        y2 = st.sidebar.slider("Y2", min_value=0, max_value=1000, value=150)

        # Define the bounding box
        bounding_box = ((x1, y1), (x2, y2))

    with col1:
        st.subheader("Uploaded Image")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.subheader("Modified Image")
            if uploaded_file is not None:
                # Modify and display the image
                if draw_bbox == True:
                    modified_image = modify_image(image, threshold, threshold2, option, draw_bbox, bounding_box)
                else:
                    modified_image = modify_image(image, threshold, threshold2, option)

                # Use the image_zoom function to display the modified image with zoom functionality
                image_zoom(modified_image, mode="mousemove", size=512, zoom_factor=zoom_factor)

                
                        # Perform any additional processing here using bounding box coordinates
                if st.sidebar.button("Process Bounding Box"):
                    # Perform any additional processing here
                    # For example, you can extract the region inside the bounding box
                    min_y = min(y1, y2)
                    max_y = max(y1, y2)
                    min_x = min(x1, x2)
                    max_x = max(x1, x2)
                    region_of_interest = modified_image[min_y:max_y, min_x:max_x]

                        # Display the extracted region
                    st.sidebar.image(region_of_interest, caption="Extracted Region",
                                            use_column_width=True)
                    
                            # Display the modified image with drawn bounding boxes
                    st.sidebar.image(modified_image, caption="Modified Image", use_column_width=True)

                # Download button for modified image
                download_button = st.button("Download Modified Image")
                if download_button:
                    download_image(modified_image)

if __name__ == "__main__":
    main()
