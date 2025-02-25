#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python dlib imutils numpy')


# In[2]:


get_ipython().system('pip install cmake')


# In[3]:


get_ipython().system('pip install dlib')


# In[ ]:


get_ipython().system('pip install imutils')


# In[4]:


import cv2
import dlib
import imutils
import numpy as np

print("Libraries installed successfully!")


# In[1]:


import cv2
import dlib
import numpy as np
import imutils

# Load the image
image_path = "IMG_4189 2.JPG"
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

# Detect faces in the image
faces = detector(gray_image, 1)  # Use 1 for detecting faces, higher values can be used for more accuracy

# Check if faces are detected
if len(faces) == 0:
    print("No faces detected.")
else:
    print(f"Found {len(faces)} face(s).")




# In[ ]:


import os

# Paths
image_path = "IMG_4189 2.JPG"
predictor_path = "shape_predictor_68_face_landmarks.dat"

# Check if files exist
if not os.path.exists(image_path):
    print("❌ Image file not found!")
    exit()

if not os.path.exists(predictor_path):
    print("❌ Model file not found! Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

# Load the image
image = cv2.imread(image_path)

if image is None:
    print("❌ Error: Image not loaded!")
    exit()

# Resize and convert to grayscale
image = imutils.resize(image, width=600)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Detect faces
faces = detector(gray_image, 0)  # Change 1 to 0 if faces aren't detected

print(f"✅ Faces detected: {len(faces)}")

if len(faces) == 0:
    print("❌ No faces detected. Try another image.")
    exit()

# Draw landmarks
for face in faces:
    landmarks = predictor(gray_image, face)

    for n in range(0, 68):
        x, y = landmarks.part(n).x, landmarks.part(n).y
        print(f"Landmark {n}: ({x}, {y})")  # Debugging output
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

# Show image
cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import cv2
import dlib
import imutils
import numpy as np
import ipywidgets as widgets
from IPython.display import display, Image
from io import BytesIO

# Helper function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Jewelry recommendations based on face shape
def get_jewelry_recommendations(face_shape):
    if face_shape == "Round":
        return "For a round face: Try long or angular earrings like chandelier or drop earrings to add length. A V-shaped or long necklace will also elongate your face."
    elif face_shape == "Oval":
        return "For an oval face: Most styles work, but oval or teardrop earrings can add balance. Short necklaces or chokers complement your face shape well."
    elif face_shape == "Heart":
        return "For a heart-shaped face: Drop or teardrop earrings will balance the sharp chin. V-neck or low-hanging necklaces help create a more proportionate look."
    elif face_shape == "Square":
        return "For a square face: Soft, rounded earrings like hoops or drop earrings will soften angular features. Curved or rounded necklaces, like U-shaped ones, are great for softening the jawline."
    else:
        return "Jewelry recommendations not available."

# Load the model (make sure it's in the same directory or provide full path)
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Create file upload widget
upload_widget = widgets.FileUpload(accept='.jpg, .jpeg, .png', multiple=False)

# Function to process the uploaded image
def on_upload_change(change):
    try:
        # Get the uploaded file
        file_content = upload_widget.value[0]['content']
        file_bytes = np.asarray(bytearray(file_content), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Resize and convert to grayscale
        image = imutils.resize(image, width=600)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray_image, 0)

        if len(faces) == 0:
            print("❌ No faces detected. Try uploading another image.")
        else:
            for face in faces:
                landmarks = predictor(gray_image, face)

                # Get key points for jawline, cheeks, chin, and forehead
                jawline_points = [landmarks.part(i) for i in range(17)]
                left_cheek = [landmarks.part(i) for i in range(1, 5)]
                right_cheek = [landmarks.part(i) for i in range(5, 9)]
                chin = [landmarks.part(i) for i in range(17, 23)]
                forehead = landmarks.part(27)  # Point 27 is the forehead top

                # Calculate distances for face shape classification
                cheek_width = euclidean_distance(left_cheek[3], right_cheek[3])
                face_height = euclidean_distance(forehead, landmarks.part(8))  # Point 8 is chin
                jawline_width = euclidean_distance(jawline_points[0], jawline_points[16])  # Jawline width
                chin_width = euclidean_distance(chin[0], chin[4])

                # Refined classification logic:
                if cheek_width == face_height:  # Round face shape
                    face_shape = "Round"
                elif face_height > cheek_width:  # Oval or Heart
                    if chin_width < cheek_width * 0.7:  # Narrow chin = Heart face
                        face_shape = "Heart"
                    else:  # Oval face shape (longer face)
                        face_shape = "Oval"
                elif jawline_width > face_height * 0.8:  # Strong jawline = Square
                    face_shape = "Square"
                else:  # Heart-shaped face
                    face_shape = "Heart"

                # Jewelry recommendations
                jewelry_recommendations = get_jewelry_recommendations(face_shape)

                # Print the recommendations for the user
                print(f"Detected Face Shape: {face_shape}")
                print(jewelry_recommendations)

                # Draw the result on the image
                x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

                # Display the face shape text on the image
                cv2.putText(image, f"Face Shape: {face_shape}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Convert image to display in notebook
            _, buffer = cv2.imencode('.png', image)
            display_image = buffer.tobytes()
            display(Image(data=display_image))
    except Exception as e:
        print(f"Error: {e}")

# Display the widget
upload_widget.observe(on_upload_change, names='value')
display(upload_widget)


# In[ ]:


import ipywidgets as widgets
from IPython.display import display, Image

# Example Jewelry List (Ensure the product list has unique items)
products = [
    {"name": "Gold Hoop Earrings", "type": "Earrings", "material": "Gold", "face_shape": "Round", "image": "goldearrings.jpeg"},
    {"name": "Silver Necklace", "type": "Necklace", "material": "Silver", "face_shape": "Oval", "image": "silver.jpg"},
    {"name": "Platinum Stud Earrings", "type": "Earrings", "material": "Platinum", "face_shape": "Heart", "image": "Platinum Stud Earrings.jpg"},
    # Add more products as needed
]

# Define user input widgets
material_widget = widgets.Dropdown(
    options=['Gold', 'Silver', 'Platinum'],
    value='Gold',
    description='Material:'
)

jewelry_type_widget = widgets.Dropdown(
    options=['Earrings', 'Necklace', 'Ring'],
    value='Earrings',
    description='Jewelry Type:'
)

face_shape_widget = widgets.Dropdown(
    options=['Round', 'Oval', 'Heart'],
    value='Round',
    description='Face Shape:'
)

# Function to filter and display products based on user selection
def filter_products(material, jewelry_type, face_shape):
    filtered_products = [
        product for product in products 
        if product["material"] == material and product["type"] == jewelry_type and product["face_shape"] == face_shape
    ]
    
    # Clear the previous output before displaying new products
    display_output.clear_output(wait=True)

    if filtered_products:
        for product in filtered_products:
            with display_output:
                display(Image(filename=product["image"], width=100, height=100))
                print(f"{product['name']} - {product['material']} {product['type']}")
    else:
        with display_output:
            print("No matching products found.")

# Create a container to hold the filtered products display
display_output = widgets.Output()

# Create a button to trigger filtering and display results
button = widgets.Button(description="Get Jewelry Suggestions")
button.on_click(lambda x: filter_products(material_widget.value, jewelry_type_widget.value, face_shape_widget.value))

# Display widgets and button
display(widgets.VBox([material_widget, jewelry_type_widget, face_shape_widget, display_output, button]))


# In[ ]:




