import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageProcessingApp:
    def __init__(self, root):
        # Initialize the application
        self.root = root
        self.root.title("Image Processing App")
        
        # Initialize image variables
        self.image = None
        self.processed_image = None
        
        # Create buttons
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()
        
        # Create frame for filter buttons
        self.filter_frame = tk.Frame(root)
        self.filter_frame.pack()
        
        # Create filter buttons
        self.create_filter_buttons()
        
        # Create reset button
        self.reset_button = tk.Button(root, text="Reset", command=self.reset)
        self.reset_button.pack()
    
    def create_filter_buttons(self):
        # Create filter buttons based on a list of filter names
        filters = ["LPF", "HPF", "MEAN", "MEDIAN", "Roberts", "Prewitt", "Sobel", "Erosion", "Dilation", "Open", "Close", "Hough Circle", "Region Split and Merge", "Thresholding"]
        
        for i, filter_name in enumerate(filters):
            # Create a button for each filter
            button = tk.Button(self.filter_frame, text=filter_name, command=lambda f=filter_name: self.apply_filter(f))
            # Place buttons in a 3-column grid
            button.grid(row=i//3, column=i%3, padx=5, pady=5)
    
    def upload_image(self):
        # Function to upload an image
        file_path = filedialog.askopenfilename()
        if file_path:
            # Read and display the selected image
            self.image = cv2.imread(file_path)
            self.processed_image = self.image.copy()
            self.display_image(self.processed_image)
    
    def display_image(self, image):
        # Function to display an image in the application window
        max_width = 1080
        img_height, img_width = image.shape[:2]  # Adjusted to handle grayscale images
        if img_width > max_width:
            ratio = max_width / img_width
            image = cv2.resize(image, (max_width, int(img_height * ratio)))
        
        if len(image.shape) == 3:  # RGB image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        else:  # Grayscale image
            image = Image.fromarray(image)
            image = image.convert("RGB")  # Convert grayscale to RGB for displaying
        
        image = ImageTk.PhotoImage(image)
        
        if hasattr(self, "panel"):
            self.panel.configure(image=image)
            self.panel.image = image
        else:
            self.panel = tk.Label(self.root, image=image)
            self.panel.image = image
            self.panel.pack()
    
    def apply_filter(self, filter_name):
        # Function to apply various filters to the image
        if self.image is None:
            return
        
        if filter_name == "LPF":
            # Apply Low Pass Filter
            self.processed_image = cv2.GaussianBlur(self.processed_image, (5, 5), 0)
        
        elif filter_name == "HPF":
            # Apply High Pass Filter
            self.processed_image = cv2.filter2D(self.processed_image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
        
        elif filter_name == "MEAN":
            # Apply Mean Filter
            self.processed_image = cv2.blur(self.processed_image, (5, 5))
        
        elif filter_name == "MEDIAN":
            # Apply Median Filter
            self.processed_image = cv2.medianBlur(self.processed_image, 5)
        
        elif filter_name in ["Roberts", "Prewitt", "Sobel"]:
            # Apply edge detection filters
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            if filter_name == "Roberts":
                self.processed_image = cv2.filter2D(gray, -1, np.array([[-1, 0], [0, 1]]))
            
            elif filter_name == "Prewitt":
                kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
                kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                self.processed_image = cv2.filter2D(gray, -1, kernelx) + cv2.filter2D(gray, -1, kernely)

            elif filter_name == "Sobel":
                self.processed_image = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
        
        elif filter_name == "Erosion":
            # Apply Erosion
            kernel = np.ones((5,5),np.uint8)
            self.processed_image = cv2.erode(self.processed_image,kernel,iterations = 1)
        
        elif filter_name == "Dilation":
            # Apply Dilation
            kernel = np.ones((5,5),np.uint8)
            self.processed_image = cv2.dilate(self.processed_image,kernel,iterations = 1)
        
        elif filter_name == "Open":
            # Apply Opening Morphological Operation
            kernel = np.ones((5,5),np.uint8)
            self.processed_image = cv2.morphologyEx(self.processed_image, cv2.MORPH_OPEN, kernel)
        
        elif filter_name == "Close":
            # Apply Closing Morphological Operation
            kernel = np.ones((5,5),np.uint8)
            self.processed_image = cv2.morphologyEx(self.processed_image, cv2.MORPH_CLOSE, kernel)
        
        elif filter_name == "Hough Circle":
            # Apply Hough Circle Transform
            circles = cv2.HoughCircles(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(self.processed_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        
        elif filter_name == "Region Split and Merge":
            # Apply Region Split and Merge
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            self.processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
        elif filter_name == "Thresholding":
            # Apply Thresholding
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            self.processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        # Display the processed image
        self.display_image(self.processed_image)
    
    def reset(self):
        # Reset the image to the original uploaded image
        if self.image is not None:
            self.processed_image = self.image.copy()
            self.display_image(self.processed_image)

root = tk.Tk()
app = ImageProcessingApp(root)
root.mainloop()
