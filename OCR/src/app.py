import cv2
import torch
import tkinter as tk
from tkinter import filedialog
from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn_v2
from PIL import Image, ImageTk
from utils import predict
from model import CRNN


max_object_line = 50
num_classes_line = 3
num_classes_char = 6000
line_size = (384, 24)
initial_image_pos = (225, 300)
initial_zoom_level = 0.6
confidence_threshold = 0.8

nc = 1
nh = 384
nclass = num_classes_char + 1
height = 24

character_source_traditional = "data/source/CCL6kT.txt"
character_source_simplified = "data/source/CCL6kS.txt"
model_path_det = 'models/frcnnv2_det_50_mixed.pth'
model_path_rec = 'models/crnn_rec_6000_mixed.pth'


class ImageApp:
    def __init__(self, root):
        self.load_character_list()
        self.load_model()
        self.root = root
        self.root.title("\"The Women's Journal\" OCR Demo")
        self.image_loaded = False
        self.text_traditional = ""
        self.text_simplified = ""
        self.text_traditional_break = ""
        self.text_simplified_break = ""

        self.label = tk.Label(root, text="No image selected")
        self.label.pack(pady=(20, 0))

        self.zoom_level = initial_zoom_level
        self.image = None
        self.image_id = None
        self.image_pos = initial_image_pos

        # Create a frame to hold the canvas and text box side by side
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(pady=20, padx=20)

        # Canvas to display the image
        self.canvas = tk.Canvas(self.main_frame, width=450, height=600)
        self.canvas.grid(row=0, column=0, rowspan=2, padx=10)

        # Text box to display text
        self.text_box = tk.Text(self.main_frame, width=50, height=44, highlightthickness=0, state="disabled")
        self.text_box.grid(row=0, column=1, padx=10)

        # Frame to hold the checkboxes
        self.checkbox_frame = tk.Frame(self.main_frame)
        self.checkbox_frame.grid(row=1, column=1)

        # Checkboxes
        self.simplified = tk.BooleanVar()
        self.reference = tk.BooleanVar()
        self.checkbox1 = tk.Checkbutton(self.checkbox_frame, text="Simplified", variable=self.simplified, command=self.update_text)
        self.checkbox2 = tk.Checkbutton(self.checkbox_frame, text="Reference", variable=self.reference, command=self.update_text)
        self.checkbox1.pack(side=tk.LEFT, padx=5)
        self.checkbox2.pack(side=tk.LEFT, padx=5)

        # Frame to hold the buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=(0, 20))

        # Button to select an image
        self.select_button = tk.Button(self.button_frame, text="Select Image", command=self.load_image)
        self.select_button.pack(side=tk.LEFT, padx=10)

        # Additional button
        self.other_button = tk.Button(self.button_frame, text="Recognize Text", command=self.recognize_text)
        self.other_button.pack(side=tk.LEFT, padx=10)

        # Bind the mouse wheel to the zoom function
        self.canvas.bind("<MouseWheel>", self.zoom_image)

        # Bind mouse events for dragging
        self.canvas.bind("<Button-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag_image)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drag)
        
        self.drag_data = {"x": 0, "y": 0, "x_offset": 0, "y_offset": 0}

    def load_character_list(self):
        with open(character_source_traditional, "r", encoding='UTF-8') as file:
            # Read all characters at once
            characters = file.read()
            # Convert the string to a list of characters
            self.character_list_traditional = list(characters)

        with open(character_source_simplified, "r", encoding='UTF-8') as file:
            # Read all characters at once
            characters = file.read()
            # Convert the string to a list of characters
            self.character_list_simplified = list(characters)

        # Create a dictionary to map traditional characters to simplified characters
        self.character_map = dict(zip(self.character_list_traditional, self.character_list_simplified))

    def load_model(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f'Using {self.device}')

        # Load pre-trained Faster R-CNN model for line detection
        self.model_line = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT', box_detections_per_img=max_object_line)
        in_features = self.model_line.roi_heads.box_predictor.cls_score.in_features
        self.model_line.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes_line)
        state_dict = torch.load(model_path_det) if torch.cuda.is_available() else torch.load(model_path_det, map_location=torch.device('cpu'))
        self.model_line.load_state_dict(state_dict)
        self.model_line = self.model_line.eval().to(self.device)
        print("Load detection model:", model_path_det)

        # Load pre-trained CRNN model for text recognition
        self.model_char = CRNN(nc, nh, nclass, height)
        state_dict = torch.load(model_path_rec) if torch.cuda.is_available() else torch.load(model_path_rec, map_location=torch.device('cpu'))
        self.model_char.load_state_dict(state_dict)
        self.model_char.eval().to(self.device)
        print("Load recognition model:", model_path_rec)

    def load_image(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if self.file_path:
            self.label.config(text=self.file_path)
            self.image = Image.open(self.file_path)
            self.zoom_level = initial_zoom_level
            self.image_pos = initial_image_pos
            self.show_image()
            self.image_loaded = True
            self.text_traditional = ""
            self.text_simplified = ""
            self.text_traditional_break = ""
            self.text_simplified_break = ""
            self.update_text()
            self.root.focus_force()

    def show_image(self):
        if self.image is not None:
            width, height = self.image.size
            new_size = int(width * self.zoom_level), int(height * self.zoom_level)
            resized_image = self.image.resize(new_size)
            self.photo = ImageTk.PhotoImage(resized_image)
            if self.image_id is not None:
                self.canvas.delete(self.image_id)
            self.image_id = self.canvas.create_image(*self.image_pos, image=self.photo)

    def zoom_image(self, event, max_zoom=2.0, min_zoom=0.5):
        if self.image is None:
            return
        if event.delta > 0 and self.zoom_level < max_zoom:
            self.zoom_level *= 1.1  # Zoom in
        elif event.delta < 0 and self.zoom_level > min_zoom:
            self.zoom_level /= 1.1  # Zoom out
        if self.zoom_level <= max_zoom and self.zoom_level >= min_zoom:
            self.show_image()
        #print(self.zoom_level)

    def start_drag(self, event):
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def drag_image(self, event):
        if self.image_id is None:
            return
        delta_x = event.x - self.drag_data["x"]
        delta_y = event.y - self.drag_data["y"]
        self.image_pos = (self.image_pos[0] + delta_x, self.image_pos[1] + delta_y)
        self.canvas.move(self.image_id, delta_x, delta_y)
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def stop_drag(self, event):
        self.drag_data["x"] = 0
        self.drag_data["y"] = 0

    def recognize_text(self):
        if not self.image_loaded:
            return
        text_output, image_output, confidence = predict('', self.file_path, '', self.model_line, self.model_char, self.device, self.character_list_traditional, num_classes_char, line_size, return_image_and_confidence=True, line_break=True)

        text_output_list = text_output.split(sep='\n')
        for i in range(len(text_output_list)):
            self.text_traditional_break += str(i+1) + '. ' + text_output_list[i] + '\n'
        self.text_simplified_break = ''.join([self.character_map.get(c, c) for c in self.text_traditional_break])
        self.text_traditional = ''.join(text_output_list)
        self.text_simplified = ''.join([self.character_map.get(c, c) for c in self.text_traditional])

        self.update_text()
        label_text = f"Text recognized with confidence score of {confidence:.2f}"
        if confidence < confidence_threshold:
            label_text += ", manual correction may be needed"
        self.label.config(text=label_text)
        
        self.image = Image.fromarray(cv2.cvtColor(image_output, cv2.COLOR_BGR2RGB))
        self.zoom_level = initial_zoom_level
        self.image_pos = initial_image_pos
        self.show_image()
        self.image_loaded = False

    def update_text(self):
        if self.simplified.get():
            text_output = self.text_simplified_break if self.reference.get() else self.text_simplified
        else:
            text_output = self.text_traditional_break if self.reference.get() else self.text_traditional
        self.text_box.config(state="normal")
        self.text_box.delete(1.0, "end")
        self.text_box.insert("end", text_output)
        self.text_box.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
