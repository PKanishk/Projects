import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model
from keras.layers import Layer, Dropout
from tensorflow import keras
from tensorflow.keras.layers import Layer, Dropout

# Define the custom layer if it's not already defined in your code
class FixedDropout(Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(FixedDropout, self).__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed

    def build(self, input_shape):
        self.dropout = Dropout(self.rate, noise_shape=self.noise_shape, seed=self.seed)
        super(FixedDropout, self).build(input_shape)

    def call(self, inputs, training=None):
        return self.dropout(inputs, training=training)

# Load the model with custom_objects parameter
with keras.utils.custom_object_scope({'FixedDropout': FixedDropout}):
    skin_disease_model = load_model("C:/Users/91900/Desktop/CORE PROJECT/lenet.h5")

# Dictionary to label skin disease classes
skin_disease_classes = {
    0: 'Melanocytic Nevi (nv)',
    1: 'Melanoma (mel)',
    2: 'Benign Keratosis-like Lesions (bkl)',
    3: 'Basal Cell Carcinoma (bcc)',
    4: 'Actinic Keratoses (akiec)',
    5: 'Vascular Lesions (vasc)',
    6: 'Dermatofibroma (df)',
    7: 'Squamous Cell Carcinoma (scc)'
}

top_skin = tk.Tk()
top_skin.geometry('800x600')
top_skin.title('Skin Disease Detection')
top_skin.configure(background='#CDCDCD')

label_skin = Label(top_skin, background='#CDCDCD', font=('arial', 15, 'bold'))
skin_image = Label(top_skin)

def classify_skin(file_path):
    try:
        global label_skin
        print(f"File Path: {file_path}")
        image_skin = Image.open(file_path)
        # Resize the image to match the expected input shape (32x32 pixels)
        image_skin = image_skin.resize((32, 32))
        image_skin = np.expand_dims(image_skin, axis=0)
        image_skin = np.array(image_skin)
        pred_skin = np.argmax(skin_disease_model.predict(image_skin), axis=-1)[0]
        disease_label = skin_disease_classes.get(pred_skin, 'Unknown Disease')
        print(f"Disease Label: {disease_label}")
        label_skin.configure(foreground='#011638', text=disease_label)
    except Exception as e:
        print(f"Error in classify_skin: {str(e)}")

def show_classify_button_skin(file_path):
    classify_b_skin = Button(top_skin, text="Detect Disease", command=lambda: classify_skin(file_path), padx=10, pady=5)
    classify_b_skin.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b_skin.place(relx=0.79, rely=0.46)

def upload_image_skin():
    try:
        file_path_skin = filedialog.askopenfilename()
        uploaded_skin = Image.open(file_path_skin)
        uploaded_skin.thumbnail(((top_skin.winfo_width() / 4.25), (top_skin.winfo_height() / 4.25)))
        im_skin = ImageTk.PhotoImage(uploaded_skin.convert("RGB"))

        skin_image.configure(image=im_skin)
        skin_image.image = im_skin
        label_skin.configure(text='')
        show_classify_button_skin(file_path_skin)
    except Exception as e:
        print(f"Error in upload_image_skin: {str(e)}")

upload_skin = Button(top_skin, text="Upload an image", command=upload_image_skin, padx=30, pady=10)
upload_skin.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

upload_skin.pack(side=BOTTOM, pady=50)
skin_image.pack(side=BOTTOM, expand=True)
label_skin.pack(side=BOTTOM, expand=True)
heading_skin = Label(top_skin, text="Skin Disease Detection", pady=20, font=('arial', 20, 'bold'))
heading_skin.configure(background='#CDCDCD', foreground='#364156')
heading_skin.pack()

top_skin.mainloop()
