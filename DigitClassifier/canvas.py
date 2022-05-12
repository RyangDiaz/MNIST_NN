import math
import tkinter as tk

import numpy as np
from PIL import Image
import os


class App:
    def __init__(self, classifier, width=200, height=200):
        self.root = tk.Tk()
        self.root.title("Drawing Pad")
        self.root.geometry(f"{width+2*20}x{height+(height//5)+(2*20)}")
        self.root.resizable(width=False, height=False)

        self.width = width
        self.height = height
        self.width_pad = 20
        self.height_pad = 20

        self.network = classifier

        canvas_panel = tk.Frame(self.root, width=width+2*self.width_pad, height=height+2*self.height_pad)
        button_panel = tk.Frame(self.root, width=width+2*self.width_pad, height=height//4+2*self.height_pad)

        canvas_panel.pack(side='top')
        button_panel.pack(side='bottom')

        self._create_buttons(button_panel)
        self._create_canvas(canvas_panel)

        self.canvas.bind("<Button-1>", self.save_posn)
        self.canvas.bind("<B1-Motion>", self.add_line)

    def _create_buttons(self, parent):
        width = self.width
        width_pad = self.width_pad

        erase_button = tk.Button(parent, text="Erase", width=int(0.4*width_pad), height=1, command=self.erase_button)
        save_png = tk.Button(parent, text="Save/Predict", width=int(0.4*width_pad), height=1, command=self.save_and_predict)

        train_0 = tk.Button(parent, text="0", width=int(0.2*width_pad), height=1, command=self.train_0)
        train_1 = tk.Button(parent, text="1", width=int(0.2 * width_pad), height=1, command=self.train_1)
        train_2 = tk.Button(parent, text="2", width=int(0.2 * width_pad), height=1, command=self.train_2)
        train_3 = tk.Button(parent, text="3", width=int(0.2 * width_pad), height=1, command=self.train_3)
        train_4 = tk.Button(parent, text="4", width=int(0.2 * width_pad), height=1, command=self.train_4)
        train_5 = tk.Button(parent, text="5", width=int(0.2 * width_pad), height=1, command=self.train_5)
        train_6 = tk.Button(parent, text="6", width=int(0.2 * width_pad), height=1, command=self.train_6)
        train_7 = tk.Button(parent, text="7", width=int(0.2 * width_pad), height=1, command=self.train_7)
        train_8 = tk.Button(parent, text="8", width=int(0.2 * width_pad), height=1, command=self.train_8)
        train_9 = tk.Button(parent, text="9", width=int(0.2 * width_pad), height=1, command=self.train_9)

        erase_button.grid(row=0, column=0)
        save_png.grid(row=0, column=1)
        train_0.grid(row=1, column=0)
        train_1.grid(row=1, column=1)
        train_2.grid(row=1, column=2)
        train_3.grid(row=1, column=3)
        train_4.grid(row=1, column=4)
        train_5.grid(row=1, column=5)
        train_6.grid(row=2, column=0)
        train_7.grid(row=2, column=1)
        train_8.grid(row=2, column=2)
        train_9.grid(row=2, column=3)

    def _create_canvas(self, parent):
        width = self.width
        height = self.height
        width_pad = self.width_pad
        height_pad = self.height_pad

        temp = tk.Frame(parent, width=width+2*width_pad, height=height_pad)
        temp.grid(row=0)
        temp = tk.Frame(parent, width=width_pad, height=height)
        temp.grid(row=1, column=0)
        self.canvas = tk.Canvas(parent, width=width-8, height=height-7, background='white')
        self.canvas.grid(row=1, column=1)
        temp = tk.Frame(parent, width=width_pad, height=height)
        temp.grid(row=1, column=2)
        temp = tk.Frame(parent, width=width+2*width_pad, height=height_pad)
        temp.grid(row=2)

        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

    def save_posn(self, event):
        self.lastx, self.lasty = event.x, event.y

    def add_line(self, event):
        d = math.sqrt((event.x - self.lastx)**2 + (event.y - self.lasty)**2)
        if d > 1:
            self.canvas.create_line((self.lastx, self.lasty, event.x, event.y), width=10, fill='black', tags='line')
            self.save_posn(event)

    def erase_button(self):
        self.canvas.delete('line')

    # Training buttons

    def train_0(self):
        self.save_and_train(0)

    def train_1(self):
        self.save_and_train(1)

    def train_2(self):
        self.save_and_train(2)

    def train_3(self):
        self.save_and_train(3)

    def train_4(self):
        self.save_and_train(4)

    def train_5(self):
        self.save_and_train(5)

    def train_6(self):
        self.save_and_train(6)

    def train_7(self):
        self.save_and_train(7)

    def train_8(self):
        self.save_and_train(8)

    def train_9(self):
        self.save_and_train(9)

    def save_and_predict(self):
        self.save_as_png()
        input = self.change_res(28, 28)
        self.predict_digit(input)

    def save_and_train(self, digit):
        self.save_as_png()
        input = self.change_res(28, 28)
        expected = np.zeros((10, 1))
        expected[digit] = 1
        self.network.train(input, expected)
        print(f"Trained model as {digit}")
        self.erase_button()

    def save_as_png(self):
        self.canvas.postscript(file='temp.eps')
        img_eps = Image.open('temp.eps')
        img = img_eps.convert('LA')
        img.save('output.png', quality=25)
        os.remove('temp.eps')

    def change_res(self, new_width, new_height):
        final_grid = np.zeros((new_height, new_width))

        image_file = 'output.png'
        width = self.width
        height = self.height

        if new_width > width or new_height > height:
            print("Unable to change to higher resolution")
            return

        try:
            Image.open(image_file)
        except FileNotFoundError:
            print(f"A file of the name {image_file} is not found in project directory...")
            return

        with Image.open(image_file, 'r') as image:
            image_pixel_value = [i[0] for i in list(image.getdata())]
            for i in range(len(image_pixel_value)):
                image_pixel_value[i] /= 255
            image_pixel_value = np.reshape(image_pixel_value, (height, width))

        ratioH = height/new_height
        ratioW = width/new_width

        for i in range(self.height):
            for j in range(self.width):
                final_grid[(math.floor(i/ratioH), math.floor(j/ratioW))] = image_pixel_value[(i, j)]

        vector = np.reshape(final_grid, (784, 1))
        return vector

    def predict_digit(self, input):
        output = self.network.predict(input)
        print("Predicted Value:", np.argmax(output))
        print(output)
