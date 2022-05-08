import tkinter as tk

import numpy as np
from PIL import Image
import os

from neuralnetwork import NeuralNetwork

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Drawing Pad")
        self.root.geometry("600x700")
        self.root.resizable(width=False, height=False)

        canvas_panel = tk.Frame(self.root, background='black', width=600, height=600)
        button_panel = tk.Frame(self.root, background='black', width=600, height=100)

        canvas_panel.pack(side='top')
        button_panel.pack(side='bottom')

        self._create_buttons(button_panel)
        self._create_canvas(canvas_panel)

        self.canvas.bind("<Button-1>", self.save_posn)
        self.canvas.bind("<B1-Motion>", self.add_line)

    def _create_buttons(self, parent):
        erase_button = tk.Button(parent, text="Erase", width=20, height=1, command=self.erase_button)
        save_png = tk.Button(parent, text="Save as PNG", width=20, height=1, command=self.save_as_png)

        erase_button.grid(row=0, column=0)
        save_png.grid(row=0, column=1)

    def _create_canvas(self, parent):
        temp = tk.Frame(parent, width=600, height=20, background='black')
        temp.grid(row=0)
        temp = tk.Frame(parent, width=20, height=560, background='black')
        temp.grid(row=1, column=0)
        self.canvas = tk.Canvas(parent, width=552, height=552, background='white')
        self.canvas.grid(row=1, column=1)
        temp = tk.Frame(parent, width=20, height=560, background='black')
        temp.grid(row=1, column=2)
        temp = tk.Frame(parent, width=600, height=20, background='black')
        temp.grid(row=2)

        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

    def save_posn(self, event):
        self.lastx, self.lasty = event.x, event.y

    def add_line(self, event):
        self.canvas.create_line((self.lastx, self.lasty, event.x, event.y), fill='black', width=20, tags='line')
        self.save_posn(event)

    def erase_button(self):
        self.canvas.delete('line')

    def save_as_png(self):
        self.canvas.postscript(file='temp.eps')
        img_eps = Image.open('temp.eps')
        img = img_eps.convert('LA')
        img.save('output.png')
        os.remove('temp.eps')

    @staticmethod
    def change_dim(height, width):
        image_file = 'output.png'
        new_grid = np.ndarray((height, width))
        total_pixels = height * width

        try:
            Image.open(image_file)
        except FileNotFoundError:
            raise Exception()

        with Image.open(image_file, 'r') as image:
            image_pixel_tuple = list(image.getdata())  # turns image pixels into a list
            image_pixel_value = [i[0] for i in image_pixel_tuple]
            image_pixel_value_0_to_1 = [1-i/255 for i in image_pixel_value]
            pix_list = np.asarray(image_pixel_value_0_to_1).reshape((image.height, image.width))

        for h in range(560):
            for w in range(560):
                new_grid[(h//20, w//20)] += pix_list[(h, w)]

        new_grid = new_grid/400
        new_grid[new_grid < 0] = 0
        new_grid = np.asarray(new_grid).reshape((784, 1))

        return new_grid


if __name__ == '__main__':
    canvas = App()
    canvas.root.mainloop()

    input = App.change_dim(28, 28)
    nn1 = NeuralNetwork((784, 20, 10))

    with np.load('networkparams_20node.npz', allow_pickle=True) as params:
        weights = params['weights']
        biases = params['biases']

    nn1.set_weights(weights)
    nn1.set_biases(biases)

    print(f"Predicted: {np.argmax(nn1.predict(input))}")



