import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

digits = load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = SVC()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))
print("Accuracy:", accuracy)

root = tk.Tk()
root.title("Digit Recognizer")

canvas = tk.Canvas(
    root,
    width=280,
    height=280,
    bg="black",
    bd=0,
    highlightthickness=0
)

canvas.pack()

image = Image.new("L", (280, 280), 0)
draw_image = ImageDraw.Draw(image)

last_x = None
last_y = None

def start_draw(event):
    global last_x, last_y

    last_x = event.x
    last_y = event.y

    canvas.create_oval(
        event.x - 10,
        event.y - 10,
        event.x + 10,
        event.y + 10,
        fill="white",
        outline="white"
    )

    draw_image.ellipse(
        (
            event.x - 10,
            event.y - 10,
            event.x + 10,
            event.y + 10
        ),
        fill=255
    )

def draw(event):
    global last_x, last_y

    if last_x is None or last_y is None:
        return
    canvas.create_line(
        last_x,
        last_y,
        event.x,
        event.y,
        fill="white",
        width=20,
        capstyle=tk.ROUND
    )

    draw_image.line(
        (
            last_x,
            last_y,
            event.x,
            event.y
        ),
        fill=255,
        width=20
    )

    draw_image.ellipse(
        (
            event.x - 10,
            event.y - 10,
            event.x + 10,
            event.y + 10
        ),
        fill=255
    )

    last_x = event.x
    last_y = event.y

def stop_draw(event):
    global last_x, last_y

    last_x = None
    last_y = None

canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_draw)

def predict():

    array = np.array(image)
    coords = np.where(array > 20)

    if len(coords[0]) == 0:
        print("Draw a digit first!")
        return
    x1 = coords[1].min()
    x2 = coords[1].max()
    y1 = coords[0].min()
    y2 = coords[0].max()

    print("Bounding box:", x2 - x1 + 1, "x", y2 - y1 + 1)
    print("Coordinates:", x1, y1, x2, y2)

    cropped = image.crop(
        (
            x1,
            y1,
            x2 + 1,
            y2 + 1
        )
    )

    width = cropped.width
    height = cropped.height

    size = max(width, height)
    square = Image.new(
        "L",
        (size, size),
        0
    )
    offset_x = (size - width) // 2
    offset_y = (size - height) // 2

    square.paste(
        cropped,
        (offset_x, offset_y)
    )

    resized = square.resize(
        (8, 8),
        Image.Resampling.LANCZOS
    )

    array = np.array(resized)

    array = array / 255 * 16

    plt.imshow(
        array,
        cmap="gray",
        interpolation="nearest"
    )

    plt.xticks(range(8))
    plt.yticks(range(8))
    plt.grid()
    plt.show()
    array = array.reshape(1, -1)
    print("Shape:", array.shape)
    prediction = model.predict(array)
    print("Prediction:", prediction[0])


def clear():
    global image, draw_image
    canvas.delete("all")
    image = Image.new(
        "L",
        (280, 280),
        0
    )
    draw_image = ImageDraw.Draw(image)

button_frame = tk.Frame(root)
button_frame.pack()

predict_button = tk.Button(
    button_frame,
    text="Predict",
    command=predict
)
predict_button.pack(side=tk.LEFT)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear
)

clear_button.pack(side=tk.LEFT)
root.mainloop()