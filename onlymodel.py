import os.path
from PIL import ImageGrab, Image
from ultralytics import YOLO
import torch


def screen_monitor():
    image = ImageGrab.grab()
    return image


def connecting_model(path_to_model):
    global model, load_model

    model = YOLO(path_to_model, task='classify')

    torch.cuda.set_device(0)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model.to(device)


def classify_nsfw():

    connecting_model(os.path.normpath("./512old.pt"))

    while True:
        image = screen_monitor()

        output = model.predict(image, verbose=False)[0]

        probs = torch.sort(output.probs.data)

        out = {output.names[idx]: f"{prob:.3f}" for idx, prob in zip(probs.indices.tolist(), probs.values.tolist())}

        print(f"{out}")


classify_nsfw()
