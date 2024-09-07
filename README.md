# NSFW-Model
This repository was created for a model that was trained to classify nsfw content (don't judge strictly, I'm new to creating models)

This link shows [google colab](https://colab.research.google.com/drive/1ViMkFhYHSUeLEfSiLpmhCB66TJgxoE6w?usp=sharing) in which the model was trained. The model is based on YOLOv8. Training parameters ```model.train(data="/content/dataset", epochs=50, imgsz=512, batch=32, pretrained=True, plots=True, device='cuda', verbose=True)```. The resolution was chosen 512px as it was optimal for work on the processor during testing

Data for training you can find in this [repository](https://github.com/Serfetto/NSFW-Data) or in the colab itself

You can find the trained model at this [link](https://drive.google.com/file/d/1Vl5pY9ERFb-L5eF73Qt9Dumkmlrgykyk/view?usp=sharing)
