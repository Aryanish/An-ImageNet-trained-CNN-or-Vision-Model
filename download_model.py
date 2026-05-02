import requests
import os

model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

if not os.path.exists("mobilenetv2.onnx"):
    print("Downloading model...")
    r = requests.get(model_url, allow_redirects=True)
    with open("mobilenetv2.onnx", "wb") as f:
        f.write(r.content)

if not os.path.exists("imagenet_classes.txt"):
    print("Downloading labels...")
    r = requests.get(labels_url)
    with open("imagenet_classes.txt", "w") as f:
        f.write(r.text)

print("Done.")
