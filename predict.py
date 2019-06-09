import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import cv2

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import os
import glob

target_size = (300, 300) #fixed size for InceptionV3 architecture


def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return preds[0]


def plot_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  plt.imshow(image)
  plt.axis('off')

  plt.figure()
  labels = ("Adidas", "MK","Nike","Puma","TH")
  print(preds)
  plt.barh([0, 1,2,3,4], preds, alpha=0.5)
  plt.yticks([0, 1,2,3,4], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()
  # plt.savefig('prediction.png')



if __name__=="__main__":
  # a = argparse.ArgumentParser()
  # a.add_argument("--image", help="path to image")
  # a.add_argument("--image_url", help="url to image")
  # a.add_argument("--model")
  # args = a.parse_args()

  # if args.image is None and args.image_url is None:
  #   a.print_help()
  #   sys.exit(1)
  model = load_model("brands_3.model")
  os.chdir("testing_snapit/")
  all_filenames = [i for i in glob.glob('*.{}'.format("jpg"))]
  for img_path in all_filenames:
      img = Image.open(img_path)
      print(img_path)
      preds = predict(model, img, target_size)
      plot_preds(img, preds)

  # if args.image_url is not None:
  #   response = requests.get(args.image_url)
  #   img = Image.open(BytesIO(response.content))
  #   preds = predict(model, img, target_size)
  #   plot_preds(img, preds)
