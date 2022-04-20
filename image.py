# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 Christian Kreuzberger
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse

import cv2
from object_detector import ObjectDetector, ObjectDetectorOptions
import utils


def run(model: str, source_image_file: str, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Run inference on image stored in file

  Args:
    model: Name of the TFLite object detection model.
    source_image_file: Path to the image that should be analyzed
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Initialize the object detection model
  options = ObjectDetectorOptions(
      num_threads=num_threads,
      score_threshold=0.5,
      max_results=5,
      enable_edgetpu=enable_edgetpu)
  detector = ObjectDetector(model_path=model, options=options)

  # Grab frame from file
  print(source_image_file)

  image = cv2.imread(source_image_file, cv2.IMREAD_COLOR)
  print(image.shape)
  cv2.imshow('object_detector', image)

  # from PIL import Image
  # image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

  # Run object detection estimation using the model.
  detections = detector.detect(image)

  print("Detected {} things:".format(len(detections)))

  for dt in detections:
    print(dt.categories)

  # Draw keypoints and edges on input image
  image = utils.visualize(image, detections)

  cv2.imshow('object_detector', image)

  print("Press escape to exit")
  try:
    cv2.waitKey(0)
  except KeyboardInterrupt:
    pass
  finally:
    cv2.destroyAllWindows()
  print("done!")


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite') 
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  parser.add_argument(
    'sourceImage',
    help='Image to analyze',
    action='store',
    default=False
  )
  args = parser.parse_args()

  run(args.model, args.sourceImage,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()
