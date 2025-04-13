# Description
"""
Handwritten Signature Detection

Dataset Description:

In today’s digital world, automatic signature detection in documents is crucial for many sectors
including banking, legal, and administrative fields. This assignment challenges you to develop a
machine learning model capable of accurately detecting and localizing signatures using a
provided dataset. This dataset consists of training and testing images, each paired with precise
ground truth data to facilitate the development and evaluation of your model.

The dataset consists of four folders:
- ‘TrainImages’ folder: contains the training images (660 images)
- ‘TrainGroundTruth’ folder: contains the corresponding detection labels (660 files)
- ‘TestImages’ folder: contains the testing images (115 images)
- ‘TestGroundTruth’ folder: contains the corresponding detection labels (115 files)

Each image has a corresponding text file with the same name. The text file has 1 or more rows, each row
representing a bounding box: x1, y1, x2, y2

The notebook includes training, evaluation, a plot of training/validation loss, and a sample test prediction.
"""

