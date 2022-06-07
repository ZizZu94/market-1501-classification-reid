# Market-1501 Classification and ReIdentification using ResNet50

In this assignment, we explored the use of neural
networks to solve two common computer vision tasks using
the `PyTorch` framework. We had given a video-surveillance
dataset containing images of multiple persons each of which
was captured multiple times by different cameras along with a
set of annotations that specify attributes of each person such
as age, gender and clothing. The first part of the assignment
was consisted in building a multi-class classifier to predict
such attributes for each image. In the second part of the assignment,
were asked to solve a person re-identification
problem where a query image of a person was given and all
the images of the same person must be retrieved from a collection
of images.

## Dataset

The dataset used for this project is a version
of the Market-1501 person re-identification dataset.

![alt text](https://github.com/ZizZu94/market-1501-classification-reid/blob/main/img/dataset.png?raw=true)

Each image in the dataset corresponds to a tight crop of a pedestrian
and the same person appears multiple times in the dataset.
Moreover, while the differences between some persons are
marked and easy to spot, some other cases are difficult to
distinguish.

## Dependencies

Python

```
$ sudo apt-get install python3 python3-pip
```

PyTorch

```
$ pip install pytorch
# $ conda install pytorch
```

Tensorflow

```
$ pip install tensorflow
# $ conda install -c conda-forge tensorflow
```

NumPy

```
$ pip install numpy
# $ conda install numpy
```

Pandas

```
$ pip install pandas
# $ conda install pandas
```

Matplotlib

```
$ pip install matplotlib
# $ conda install matplotlib
```

## Classification task

As the first task, we built a
multi-class classification model. We split the training images into two sub folders: training and validation. Then we used
them to train our model, which can predict different attributes given as input an image of a random pedestrian taken from the test folder. The predicted attributes are the following:

![alt text](https://github.com/ZizZu94/market-1501-classification-reid/blob/main/img/dataset-2.png?raw=true)

[Classification notebook](https://github.com/ZizZu94/market-1501-classification-reid/blob/main/source-code/1_classification_dl_project.ipynb)

## ReIdentification task

The second task of our project was quite challenging, we had to build a re-identification model, which can identify all the images of the same person, despite they were taken from different angles, different locations, different scales, different zoom settings and also captured with a different type of camera. The idea is, for each image in the queries directory, we produced an ordered list of images from the test directory that we believe correspond to the same person identity depicted in the query image.

[ReIdentification notebook](https://github.com/ZizZu94/market-1501-classification-reid/blob/main/source-code/2_2_reID_dl_project.ipynb)
