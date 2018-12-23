# Dataset Pre-processing

## Generic Pre-processing
After downloading the dataset, please create a new folder named "images" under the folder "miniimagenet" or "cub", and put all images in this folder. The provided data loader will read images from the "images" folder by default. Of course, it is also OK to change the read path. For example, for the miniimagenet dataset, please change the line 10 of "./feat/dataloader/mini_imagenet.py" as the path of the downloaded images.

We assume all the images in the folder are the original ones (except a crop based on bounding boxes for CUB, see details below), and the data loader will do transformations on those raw images, such as resize and normalization. All the images will be resized as 84x84 for ConNet backbone, and 80x80 for ResNet backbone.

### MiniImageNet
The MiniImageNet dataset is a subset of the ImageNet that includes a total number of 100 classes and 600 examples per class. We use the [Ravi's split](https://github.com/twitter/meta-learning-lstm), and use 64 classes as SEEN categories, 16 and 20 as two sets of UNSEEN categories for model validation and evaluation respectively. To download this dataset, please email [Sachin Ravi](http://www.cs.princeton.edu/~sachinr/) for further details and instructions.

### CUB
[Caltech-UCSD Birds (CUB) 200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) is initially designed for fine-grained classification. It contains in total 11,788 images of birds over 200 species. On CUB, we follow the [previous setting](https://arxiv.org/abs/1707.02610) randomly sampling 100 species as SEEN classes, another two 50 species are used as two UNSEEN sets. Since there is no public class split for CUB, we use our own split as saved in the "CUB" folder. We crop all images with given bounding box before training. We only test CUB with ConvNet backbone in our work.
