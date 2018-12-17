# FEAT
The code repository for "Learning Embedding Adaptation for Few-Shot Learning" in PyTorch

## Few-Shot Learning via Transformer

Few-shot learning methods address this challenge by learning an instance embedding function from seen classes, and apply the function to instances from unseen classes with limited labels. This style of transfer learning is task-agnostic: the embedding function is not learned optimally discriminative with respect to the unseen classes, where discerning among them is the target task. In this work, we propose a novel approach to adapt the embedding model to the target classification task, yielding embeddings that are task-specific and are discriminative. To this end, we employ a type of self-attention mechanism called Transformer to transform the embeddings from task-agnostic to task-specific by focusing on relating instances from the test instances to the training instances in both seen and unseen classes.

![Few-Shot Learning via Transformer](imgs/teaser.pdf)

## MiniImageNet Dataset

The MiniImageNet dataset is a subset of the ImageNet that includes a total number of 100 classes and 600 examples per class. We follow the [@previous setup](https://github.com/twitter/meta-learning-lstm), and use 64 classes as SEEN categories, 16 and 20 as two sets of UNSEEN categories for model validation and evaluation respectively.

### Dataset splits

We implemented the Vynials splitting method as in [[Matching Networks for One Shot Learning](https://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning)]. That sould be the same method used in the paper (in fact I download the split files from the "offical" [repo](https://github.com/jakesnell/prototypical-networks/tree/master/data/omniglot/splits/vinyals)). We then apply the same rotations there described. In this way we should be able to compare results obtained by running this code with results described in the reference paper.

## Training

To train the Protonet on this task, cd into this repo's `src` root folder and execute:

    $ python train.py


The script takes the following command line options:

- `dataset_root`: the root directory where tha dataset is stored, default to `'../dataset'`

- `nepochs`: number of epochs to train for, default to `100`

- `learning_rate`: learning rate for the model, default to `0.001`

- `lr_scheduler_step`: StepLR learning rate scheduler step, default to `20`

- `lr_scheduler_gamma`: StepLR learning rate scheduler gamma, default to `0.5`

- `iterations`: number of episodes per epoch. default to `100`

- `classes_per_it_tr`: number of random classes per episode for training. default to `60`

- `num_support_tr`: number of samples per class to use as support for training. default to `5`

- `num_query_tr`: nnumber of samples per class to use as query for training. default to `5`

- `classes_per_it_val`: number of random classes per episode for validation. default to `5`

- `num_support_val`: number of samples per class to use as support for validation. default to `5`

- `num_query_val`: number of samples per class to use as query for validation. default to `15`

- `manual_seed`: input for the manual seeds initializations, default to `7`

- `cuda`: enables cuda (store `True`)

Running the command without arguments will train the models with the default hyperparamters values (producing results shown above).


## Helpful links

 - http://pytorch.org/docs/master/data.html: Official PyTroch documentation about Dataset classes, Dataloaders and Samplers

## .bib citation
cite the paper as follows (copied-pasted it from arxiv for you):

    @article{DBLP:journals/corr/SnellSZ17,
      author    = {Jake Snell and
                   Kevin Swersky and
                   Richard S. Zemel},
      title     = {Prototypical Networks for Few-shot Learning},
      journal   = {CoRR},
      volume    = {abs/1703.05175},
      year      = {2017},
      url       = {http://arxiv.org/abs/1703.05175},
      archivePrefix = {arXiv},
      eprint    = {1703.05175},
      timestamp = {Wed, 07 Jun 2017 14:41:38 +0200},
      biburl    = {http://dblp.org/rec/bib/journals/corr/SnellSZ17},
      bibsource = {dblp computer science bibliography, http://dblp.org}
    }


## Acknowledgment
We thank following repos providing helpful components/functions in our work.
