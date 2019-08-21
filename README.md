# Deep Fundamental Matrix Estimation

This repository contains code for training a network as described in our [paper](http://vladlen.info/papers/deep-fundamental.pdf):

>Deep Fundamental Matrix Estimation  
René Ranftl and Vladlen Koltun  
European Conference on Computer Vision (ECCV), 2018

The trained network can be used to compute a fundamental matrix based on noisy point correspondences.

## Setup


Create and activate conda environment:

    ```shell
    conda env create -f environment.yml
    conda activate dfe
    ```

## Train

An example training with a colmap dataset can be performed with the following command:

```shell
python train.py --dataset [dataset folders]
```

Dataset folders are assumed to be in Colmap format. Multple folders can be listed.

An example can be found here: [Family](https://drive.google.com/open?id=1b4lb5La3dzn_D87sy-fpgCAbEnGRSLrL).

To get help:

```shell
python train.py -h
```

## Test

```shell
python test.py --dataset [dataset folders] --model [path to model]
```

A pre-trained model that was trained on Tanks and Temples can be found in the models folder.


## Citation

Please cite our paper if you use this code in your research:

```bibtex
@InProceedings{Ranftl2018,
    author = {Ranftl, Rene and Koltun, Vladlen},
    title = {Deep Fundamental Matrix Estimation},
    booktitle = {The European Conference on Computer Vision (ECCV)},
    year = {2018}
}
```

## License

MIT License
