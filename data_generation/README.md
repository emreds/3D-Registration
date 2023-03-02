# Dataset Generation

The dataset in the context of the 3D-Registration project was created in two different ways:

1. By hand a model of a CocaCola can was put into the a pre-existing blender pipeline to generate different views
2. A tool/framework called [BlenderProc](https://github.com/DLR-RM/BlenderProc) was used

## Preface

The goal was to generate a synthetic dataset of one or multiple objects with different patterns. The pattern should be used to mimic a texture that could be applied to real world objects using UV-Paint. By comparing the differences between the original (untextured) objects with the textured ones we can infer possible advancements by using certain textures. More about this can be read in the main README file, in the root of this repository. 

### Patterns
in [Patterns](./Patterns/) an example set of textures that were used within our project can be found. 

## 1. Data Generation by Blender

The file for this ([1_synthetic_dataset.py](./1-blender/1_synthetic_dataset.py)) procedure was provided by the advisors and only slightly adjusted. Some of the adjustments include the output of some correspondences of 3D points that are visible in one scene to the UV map. 
![img](./1-blender/example_output/img.png)
To draw the correspondeces the script [draw_correspondences](./1-blender/draw_correspondences.py) was used.

This script is limited to the number of views and does not include any background, thus it might not be the perfect solution for simulating a real dataset, but it is a good starting point. 

## 2. BlenderProc

BlenderProc provides a lot more features that can be used to render realistic datasets.
As a base of our dataset we were using the [T-Less](http://cmp.felk.cvut.cz/t-less/download.html) dataset. It provides 30 industry relevant CAD Models. Many of the are not symmetric which further fitted our use case. All models are textureless, so we could easyly apply our texture. 

To get started with BlenderProc we highly recommend to read their README, it provides all necessary information in a very short fashion.
It also includes certain scripts to output the data in the BOP format which we were using, so only little needed to be modified there. 
Within their repository: `examples/datasets/bop_object_on_surface_sampling` was used as a base script for our use case. The final version can be found [here](./2-blenderproc/generate_dataset.py)

### BlenderProc Instructions

Specifically for the bop script we used, we recommend to read the instruction of the respective base of that script --> [here](https://github.com/DLR-RM/BlenderProc/tree/main/examples/datasets/bop_object_on_surface_sampling)

#### Install

```bash
pip install blenderproc
blenderproc download cc_textures resources
```

#### Usage
General usage as follows:

```bash
belnderproc run python_script.py
```

Concrete usage of our script
```bash
blenderproc run generate_dataset.py ./path/to/data/[dataset_name] dataset_name resources/ output/path object_id pattern_id
```

To generate the final dataset on the cluster we used a separate run script that generated `5000` images per object and per pattern, i.e. each object + pattern combination is rendered in 25 different scenes. With our 6 patterns (5 + 1 withot textures) and 3 objects this results in a dataset with a total of `90,000` images.
The run script is included [here](./2-blenderproc/run.sh).