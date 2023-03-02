# Praktikum on 3D Vision - 3D Registration

## Project's overview
Our project is focused on the task of finding point correspondences of rigid 3D objects to a canonical model. To achieve this goal, we conducted a thorough review of related work in the field and explored various possibilities. We experimented with template matching, but found its disadvantages when compared to deep learning methods. After careful consideration, we chose SurfEmb's architecture to perform our task.

To enable dense correspondence, we applied artificial texture to our models. We created our own datasets consisting of five patterns and one non-textured pattern. We then used our method to perform inference on the new dataset to qualitatively show the optimisation. Finally, we evaluated our results both quantitatively and qualitatively, to show the improvement in performance when we applied textures. Our work shows that deep learning methods like SurfEmb's architecture can significantly improve the accuracy of finding point correspondences in 3D objects, especially when used with artificial texture. 

## Install requirements

Download surfemb:

```shell
$ git clone https://github.com/emreds/3D-Registration.git
$ cd surfemb
```

[Install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
, create a new environment, *surfbase*, and activate it:

```shell
$ conda create --name surfbase python=3.8
$ pip install -r requirements.txt
$ conda activate surfbase
```

## Prepare Datasets
<p align="center">
<img src="/src/texture1.png" alt="Texture 1" width="150" /> <img src="/src/texture2.png" alt="Texture 2" width="150" />
<img src="/src/texture3.png" alt="Texture 3" width="150" /> <img src="/src/texture4.png" alt="Texture 4" width="150" /> <img src="/src/texture5.png" alt="Texture 15" width="150" /> 
</p>

Here's a list of download links for the patterns displayed above, in the order shown:

[Pattern 1](https://drive.google.com/file/d/1qyEU6JLRu_F-yJenvICj-Dw6m439BoBl/view?usp=share_link),
[Pattern 2](https://drive.google.com/file/d/1O6M3pn_LnLwZcPaBHkRI5bgTNc2azQYi/view?usp=share_link),
[Pattern 3](https://drive.google.com/file/d/1t9XxAhy8RqP2EGjwcgXHQjHRAtFRnQEW/view?usp=share_link),
[Pattern 4](https://drive.google.com/file/d/16mltzm_U-t9DVue9Cy9bs7rna-gYOZi5/view?usp=share_link),
[Pattern 5](https://drive.google.com/file/d/1Y7Pxujt-MMd5WUqK632Ief3CFP_r3HIy/view?usp=share_link),
[No Texture](https://drive.google.com/file/d/1Qj-3BxYzInYxK1JtLhvgzssxfRZWi-l2/view?usp=share_link)

Original datasets can be downloaded through the following link in accordance with the BOP's format: [Original Bop](https://bop.felk.cvut.cz/datasets/).

Extract the datasets under ```data/bop``` (or make a symbolic link).

The following images display the rendered objects with applied and selected patterns:
<img src="/src/p1.png" width="250" /> <img src="/src/p2.png" width="250" /> <img src="/src/p3.png" width="250" />

## Training
To observe differences, train a model using varying numbers of epochs.
Configure the following settings in the training script:
```shell
import wandb
wandb.log({'epoch': num})
```

The value of ```num``` can be selected from the following options: ```5```, ```10```, or ```20```.

| number of epochs | convergent speed | perceptibility of differences |
| ---------------- | ---------------- | ----------------------------- |
| 20 epochs        | convergence      | imperceptible                 |
| 10 epochs        | near convergence | barely noticeable             |
| 5 epochs         | no convergence   | obvious                       |

The following figure illustrates this concept:
<p align="center">
<img src="/src/5_10_20.png" width="600" />
</p>

```shell
$ python -m surfemb.scripts.train [dataset] --gpus [gpu ids]
```

For example, to train a model on *T-LESS-Nonetextured* on *cuda:0*

```shell
$ python -m surfemb.scripts.train tlessnonetextured --gpus 0
```

## Inference data

We use the detections from [CosyPose's](https://github.com/ylabbe/cosypose) MaskRCNN models, and sample surface points
evenly for inference.  
For ease of use, this data can be downloaded and extracted as follows:

```shell
$ wget https://github.com/rasmushaugaard/surfemb/releases/download/v0.0.1/inference_data.zip
$ unzip inference_data.zip
```

**OR**

<details>
<summary>Extract detections and sample surface points</summary>

### Surface samples

First, flip the normals of ITODD object 18, which is inside out. 

Then remove invisible parts of the objects

```shell
$ python -m surfemb.scripts.misc.surface_samples_remesh_visible [dataset] 
```

sample points evenly from the mesh surface

```shell
$ python -m surfemb.scripts.misc.surface_samples_sample_even [dataset] 
```

and recover the normals for the sampled points.

```shell
$ python -m surfemb.scripts.misc.surface_samples_recover_normals [dataset] 
```

### Detection results

Download CosyPose in the same directory as SurfEmb was downloaded in, install CosyPose and follow their guide to
download their BOP-trained detection results. Then:

```shell
$ python -m surfemb.scripts.misc.load_detection_results [dataset]
```

</details>

## Inference inspection

To see pose estimation examples on the training images run

```shell
$ python -m surfemb.scripts.infer_debug [model_path] --device [device]
```
*[device]* could for example be *cuda:0* or *cpu*. 
Here is an example of inference on a training image:
<p align="center">
<img src="/src/inf_insp.png" width="600" />
</p>
By performing inference inspection, we can visually observe how applying different textures results in varied correspondence accuracy.

Add ```--real``` to use the test images with simulated crops based on the ground truth poses, or further
add ```--detections``` to use the CosyPose detections.

## Inference for BOP evaluation

### Notice
If you would like to create your own inference data, please adjust the bop_challenge file accordingly for inference/evaluation purposes.

Inference is run on the (real) test images with CosyPose detections:

```shell
$ python -m surfemb.scripts.infer [model_path] --device [device]
```

Pose estimation results are saved to ```data/results```.  
To obtain results with depth (requires running normal inference first), run

```shell
$ python -m surfemb.scripts.infer_refine_depth [model_path] --device [device]
```

The results can be formatted for BOP evaluation using

```shell
$ python -m surfemb.scripts.misc.format_results_for_eval [poses_path]
```

Either upload the formatted results to the BOP Challenge website or evaluate using
the [BOP toolkit](https://github.com/thodan/bop_toolkit).

## Extra

Custom dataset:
Format the dataset as a BOP dataset and put it in *data/bop*.

# Credits
This project for a practical course `3D Computer Vision` offered by, Technical University of Munich.
Team members are: 
- Emre Demir
- Wenzhao Tang 
- Julian Darth  