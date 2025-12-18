# UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations

<!-- ### [Imitating Human Videos via Cross-Embodiment Skill Representations](https://uniskill.github.io)    -->
[Hanjung Kim*](https://kimhanjung.github.io/), [Jaehyun Kang*](https://kang-jaehyun.github.io/), [Hyolim Kang](https://sites.google.com/view/hyolim), [Meedeum Cho](https://chomeed.github.io), [Seon Joo Kim](https://www.ciplab.kr), [Youngwoon Lee](https://youngwoon.github.io/)

[[`arXiv`](https://arxiv.org/abs/2505.08787)][[`Project`](https://kimhanjung.github.io/UniSkill/)][[`Dataset`](https://huggingface.co/datasets/HanjungKim/UniSkill)][[`BibTeX`](#citing-uniskill)]

![](https://kimhanjung.github.io/images/uniskill.png)

### Features
* A universal skill represenation (**UniSkill**) that enables cross-embodiment imitation by learning from large-scale video datasets, without requiring scene-aligned data between embodiments.
* UniSkill supports skill transfer across agents with different morphologies, incluiding human-to-robot and robot-to-robot adaptation.
* UniSkill leverages large-scale video pretraining to capture shared interaction dynamics, enhancing adaptability to novel settings.

## Todos
We will be releasing all the following contents:
- [x] FSD & ISD training code
- [x] FSD & ISD checkpoint
- [x] Real-world Dataset
- [x] Skill-Conditioned Policy training & inference code
- [x] Skill-Conditioned Policy checkpoint
- [x] Skill-extraction code

## Installation

- Linux or macOS with Python ≥ 3.10
- PyTorch ≥ 2.3 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. 
- `pip install -r requirements.txt`

## Getting Started

For dataset preparation instructions, refer to [Preparing Datasets for UniSkill](diffusion/dataset/README.md).

We provide the script [train_uniskill.py](./diffusion/train_uniskill.py) for training UniSkill.

### Training UniSkill (FSD & ISD)

To train the **Forward Skill Dynamics (FSD)** and **Inverse Skill Dynamics (ISD)** models of UniSkill, first set up your custom datasets. Once your dataset is ready, run the following command:

``` bash
cd diffusion

python train_uniskill.py \
    --do_classifier_free_guidance \
    --pretrained_model_name_or_path timbrooks/instruct-pix2pix \
    --allow_tf32 \
    --train_batch_size 32 \
    --dataset_name {Your Dataset} \
    --output_dir {output_dir} \
    --num_train_epochs 50 \
    --report_name {report_name} \
    --learning_rate 1e-4 \
    --validation_steps 50
```

### Multi-GPU Training

For multi-GPU training, first modify the configuration file [hf.yaml](./hf.yaml) as needed. Then, run the following command:

``` bash
accelerate launch --config_file hf.yaml diffusion/train_uniskill.py \
    --do_classifier_free_guidance \
    --pretrained_model_name_or_path timbrooks/instruct-pix2pix \
    --allow_tf32 \
    --train_batch_size 32 \
    --dataset_name {Your Dataset} \
    --output_dir {output_dir} \
    --num_train_epochs 50 \
    --report_name {report_name} \
    --learning_rate 1e-4 \
    --validation_steps 50
```

Make sure to replace `{Your Dataset}`, `{output_dir}`, and `{report_name}` with the appropriate values.

### Training UniSkill (Universal Skill-Conditioned Policy)

For policy training, please refer to the [Uniskill-policy](https://github.com/kang-jaehyun/UniSkill-Policy/) repository.

To train the policy effectively, we pre-extract skills from the data. We provide a skill extraction script, [extract_skill.py](./extract_skill.py), which you can use as a reference.

Alternatively, you can use a pre-trained IDM directly within your policy to extract skills on the fly.

## Download

- [FSD & ISD](https://huggingface.co/HanjungKim/UniSkill)

- [Skill-conditioned Diffuion Policy](https://github.com/kimhanjung)

## <a name="CitingUniSkill"></a>Citing UniSkill

```BibTeX
@inproceedings{
    kim2025uniskill,
    title={UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations},
    author={Hanjung Kim and Jaehyun Kang and Hyolim Kang and Meedeum Cho and Seon Joo Kim and Youngwoon Lee},
    booktitle={9th Annual Conference on Robot Learning},
    year={2025}
}
```