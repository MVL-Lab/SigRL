## Setup

```bash
pip install -r requirements.txt
```

## Preparation

1. Download pretrained VLP(ViT-B/16) model from [OpenAI CLIP](https://github.com/openai/CLIP).
2. Download images of NUS-WIDE dataset  from [NUS-WIDE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html).
3. Download annotations following the [BiAM](https://github.com/akshitac8/BiAM) from [here](https://drive.google.com/drive/folders/1jvJ0FnO_bs3HJeYrEJu7IcuilgBipasA?usp=sharing).
4. Download other files from [here](https://drive.google.com/drive/folders/1kTb83_p92fM04OAkGyiHypOgwtxc4wVa?usp=sharing).

The organization of the dataset directory is shown as follows.

```bash
NUS-WIDE
  ├── features
  ├── Flickr
  ├── Concepts81.txt
  ├── Concepts925.txt
  ├── img_names.pkl
  ├── label_emb.pt
  └── test_img_names.pkl
```

## Training on NUS-WIDE

```bash
python3 train_nus.py 
```

## Testing on NUS-WIDE

```bash
python3 engine_nus.py
```

## Acknowledgement

We would like to thank [BiAM](https://github.com/akshitac8/BiAM) and [timm](https://github.com/rwightman/pytorch-image-models) for the codebase.

Many thanks to the author of 
[Open-Vocabulary Multi-Label Classification via Multi-modal Knowledge Transfer](https://github.com/sunanhe/MKT).
Our scripts are highly based on their scripts.

