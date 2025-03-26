
##  Installation
1. Create a Conda environment for the code:
```
conda create --name SPML python=3.8.8
```
2. Activate the environment:
```
conda activate SPML
```
3. Install the dependencies:
```
pip install -r requirements.txt
```

## Preparing Datasets
### Downloading Data
#### PASCAL VOC

1. Run the following commands:

```
cd {PATH-TO-THIS-CODE}/data/pascal
curl http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar --output pascal_raw.tar
tar -xf pascal_raw.tar
rm pascal_raw.tar
```

#### MS-COCO

1. Run the following commands:

```
cd {PATH-TO-THIS-CODE}/data/coco
curl http://images.cocodataset.org/annotations/annotations_trainval2014.zip --output coco_annotations.zip
curl http://images.cocodataset.org/zips/train2014.zip --output coco_train_raw.zip
curl http://images.cocodataset.org/zips/val2014.zip --output coco_val_raw.zip
unzip -q coco_annotations.zip
unzip -q coco_train_raw.zip
unzip -q coco_val_raw.zip
rm coco_annotations.zip
rm coco_train_raw.zip
rm coco_val_raw.zip
```

### Formatting Data
For PASCAL VOC  and MS-COCO, use Python code to format data:
```
cd {PATH-TO-THIS-CODE}
python preproc/format_pascal.py
python preproc/format_coco.py
```


### Generating Single Positive Annotations
In the last step, run `generate_observed_labels.py` to yield single positive annotations from full annotations of each dataset:
```
python preproc/generate_observed_labels.py --dataset {DATASET}
```
`{DATASET}` should be replaced by `pascal` or `coco`.

## Training and Evaluation
Run `main_clip.py` to train and evaluate a model:
```
python main_clip.py 
```
## Results:

## Acknowledgement:
Many thanks to the authors of [Vision-Language Pseudo-Labels for Single-Positive Multi-Label Learning](https://github.com/mvrl/VLPL).[single-positive-multi-label](https://github.com/elijahcole/single-positive-multi-label) and [SPML-AckTheUnknown
](https://github.com/Correr-Zhou/SPML-AckTheUnknown).

Our scripts are highly based on their scripts.
