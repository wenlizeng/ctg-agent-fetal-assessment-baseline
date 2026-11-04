# Released training dataset

For Task1 and Task4 reference FHR labels, see the [.fetal binary format and Python loading guide](FETAL_FORMAT.md).

This directory contains the public challenge training inputs, labels and manifests using the following layout:

```text
data/
├── task1/
│   ├── data/train/wav/
│   ├── labels/train/fetal/
│   └── manifests/train.csv
├── task2/
│   ├── data/train/t2_00001.csv
│   └── manifests/train_windows.csv
├── task3/
│   ├── data/train/
│   └── manifests/train.csv
└── task4/
    ├── data/train/wav/
    ├── labels/task1_fetal/train/
    ├── labels/task2_segmentation/train/
    ├── labels/task3_classification/train.csv
    └── manifests/train.csv
```

Manifest paths are relative to this `data/` directory. The participant training data have been streamlined. All published rows belong to the training set.

Only training inputs, training labels and training manifests belong here. Validation and test inputs and labels remain private.
