# Released training dataset

This directory contains the public challenge training inputs, labels and manifests using the following layout:

```text
data/
├── task1/
│   ├── data/train/wav/
│   ├── labels/train/fetal/
│   └── manifests/train.csv
├── task2/
│   ├── data/train/<database>__<source-file>.csv
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

Paths stored in the official manifests are relative to this `data/` directory. Task2 is released as one flat training set. Its filenames use `<database>__<source-file>.csv` only to avoid source-record name collisions; the database prefix is provenance metadata, not a separate challenge subset.

The manifest columns are limited to sample IDs, data paths, training labels, window offsets, and grouping IDs. `source_record_id` identifies windows from the same original recording; Task3 additionally supplies an anonymous `patient_id` so all windows from one patient can be kept together in future training splits. All published rows belong to the training set.

Only training inputs, training labels and training manifests belong here. Validation and test inputs and labels remain private.
