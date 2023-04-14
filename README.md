# Disagreement Attention and Alternating Deep Supervision

Official Pytorch implementation utilised on the paper: Disagreement attention: Let us agree to disagree on computed tomography segmentation

In general terms the application contains:

1. The **Standard attention gate** (AttentionBlock) and novel **Disagreement-based attention modules** (PureDABlock, EmbeddedDABlock and MixedEmbeddedDABlock).

2. **XAttentionUNet**: A modified Attention UNet that receives the attention class to be employed as an argument.

3. **XAttentionUNet_ADSV**: A modified XAttentionUNet that employs the proposed Alternating Deep Supervision.

4. UNet2D, UNet3D, Attention Unet, UNet with grid attention.

5. The dataset processors and model managers have been moved to [github.com/giussepi/gtorch_utils](https://github.com/giussepi/gtorch_utils).

## Installation

1. Clone this repository

2. [OPTIONAL] Create your virtual enviroment and activate it

3. Install [OpenSlide](https://openslide.org/download/)

4. Install the requirements

   `pip install -r requirements`

5. Install pytorch 1.10.0 following the instructions provided on the page [pytorch.org/get-started/previous-versions/#v1100](https://pytorch.org/get-started/previous-versions/#v1100)

6. Make a copy of the configuration file, review it thoroughly and update it properly (especially `PROJECT_PATH`, `CT82_SAVING_PATH` and `LITS17_SAVING_PATH`)

   `cp settings.py.template settings.py`


## Usage

The main rule is running everything from the `main.py`

### Running tests

1. Make `get_test_datasets.sh` executable and download the testing datasets

	``` shell
	chmod +x get_test_datasets.sh
	./get_test_datasets.sh
	```

2. Make `run_tests.sh` executable and run it:

	```shell
	chmod +x run_tests.sh
	./run_tests.sh
	```

### Running TensorBoard

1. Make `run_tensorboard.sh` executable:

	`chmod +x run_tensorboard.sh`

2. Execute it

	`./run_tensorboard.sh`


### Debugging

Just open your `settings.py` and set `DEBUG = True`. This will set the log level to debug and your dataloader will not use workers so you can use `pdb.set_trace()` without any problem.

## Main Features

**Note**: Always see the class or function definition to pass the correct parameters and see all available options.

### TCIA Pancreas CT-82 dataset [^1][^2][^3]

The instructions to get and prepare the dataset for processing are available at
[github.com/giussepi/gtorch_utils/blob/main/gtorch_utils/datasets/segmentation/datasets/ct82/README.md](https://github.com/giussepi/gtorch_utils/blob/main/gtorch_utils/datasets/segmentation/datasets/ct82/README.md)

The code to process the dataset is fully integrated with the configuration file. Thus, you need to locate the code to process the CT-82 dataset in the `main.py`, uncomment, and follow the provided instructions.

### LiTS17 dataset [^4]
The instructions to get and prepare the dataset for processing are available at
[github.com/giussepi/gtorch_utils/blob/main/gtorch_utils/datasets/segmentation/datasets/lits17/README.md](https://github.com/giussepi/gtorch_utils/blob/main/gtorch_utils/datasets/segmentation/datasets/lits17/README.md).

The code to process the dataset is fully integrated with the configuration file. Thus, you need to locate the code for the dataset you want to generate in the `main.py`, uncomment, and follow the provided instructions.



### Predict masks for whole CT scans and visualize them
1. Modify your model manager to support these feature using the `CT3DNIfTIMixin`. E.g.:

``` python
class CTModelMGR(CT3DNIfTIMixin, ModelMGR):
    pass
```

2. Replace your old ModelMGR by the new one and provide and initial weights

``` python
mymodel = CTModelMGR(
	...
	ini_checkpoint='<path to your best checkpoint>',
	...
)
```

3. Make the 3D mask prediction

``` python
mymodel.predict('<path to your CT folder>/CT_119.nii.gz')
```

4. Visualize all the 2D masks

``` python
id_ = '119'

mymodel.plot_2D_ct_gt_preds(
	ct_path=f'<path to your CT folder>/CT_{id_}.nii.gz',
	gt_path=f'<path to your CT folder>/label_{id_}.nii.gz',
    pred_path=f'pred_CT_{id_}.nii.gz'
)
```


### Training, Testing and Plotting on TensorBoard
Use the `ModelMGR` to train models and make predictions.

``` python
    model7 = ModelMGR(
        model=XAttentionUNet,
        model_kwargs=dict(
            n_channels=1, n_classes=1, bilinear=False, batchnorm_cls=get_batchnormxd_class(),
            init_type=UNet3InitMethod.KAIMING, data_dimensions=settings.DATA_DIMENSIONS,
            da_block_cls=intra_model.MixedEmbeddedDABlock,  # EmbeddedDABlock, PureDABlock, AttentionBlock
            dsv=True,
        ),
        # remaining configuration #############################################
        cuda=settings.CUDA,
        multigpus=settings.MULTIGPUS,
        patch_replication_callback=settings.PATCH_REPLICATION_CALLBACK,
        epochs=settings.EPOCHS,
        intrain_val=2,
        optimizer=torch.optim.Adam,
        optimizer_kwargs=dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-6),
        sanity_checks=False,
        labels_data=LiTS17OnlyLiverLabels,  # LiTS17OnlyLesionLabels,  # CT82Labels,  # LiTS17OnlyLiverLabels
        data_dimensions=settings.DATA_DIMENSIONS,
        dataset=LiTS17CropDataset,  # CT82Dataset,  # LiTS17Dataset
        dataset_kwargs={
            'train_path': settings.LITS17_TRAIN_PATH,  # settings.CT82_TRAIN_PATH,  # settings.LITS17_TRAIN_PATH
            'val_path': settings.LITS17_VAL_PATH,   # settings.CT82_VAL_PATH,  # settings.LITS17_VAL_PATH
            'test_path': settings.LITS17_TEST_PATH,  # settings.CT82_TEST_PATH,  # settings.LITS17_TEST_PATH
            'cotraining': settings.COTRAINING,
            'cache': settings.DB_CACHE,
        },
        train_dataloader_kwargs={
            'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': True, 'num_workers': settings.NUM_WORKERS,
            'pin_memory': False
        },
        testval_dataloader_kwargs={
            'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': False, 'num_workers': settings.NUM_WORKERS,
            'pin_memory': False, 'drop_last': True
        },
        lr_scheduler=torch.optim.lr_scheduler.StepLR,
        lr_scheduler_kwargs={'step_size': 250, 'gamma': 0.5},
        lr_scheduler_track=LrShedulerTrack.NO_ARGS,
        criterions=[
            # torch.nn.BCEWithLogitsLoss()
            loss_functions.BceDiceLoss(with_logits=True),
            # loss_functions.SpecificityLoss(with_logits=True),
        ],
        mask_threshold=0.5,
        metrics=settings.get_metrics(),
        metric_mode=MetricEvaluatorMode.MAX,
        earlystopping_kwargs=dict(min_delta=1e-3, patience=np.inf, metric=True),  # patience=10
        checkpoint_interval=0,
        train_eval_chkpt=False,
        last_checkpoint=True,
        ini_checkpoint='',
        dir_checkpoints=os.path.join(settings.DIR_CHECKPOINTS, 'exp1'),
        tensorboard=False,
        # TODO: there a bug that appeared once when plotting to disk after a long training
        # anyway I can always plot from the checkpoints :)
        plot_to_disk=False,
        plot_dir=settings.PLOT_DIRECTORY,
        memory_print=dict(epochs=settings.EPOCHS//2),
    )
    model7()

```

### Showing Logs Summary from a Checkpoint
Use the `ModelMGR.print_data_logger_summary` method to do it.

``` python
model = ModelMGR(<your settings>, ini_checkpoint='chkpt_X.pth.tar', dir_checkpoints=settings.DIR_CHECKPOINTS)
model.print_data_logger_summary()
```

The summary will be a table like this one

| key         | Validation   |   corresponding training value |
|-------------|--------------|--------------------------------|
| Best metric | 0.7495       |                         0.7863 |
| Min loss    | 0.2170       |                         0.1691 |
| Max LR      |              |                         0.001  |
| Min LR      |              |                         1e-07  |


## LOGGING
This application employs [logzero](https://logzero.readthedocs.io/en/latest/). Thus, some functionalities can print extra data. To enable this just open your `settings.py` and set `DEBUG = True`. By default, the log level is set to [logging.INFO](https://docs.python.org/2/library/logging.html#logging-levels).

## Reference:
You are free to utilise this program or any of its components. If so, please reference the following paper:
[Disagreement attention: Let us agree to disagree on computed tomography segmentation](https://www.sciencedirect.com/science/article/pii/S1746809423002021).


[^1]: Holger R. Roth, Amal Farag, Evrim B. Turkbey, Le Lu, Jiamin Liu, and Ronald M. Summers. (2016). Data From Pancreas-CT. The Cancer Imaging Archive. [https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU](https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU)
[^2]: Roth HR, Lu L, Farag A, Shin H-C, Liu J, Turkbey EB, Summers RM. DeepOrgan: Multi-level Deep Convolutional Networks for Automated Pancreas Segmentation. N. Navab et al. (Eds.): MICCAI 2015, Part I, LNCS 9349, pp. 556–564, 2015.  ([paper](http://arxiv.org/pdf/1506.06448.pdf))
[^3]: Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: [https://doi.org/10.1007/s10278-013-9622-7](https://doi.org/10.1007/s10278-013-9622-7)
[^4]: P. Bilic et al., “The liver tumor segmentation benchmark (LiTS),” arXiv e-prints, p. arXiv:1901.04056, Jan. 2019. [Online]. Available: [https://arxiv.org/abs/1901.04056](https://arxiv.org/abs/1901.04056)
