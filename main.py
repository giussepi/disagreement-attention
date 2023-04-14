# -*- coding: utf-8 -*-
""" main """

import glob
import os
from collections import defaultdict

import logzero
import matplotlib.pyplot as plt
import numpy as np
import torch
from gtorch_utils.datasets.segmentation.datasets.ct82.datasets import CT82Dataset, CT82Labels
from gtorch_utils.datasets.segmentation.datasets.ct82.processors import CT82MGR
from gtorch_utils.datasets.segmentation.datasets.lits17.processors import LiTS17MGR, LiTS17CropMGR
from gtorch_utils.datasets.segmentation.datasets.lits17.datasets import LiTS17OnlyLiverLabels, \
    LiTS17Dataset, LiTS17OnlyLesionLabels, LiTS17CropDataset
from gtorch_utils.nns.managers.standard import ModelMGR
from gtorch_utils.nns.managers.adsv import ADSVModelMGR
from gtorch_utils.nns.managers.callbacks.metrics.constants import MetricEvaluatorMode
from gtorch_utils.nns.mixins.constants import LrShedulerTrack
from gtorch_utils.nns.mixins.images_types import CT3DNIfTIMixin
from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod
from gtorch_utils.nns.utils.sync_batchnorm import get_batchnormxd_class
from gtorch_utils.segmentation import loss_functions
from gtorch_utils.segmentation.loss_functions.dice import dice_coef_loss
from gutils.images.images import NIfTI, ProNIfTI
from monai.transforms import ForegroundMask
from skimage.exposure import equalize_adapthist
from tabulate import tabulate
from torchinfo import summary
from tqdm import tqdm

import settings
from nns.models import XAttentionUNet, UNet2D, \
    UNet_Grid_Attention, UNet_Att_DSV, SingleAttentionBlock, \
    MultiAttentionBlock, UNet3D, XAttentionUNet_ADSV
from nns.models.layers.disagreement_attention import intra_model


logzero.loglevel(settings.LOG_LEVEL)


def main():
    ###########################################################################
    #                                 Training                                 #
    ###########################################################################

    # # NOTE: for XAttentionUNet_ADSV employ ADSVModelMGR instead of ModelMGR

    class CTModelMGR(CT3DNIfTIMixin, ModelMGR):
        pass

    model7 = CTModelMGR(
        # UNet3D ##############################################################
        # model=UNet3D,
        # model_kwargs=dict(feature_scale=1, n_channels=1, n_classes=1, is_batchnorm=True),
        # XAttentionUNet & XAttentionUNet_ADSV ###############################
        model=XAttentionUNet,
        model_kwargs=dict(
            n_channels=1, n_classes=1, bilinear=False, batchnorm_cls=get_batchnormxd_class(),
            init_type=UNet3InitMethod.KAIMING, data_dimensions=settings.DATA_DIMENSIONS,
            da_block_cls=intra_model.MixedEmbeddedDABlock,  # EmbeddedDABlock, PureDABlock, AttentionBlock
            dsv=True,
        ),
        # UNet_Att_DSV ########################################################
        # model=UNet_Att_DSV,
        # model_kwargs=dict(
        #     feature_scale=1, n_classes=1, n_channels=1, is_batchnorm=True,
        #     attention_block_cls=SingleAttentionBlock, data_dimensions=settings.DATA_DIMENSIONS
        # ),
        # UNet_Grid_Attention #################################################
        # model=UNet_Grid_Attention,
        # model_kwargs=dict(
        #     feature_scale=1, n_classes=1, n_channels=1, is_batchnorm=True,
        #     data_dimensions=settings.DATA_DIMENSIONS
        # ),
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
    # model7()
    # summary(model7.module, (settings.BATCH_SIZE, 1, *settings.LITS17_CROP_SHAPE), depth=1, verbose=1)
    # model7.print_data_logger_summary()
    # model7.plot_and_save(None, 154)
    # id_ = '006'  # '004'
    # model7.predict(f'/media/giussepi/TOSHIBA EXT/LiTS17Liver-Pro/test/cv_fold_5/CT_{id_}.nii.gz',
    #                patch_size=(32, 80, 80))
    # model7.plot_2D_ct_gt_preds(
    #     ct_path=f'/media/giussepi/TOSHIBA EXT/LiTS17Liver-Pro/test/cv_fold_5/CT_{id_}.nii.gz',
    #     gt_path=f'/media/giussepi/TOSHIBA EXT/LiTS17Liver-Pro/test/cv_fold_5/label_{id_}.nii.gz',
    #     pred_path=f'pred_CT_{id_}.nii.gz',
    #     only_slices_with_masks=True, save_to_disk=True, dpi=300, no_axis=True, tight_layout=False,
    #     max_slices=62
    # )

    ###############################################################################
    #                                CT-82 dataset                                #
    ###############################################################################

    # PROCESSING DATASET ######################################################

    # mgr = CT82MGR(saving_path=settings.CT82_NEW_DB_PATH, target_size=settings.CT82_SIZE)
    # mgr()

    # VERIFYING GENERATED DATA ################################################

    # assert len(glob.glob(os.path.join(mgr.saving_labels_folder, r'*.nii.gz'))) == 80
    # assert len(glob.glob(os.path.join(mgr.saving_cts_folder, r'*.pro.nii.gz'))) == 80

    # files_idx = [*range(1, 83)]
    # for id_ in mgr.non_existing_ct_folders[::-1]:
    #     files_idx.pop(id_-1)

    # for subject in tqdm(files_idx):
    #     labels = NIfTI(os.path.join(mgr.saving_labels_folder, f'label_{subject:02d}.nii.gz'))
    #     cts = ProNIfTI(os.path.join(mgr.saving_cts_folder, f'CT_{subject:02d}.pro.nii.gz'))
    #     if settings.CT82_SIZE[-1] != -1:
    #         assert labels.shape == cts.shape == settings.CT82_SIZE, (labels.shape, cts.shape, settings.CT82_SIZE)
    #     else:
    #         assert labels.shape == cts.shape, (labels.shape, cts.shape)
    #         assert labels.shape[:2] == cts.shape[:2] == settings.CT82_SIZE[:2], (
    #             labels.shape, cts.shape, settings.CT82_SIZE)

    # NOTE: After running the following lines the image file 'visual_verification.png' will be
    #       created at the project root folder. You have to open it and it will be continuosly
    #       updated with new CT and mask data until the specified number of 2D scans is completely
    #       iterated (this is how it works in Ubuntu Linux 20.04.6 LTS ). See the definition of
    #       CT82MGR.perform_visual_verification to see more options
    #       https://github.com/giussepi/gtorch_utils/blob/main/gtorch_utils/datasets/segmentation/datasets/ct82/processors/ct82mgr.py#L215
    # mgr.perform_visual_verification(80, scans=[70], clahe=True)
    # # mgr.perform_visual_verification(1, scans=[72], clahe=True)
    # os.remove(mgr.VERIFICATION_IMG)

    # SPLITTING DATASET ###########################################################

    # mgr.split_processed_dataset(.20, .20, shuffle=False)  # to easily apply 5-fold CV later

    # GETTING SUBDATASETS AND PLOTTING SOME CROPS #############################

    # train, val, test = CT82Dataset.get_subdatasets(
    #     train_path=settings.CT82_TRAIN_PATH, val_path=settings.CT82_VAL_PATH, test_path=settings.CT82_TEST_PATH)
    # for db_name, dataset in zip(['train', 'val', 'test'], [train, val, test]):
    #     print(f'{db_name}: {len(dataset)}')
    #     data = dataset[0]

    #     # NIfTI.save_numpy_as_nifti(data['image'].detach().cpu().squeeze().permute(
    #     #     1, 2, 0).numpy(), f'{db_name}_img_patch.nii.gz')
    #     # NIfTI.save_numpy_as_nifti(data['mask'].detach().cpu().squeeze().permute(
    #     #     1, 2, 0).numpy(), f'{db_name}_mask_patch.nii.gz')

    #     print(data['image'].shape, data['mask'].shape)
    #     print(data['label'], data['label_name'], data['updated_mask_path'], data['original_mask'])

    #     print(data['image'].min(), data['image'].max())
    #     print(data['mask'].min(), data['mask'].max())

    #     if len(data['image'].shape) == 4:
    #         img_id = np.random.randint(0, data['image'].shape[-3])
    #         fig, axis = plt.subplots(1, 2)
    #         axis[0].imshow(
    #             equalize_adapthist(data['image'].detach().numpy()).squeeze().transpose(1, 2, 0)[..., img_id],
    #             cmap='gray'
    #         )
    #         axis[1].imshow(
    #             data['mask'].detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
    #         plt.show()
    #     else:
    #         fig, axis = plt.subplots(2, 4)
    #         for idx, d in zip([*range(4)], dataset):
    #             img_id = np.random.randint(0, d['image'].shape[-3])
    #             axis[0, idx].imshow(
    #                 equalize_adapthist(d['image'].detach().numpy()[0, ...]).squeeze().transpose(1, 2, 0)[..., img_id],
    #                 cmap='gray'
    #             )
    #             axis[1, idx].imshow(
    #                 d['mask'].detach().numpy()[0, ...].squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
    #         plt.show()

    # DATASET INSIGHTS ###################################################

    # CT82MGR().get_insights(verbose=True)

    # DICOM files 18942
    # NIfTI labels 82
    # MIN_VAL = -2048
    # MAX_VAL = 3071
    # MIN_NIFTI_SLICES_WITH_DATA = 46
    # MAX_NIFTI_SLICES_WITH_DATA = 145
    # folders PANCREAS_0025 and PANCREAS_0070 are empty
    # MIN DICOMS per subject 181
    # MAX DICOMS per subject 466

    ###########################################################################
    #                        LiTS17 ONLY LIVER DATASET                        #
    ###########################################################################

    # GENERATING DATASET ######################################################

    # Make sure to update the settings.py by:
    # 1. Commenting the code for the LITS17 Lesion 16 32x160x160-crops dataset
    # 2. Uncommenting the code for the LITS17 Liver 1 32x80x80-crops dataset and setting
    #    the right path for LITS17_SAVING_PATH

    # Update '/media/giussepi/TOSHIBA EXT/LITS/train' to reflect the right location
    # on your system of the folder containing the LiTS17 training segmentation and volume NIfTI files
    # see https://github.com/giussepi/gtorch_utils/blob/main/gtorch_utils/datasets/segmentation/datasets/lits17/README.md

    # mgr = LiTS17MGR('/media/giussepi/TOSHIBA EXT/LITS/train',
    #                 saving_path=settings.LITS17_NEW_DB_PATH,
    #                 target_size=settings.LITS17_SIZE, only_liver=True, only_lesion=False)
    # mgr()

    # DATASET INSIGHTS ########################################################

    # mgr.get_insights(verbose=True)

    # labels files: 131, CT files: 131
    #                           value
    # ------------------------  ---------------------------------------------------------
    # Files without label 1     []
    # Files without label 2     [32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]
    # Total CT files            131
    # Total segmentation files  131
    #
    #                         min    max
    # -------------------  ------  -----
    # Image value          -10522  27572
    # Slices with label 1      28    299
    # Slices with label 2       0    245
    # Height                  512    512
    # Width                   512    512
    # Depth                    74    987

    # print(mgr.get_lowest_highest_bounds())  # (-2685.5, 1726.5)

    # PLOTTING SOME 2D SCANS ##################################################

    # NOTE: After running the following lines the image file 'visual_verification.png' will be
    #       created at the project root folder. You have to open it and it will be continuosly
    #       updated with new CT and mask data until the specified number of 2D scans is completely
    #       iterated (this is how it works in Ubuntu Linux 20.04.6 LTS ). See the definition of
    #       LiTS17MGR.perform_visual_verification to see more options
    #       https://github.com/giussepi/gtorch_utils/blob/main/gtorch_utils/datasets/segmentation/datasets/lits17/processors/lits17mgr.py#L261

    # mgr.perform_visual_verification(68, scans=[40, 64], clahe=True)  # ppl 68 -> scans 64
    # os.remove(mgr.VERIFICATION_IMG)

    # ANALYZING NUMBER OF SCANS ON GENERATED LISTS17 WITH ONLY LIVER LABEL ####

    # counter = defaultdict(lambda: 0)
    # for f in tqdm(glob.glob(os.path.join(settings.LITS17_NEW_DB_PATH, '**/label_*.nii.gz'), recursive=True)):
    #     scans = NIfTI(f).shape[-1]
    #     counter[scans] += 1
    #     if scans < 32:
    #         logzero.logger.info(f'{f} has {scans} scans')
    # # a = [*counter.keys()]
    # # a.sort()
    # # print(a)
    # logzero.logger.info('SUMMARY')
    # for i in range(29, 32):
    #     logzero.logger.info(f'{counter[i]} label files have {i} scans')

    # @LiTS17Liver-Pro the labels are [29, 32, 26, ..., 299
    # and we only have 3 cases with 29 scans so we can get rid of them to
    # use the same crop size as CT-82
    # these cases are the 000, 001, 054

    # after manually removing files without the desired label and less scans than 32
    # (000, 001, 054 had 29 scans) we ended up with 256 files
    # mgr.split_processed_dataset(.20, .20, shuffle=True)

    # GETTING 2D SCANS INFORMATION ############################################

    # min_ = float('inf')
    # max_ = float('-inf')
    # min_scans = float('inf')
    # max_scans = float('-inf')
    # for f in tqdm(glob.glob(os.path.join(settings.LITS17_NEW_DB_PATH, '**/label_*.nii.gz'), recursive=True)):
    #     data = NIfTI(f).ndarray
    #     num_scans_with_labels = data.sum(axis=0).sum(axis=0).astype(bool).sum()
    #     min_scans = min(min_scans, data.shape[-1])
    #     max_scans = max(max_scans, data.shape[-1])
    #     min_ = min(min_, num_scans_with_labels)
    #     max_ = max(max_, num_scans_with_labels)
    #     assert len(np.unique(data)) == 2
    #     assert 1 in np.unique(data)
    #     # print(np.unique(NIfTI(f).ndarray))

    # table = [
    #     ['', 'value'],
    #     ['min 2D scan number with data per label file', min_],
    #     ['max 2D scan number with data per label file', max_],
    #     ['min number of 2D scans per CT', min_scans],
    #     ['max number of 2D scans per CT', max_scans],
    # ]
    # logzero.logger.info('\n%s', str(tabulate(table, headers="firstrow")))

    #                                                value
    # -------------------------------------------  -------
    # min 2D scan number with data per label file       32
    # max 2D scan number with data per label file      299
    # min number of 2D scans per CT                     32
    # max number of 2D scans per CT                    299

    # GETTING SUBDATASETS AND PLOTTING SOME CROPS #############################

    # train, val, test = LiTS17Dataset.get_subdatasets(
    #     settings.LITS17_TRAIN_PATH,
    #     settings.LITS17_VAL_PATH,
    #     settings.LITS17_TEST_PATH,
    # )
    # for db_name, dataset in zip(['train', 'val', 'test'], [train, val, test]):
    #     logzero.logger.info(f'{db_name}: {len(dataset)}')
    #     data = dataset[0]
    #     # logzero.logger.info(data['image'].shape, data['mask'].shape)
    #     # logzero.logger.info(data['label'], data['label_name'], data['updated_mask_path'], data['original_mask'])
    #     # logzero.logger.info(data['image'].min(), data['image'].max())
    #     # logzero.logger.info(data['mask'].min(), data['mask'].max())

    #     if len(data['image'].shape) == 4:
    #         img_id = np.random.randint(0, data['image'].shape[-3])
    #         fig, axis = plt.subplots(1, 2)
    #         axis[0].imshow(
    #             equalize_adapthist(data['image'].detach().numpy()).squeeze().transpose(1, 2, 0)[..., img_id],
    #             cmap='gray'
    #         )
    #         axis[1].imshow(data['mask'].detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
    #         plt.show()
    #     else:
    #         num_crops = dataset[0]['image'].shape[0]
    #         imgs_per_row = 4
    #         for ii in range(0, len(dataset), imgs_per_row):
    #             fig, axis = plt.subplots(2, imgs_per_row*num_crops)
    #             # for idx, d in zip([*range(imgs_per_row)], dataset):
    #             for idx in range(imgs_per_row):
    #                 d = dataset[idx+ii]
    #                 for cidx in range(num_crops):
    #                     img_id = np.random.randint(0, d['image'].shape[-3])
    #                     axis[0, idx*num_crops+cidx].imshow(
    #                         equalize_adapthist(d['image'][cidx].detach().numpy()
    #                                            ).squeeze().transpose(1, 2, 0)[..., img_id],
    #                         cmap='gray'
    #                     )
    #                     axis[0, idx*num_crops+cidx].set_title(f'CT{idx}-{cidx}')
    #                     axis[1, idx*num_crops+cidx].imshow(d['mask'][cidx].detach().numpy(
    #                     ).squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
    #                     axis[1, idx*num_crops+cidx].set_title(f'Mask{idx}-{cidx}')

    #             fig.suptitle('CTs and Masks')
    #             plt.show()

    ###########################################################################
    #                 LiTS17 16-crop lesion dataset 32x160x160                #
    ###########################################################################

    # GENERATING DATASET  #####################################################

    # Make sure to update the settings.py by:
    # 1. Commenting the code for the LITS17 Liver 1 32x80x80-crops dataset
    # 2. Uncommenting the code for the LITS17 Lesion 16 32x160x160-crops dataset and setting
    #    the right path for LITS17_SAVING_PATH

    # Update '/media/giussepi/TOSHIBA EXT/LITS/train' to reflect the right location
    # on your system of the folder containing the LiTS17 training segmentation and volume NIfTI files
    # see https://github.com/giussepi/gtorch_utils/blob/main/gtorch_utils/datasets/segmentation/datasets/lits17/README.md

    # mgr = LiTS17MGR('/media/giussepi/TOSHIBA EXT/LITS/train',
    #                 saving_path=settings.LITS17_NEW_DB_PATH,
    #                 target_size=settings.LITS17_SIZE, only_liver=False, only_lesion=True)
    # mgr.get_insights(verbose=True)
    # print(mgr.get_lowest_highest_bounds())
    # mgr()

    # ENSURING THE DATASET WAS PROPERLY GENERATED #############################

    # mgr.verify_generated_db_target_size()

    # NOTE: After running the following lines the image file 'visual_verification.png' will be
    #       created at the project root folder. You have to open it and it will be continuosly
    #       updated with new CT and mask data until the specified number of 2D scans is completely
    #       iterated (this is how it works in Ubuntu Linux 20.04.6 LTS ). See the definition of
    #       LiTS17MGR.perform_visual_verification to see more options
    #       https://github.com/giussepi/gtorch_utils/blob/main/gtorch_utils/datasets/segmentation/datasets/lits17/processors/lits17mgr.py#L261

    # mgr.perform_visual_verification(68, scans=[127, 135], clahe=True)  # ppl 68 -> scans 127-135
    # os.remove(mgr.VERIFICATION_IMG)

    # SPLITTING DATASET #######################################################

    # after manually removing Files without label 2
    # [32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]
    # we ended up with 118 files
    # mgr.split_processed_dataset(.20, .20, shuffle=True)

    # GENERATING CROP DATASET #################################################

    # we aim to work with crops masks with an minimum area of 16 so min_mask_area
    # for the following heightxheight crops are:
    # height x height x min_mask_area = 16
    # 80x80x25e-4 = 16
    # 160x160x625e-6 = 16

    # LiTS17CropMGR(
    #     settings.LITS17_NEW_DB_PATH,
    #     patch_size=tuple([*settings.LITS17_CROP_SHAPE[1:], settings.LITS17_CROP_SHAPE[0]]),
    #     patch_overlapping=(.75, .75, .75), only_crops_with_masks=True, min_mask_area=625e-6,
    #     foregroundmask_threshold=.59, min_crop_mean=.63, crops_per_label=settings.LITS17_NUM_CROPS,
    #     adjust_depth=True, centre_masks=True, saving_path=settings.LITS17_NEW_CROP_DB_PATH
    # )()

    # crops per lael = 4
    # Total crops created: 472
    # Label 2 crops: 0
    # Label 1 crops: 472
    # Label 0 crops: 0
    # crops per label = 16
    # Total crops created: 1888
    # Label 2 crops: 0
    # Label 1 crops: 1888
    # Label 0 crops: 0

    # GETTING SUBDATASETS AND PLOTTING SOME CROPS #############################

    # train, val, test = LiTS17CropDataset.get_subdatasets(
    #     settings.LITS17_TRAIN_PATH,
    #     settings.LITS17_VAL_PATH,
    #     settings.LITS17_TEST_PATH,
    # )
    # for db_name, dataset in zip(['train', 'val', 'test'], [train, val, test]):
    #     logzero.logger.info(f'{db_name}: {len(dataset)}')
    #     # for _ in tqdm(dataset):
    #     #     pass
    #     # data = dataset[0]

    #     for data_idx in range(len(dataset)):
    #         data = dataset[data_idx]
    #         # logzero.logger.info(data['image'].shape, data['mask'].shape)
    #         # logzero.logger.info(data['label'], data['label_name'], data['updated_mask_path'], data['original_mask'])
    #         # logzero.logger.info(data['image'].min(), data['image'].max())
    #         # logzero.logger.info(data['mask'].min(), data['mask'].max())
    #         if len(data['image'].shape) == 4:
    #             img_ids = [np.random.randint(0, data['image'].shape[-3])]

    #             # uncomment these lines to only plot crops with masks
    #             # if 1 not in data['mask'].unique():
    #             #     continue
    #             # else:
    #             #     # selecting an idx containing part of the mask
    #             #     img_ids = data['mask'].squeeze().sum(axis=-1).sum(axis=-1).nonzero().squeeze()

    #             foreground_mask = ForegroundMask(threshold=.59, invert=True)(data['image'])
    #             std, mean = torch.std_mean(data['image'], unbiased=False)
    #             fstd, fmean = torch.std_mean(foreground_mask, unbiased=False)

    #             # once you have chosen a good mean, uncomment the following
    #             # lines and replace .63 with your chosen mean to verify that
    #             # only good crops are displayed.
    #             # if fmean < .63:
    #             #     continue

    #             logzero.logger.info(f"SUM: {data['image'].sum()}")
    #             logzero.logger.info(f"STD MEAN: {std} {mean}")
    #             logzero.logger.info(f"SUM: {foreground_mask.sum()}")
    #             logzero.logger.info(f"foreground mask STD MEAN: {fstd} {fmean}")

    #             for img_id in img_ids:
    #                 fig, axis = plt.subplots(1, 3)
    #                 axis[0].imshow(
    #                     equalize_adapthist(data['image'].detach().numpy()).squeeze().transpose(1, 2, 0)[..., img_id],
    #                     cmap='gray'
    #                 )
    #                 axis[0].set_title('Img')
    #                 axis[1].imshow(data['mask'].detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
    #                 axis[1].set_title('mask')
    #                 axis[2].imshow(foreground_mask.detach().numpy().squeeze()
    #                                .transpose(1, 2, 0)[..., img_id], cmap='gray')
    #                 axis[2].set_title('foreground_mask')
    #                 plt.show()
    #                 plt.clf()
    #                 plt.close()
    #         else:
    #             num_crops = dataset[0]['image'].shape[0]
    #             imgs_per_row = 4
    #             for ii in range(0, len(dataset), imgs_per_row):
    #                 fig, axis = plt.subplots(2, imgs_per_row*num_crops)
    #                 # for idx, d in zip([*range(imgs_per_row)], dataset):
    #                 for idx in range(imgs_per_row):
    #                     d = dataset[idx+ii]
    #                     for cidx in range(num_crops):
    #                         img_id = np.random.randint(0, d['image'].shape[-3])
    #                         axis[0, idx*num_crops+cidx].imshow(
    #                             equalize_adapthist(d['image'][cidx].detach().numpy()
    #                                                ).squeeze().transpose(1, 2, 0)[..., img_id],
    #                             cmap='gray'
    #                         )
    #                         axis[0, idx*num_crops+cidx].set_title(f'CT{idx}-{cidx}')
    #                         axis[1, idx*num_crops+cidx].imshow(d['mask'][cidx].detach().numpy(
    #                         ).squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
    #                         axis[1, idx*num_crops+cidx].set_title(f'Mask{idx}-{cidx}')

    #                 fig.suptitle('CTs and Masks')
    #                 plt.show()

    # end of main #############################################################
    logzero.logger.info('End of main.py :)')


if __name__ == '__main__':
    main()
