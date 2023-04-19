# -*- coding: utf-8 -*-
""" main """

import glob
import os
from collections import defaultdict

import logzero
import matplotlib.pyplot as plt
import numpy as np
import torch
from gtorch_utils.constants import DB
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
        labels_data=LiTS17OnlyLiverLabels,  # LiTS17OnlyLesionLabels,  # CT82Labels
        data_dimensions=settings.DATA_DIMENSIONS,
        dataset=LiTS17Dataset,  # LiTS17CropDataset,  # CT82Dataset
        dataset_kwargs={
            'train_path': settings.LITS17_TRAIN_PATH,  # settings.CT82_TRAIN_PATH
            'val_path': settings.LITS17_VAL_PATH,   # settings.CT82_VAL_PATH,
            'test_path': settings.LITS17_TEST_PATH,  # settings.CT82_TEST_PATH,
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

    # end of main #############################################################
    logzero.logger.info('End of main.py :)')


if __name__ == '__main__':
    main()
