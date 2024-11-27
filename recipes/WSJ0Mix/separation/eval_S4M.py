#######
# !/usr/bin/env/python3
"""Recipe for training a neural speech separation system on wsjmix the
dataset. The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer.yaml
> python train.py hparams/dualpath_rnn.yaml
> python train.py hparams/convtasnet.yaml

The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both wsj2mix and
wsj3mix.

Modifications were made to the compute_forward, compute_objectives, and save_results method to allow for training and testing for codecformer.
This training script will not be compatible with other speech separation models.

Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020

Modified by
 * Yip Jia Qi 2024
"""
import csv
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

import speechbrain as sb
from speechbrain.inference.separation import SepformerSeparation as pre_separator
import speechbrain.nnet.schedulers as schedulers
from speechbrain.core import AMPConfig
from speechbrain.utils.distributed import run_on_main
from speechbrain.lobes.models.SepReformer import SepReformer
import yaml
from train_S4M import S4MBrain


def dataio_prep(hparams):
    """Creates data processing pipeline"""

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Provide audio pipelines

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav):
        mix_sig = sb.dataio.dataio.read_audio(mix_wav)
        return mix_sig

    @sb.utils.data_pipeline.takes("s1_wav")
    @sb.utils.data_pipeline.provides("s1_sig")
    def audio_pipeline_s1(s1_wav):
        s1_sig = sb.dataio.dataio.read_audio(s1_wav)
        return s1_sig

    @sb.utils.data_pipeline.takes("s2_wav")
    @sb.utils.data_pipeline.provides("s2_sig")
    def audio_pipeline_s2(s2_wav):
        s2_sig = sb.dataio.dataio.read_audio(s2_wav)
        return s2_sig

    if hparams["num_spks"] == 3:
        @sb.utils.data_pipeline.takes("s3_wav")
        @sb.utils.data_pipeline.provides("s3_sig")
        def audio_pipeline_s3(s3_wav):
            s3_sig = sb.dataio.dataio.read_audio(s3_wav)
            return s3_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s2)
    if hparams["num_spks"] == 3:
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s3)
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig", "s3_sig"]
        )
    else:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig"]
        )

    return train_data, valid_data, test_data


def parse_yaml(path):
    """
        Parse and return the contents of a YAML file.

        Args:
            path (str): Path to the YAML file to be parsed.

        Returns:
            dict: A dictionary containing the parsed contents of the YAML file.

        Raises:
            FileNotFoundError: If the provided path does not point to an existing file.
        """
    try:
        with open(path, 'r') as yaml_file:
            config_dict = yaml.full_load(yaml_file)
        return config_dict
    except FileNotFoundError:
        raise


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = logging.getLogger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Update precision to bf16 if the device is CPU and precision is fp16
    if run_opts.get("device") == "cpu" and hparams.get("precision") == "fp16":
        hparams["precision"] = "bf16"

    # Check if wsj0_tr is set with dynamic mixing
    if hparams["dynamic_mixing"] and not os.path.exists(
            hparams["base_folder_dm"]
    ):
        raise ValueError(
            "Please, specify a valid base_folder_dm folder when using dynamic mixing"
        )

    # Data preparation
    from prepare_data import prepare_wsjmix  # noqa

    run_on_main(
        prepare_wsjmix,
        kwargs={
            "datapath": hparams["data_folder"],
            "savepath": hparams["save_folder"],
            "n_spks": hparams["num_spks"],
            "skip_prep": hparams["skip_prep"],
            "fs": hparams["sample_rate"],
        },
    )

    # Create dataset objects
    if hparams["dynamic_mixing"]:
        from dynamic_mixing import dynamic_mix_data_prep

        # if the base_folder for dm is not processed, preprocess them
        if "processed" not in hparams["base_folder_dm"]:
            # if the processed folder already exists we just use it otherwise we do the preprocessing
            if not os.path.exists(
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
            ):
                from preprocess_dynamic_mixing import resample_folder

                print("Resampling the base folder")
                run_on_main(
                    resample_folder,
                    kwargs={
                        "input_folder": hparams["base_folder_dm"],
                        "output_folder": os.path.normpath(
                            hparams["base_folder_dm"]
                        )
                                         + "_processed",
                        "fs": hparams["sample_rate"],
                        "regex": "**/*.wav",
                    },
                )
                # adjust the base_folder_dm path
                hparams["base_folder_dm"] = (
                        os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )
            else:
                print(
                    "Using the existing processed folder on the same directory as base_folder_dm"
                )
                hparams["base_folder_dm"] = (
                        os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )

        # Collecting the hparams for dynamic batching
        dm_hparams = {
            "train_data": hparams["train_data"],
            "data_folder": hparams["data_folder"],
            "base_folder_dm": hparams["base_folder_dm"],
            "sample_rate": hparams["sample_rate"],
            "num_spks": hparams["num_spks"],
            "training_signal_len": hparams["training_signal_len"],
            "dataloader_opts": hparams["dataloader_opts"],
        }
        train_data = dynamic_mix_data_prep(dm_hparams)
        _, valid_data, test_data = dataio_prep(hparams)
    else:
        train_data, valid_data, test_data = dataio_prep(hparams)

    # # Load pretrained model if pretrained_separator is present in the yaml
    # if "pretrained_separator" in hparams:
    #     run_on_main(hparams["pretrained_separator"].collect_files)
    #     hparams["pretrained_separator"].load_collected()

    # Load pretrained(teacher) separator model
    teacher = None
    # if hparams["use_teacher"]:
    #     print("Using teacher")
    #     model_path = "/home/salmanhu/projects/def-ravanelm/salmanhu/speechbrain/recipes/WSJ0Mix/SepReformer-pretrained/epoch.0180.pth"
    #     state_dict = torch.load(model_path, map_location=torch.device('cuda'))
    #
    #     config_path = "/home/salmanhu/projects/def-ravanelm/salmanhu/speechbrain/recipes/WSJ0Mix/SepReformer-pretrained/configs.yaml"
    #     yaml_dict = parse_yaml(config_path)
    #
    #     config =yaml_dict["config"]
    #
    #     teacher = SepReformer(**config["model"]).to("cuda")
    #     teacher.load_state_dict(state_dict["model_state_dict"])
    #     teacher.eval()

    # Brain class initialization
    separator = S4MBrain(
        teacher=teacher,
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    # Eval
    separator.evaluate(test_data, min_key="si-snr")
    separator.save_results(test_data)


