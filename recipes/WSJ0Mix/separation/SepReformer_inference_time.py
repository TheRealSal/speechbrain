import csv
import logging
import os
import sys

import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from train_cdf import CodecformerBrain
import time
import matplotlib.pyplot as plt
from speechbrain.lobes.models.SepReformer import SepReformer
import yaml

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

if __name__ == '__main__':
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    model_path = "/home/salmanhu/projects/def-ravanelm/salmanhu/speechbrain/recipes/WSJ0Mix/SepReformer-pretrained/epoch.0180.pth"
    state_dict = torch.load(model_path, map_location=torch.device('cuda'))

    config_path = "/home/salmanhu/projects/def-ravanelm/salmanhu/speechbrain/recipes/WSJ0Mix/SepReformer-pretrained/configs.yaml"
    yaml_dict = parse_yaml(config_path)

    config = yaml_dict["config"]

    teacher = SepReformer(**config["model"]).to("cuda")
    teacher.load_state_dict(state_dict["model_state_dict"])
    teacher.eval()


    sample_rate = 8000
    audio_lengths_sec = torch.linspace(0.3, 16, steps=40)

    inference_times = []

    with torch.no_grad():
        for length in audio_lengths_sec:
            input_length = int(length * sample_rate)  # Convert seconds to samples
            audio_input = torch.randn(1, input_length).to(teacher.device)  # Simulate 1-channel audio input
            dummy_targets = torch.randn(1, input_length, 2).to(teacher.device)

            # Warm-up
            _ = teacher.forward(audio_input)
            torch.cuda.synchronize()

            # Measure time for forward pass
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # start_time = time.time()
            start_event.record()
            _ = teacher.forward(audio_input)
            end_event.record()
            # end_time = time.time()
            torch.cuda.synchronize()

            # elapsed_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            elapsed_time_ms = start_event.elapsed_time(end_event)
            inference_times.append(elapsed_time_ms)
            print(f"Audio Length: {length:.2f}s, Inference Time: {elapsed_time_ms:.2f}ms")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(audio_lengths_sec, inference_times, marker='o')
    plt.title("SepReformer Inference Time vs. Audio Length")
    plt.xlabel("Audio Length (seconds)")
    plt.ylabel("Inference Time (ms)")

    # Save the plot to a file
    output_path = hparams["output_folder"] + "/cdf_inference_time_vs_audio_length.png"  # Customize the path as needed
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  # High-quality PNG
    print(f"Plot saved to {output_path}")