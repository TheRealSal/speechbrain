import csv
import logging
import os
import sys

import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from train_S4M import S4MBrain
import time
import matplotlib.pyplot as plt


if __name__ == '__main__':

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

    s4m = S4MBrain(
        teacher=None,
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )


    sample_rate = 8000
    audio_lengths_sec = torch.linspace(0.3, 16, steps=40)

    inference_times = []

    with torch.no_grad():
        for length in audio_lengths_sec:
            input_length = int(length * sample_rate)  # Convert seconds to samples
            audio_input = torch.randn(1, input_length)  # Simulate 1-channel audio input
            dummy_targets = torch.randn(1, input_length, 2)

            # Warm-up
            _ = s4m.infer(audio_input)
            torch.cuda.synchronize()

            # Measure time for forward pass
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            #start_time = time.time()
            start_event.record()
            _ = s4m.infer(audio_input)
            end_event.record()
            #end_time = time.time()
            torch.cuda.synchronize()

            #elapsed_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            elapsed_time_ms = start_event.elapsed_time(end_event)
            inference_times.append(elapsed_time_ms)
            print(f"Audio Length: {length:.2f}s, Inference Time: {elapsed_time_ms:.2f}ms")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(audio_lengths_sec, inference_times, marker='o')
    plt.title("S4M Inference Time vs. Audio Length")
    plt.xlabel("Audio Length (seconds)")
    plt.ylabel("Inference Time (ms)")

    # Save the plot to a file
    output_path = hparams["output_folder"] + "/s4m_inference_time_vs_audio_length.png"  # Customize the path as needed
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  # High-quality PNG
    print(f"Plot saved to {output_path}")