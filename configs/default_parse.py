#!/usr/bin/env python3

import json
import math

batch_size = 96;
img_width = 380
img_height = 96
conv_end_size = img_width / 4 * img_height / 16 * 8
rand_params = {
    "x_offs_range": [ 0, 0 ],
    "y_offs_range": [ 0, 0 ],
    "x_scale_range": [ 0.47, 0.71 ],
    "rot_range": [ -math.pi, math.pi ],
    "aspect_range": [ 0.4, 1.0 ],
    "min_contrast": 0.5,
    "x_noise_multiplier_range": [ 0, 0 ],
    "y_noise_multiplier_range": [ 0, 0 ],
    "perlin_grid_size_range": [ 10, 100 ],
    "background_color_range": [ 0.0, 1.0 ],
    "blur_stddev_range": [ 0.0001, 0.15 ],
    "no_code_prob": 0.0,
}
config = {
    "data": {
        "batch_size": batch_size,
        "label_in_frame": False,
        "render_size": 866,
        "confidence_metric": "none",
        "rand_params": rand_params,
        "extract_params": {
            "extract_width": img_width,
            "extract_height": img_height,

            "width_err_range": [ -0.10, 0.10 ],
            "height_err_range": [  -0.1, 0.1 ],
            "x_err_range": [ -0.03, 0.03 ],
            "y_err_range": [ -0.03, 0.03 ],
            "rot_err_range": [ -math.pi / 16, math.pi / 16],
            "multisample": 4,
            "dilation": 1.1,
        },
        "val_rand_params": rand_params,
        "enable_backgrounds": True,
    },
    "log_freq": 10,
    "val_freq": 100,
    "val_size": 50,
    "checkpoint_freq": 2000,
    "loss_multipliers": [],
    "train_target": "bars",
    "network": {
        "lr": 0.001,
        "layers": [
            { "conv": ["he", 3, 3, 1, 4] },
            { "relu": 0.1 },
            { "maxpool": [2, 2] },

            { "conv": ["he", 3, 3, 4, 8] },
            { "relu": 0.1 },
            { "conv": ["he", 1, 1, 8, 4] },
            { "relu": 0.1 },
            { "conv": ["he", 3, 3, 4, 8] },
            { "relu": 0.1 },
            { "maxpool": [2, 4] },

            { "conv": ["he", 1, 1, 8, 4] },
            { "relu": 0.1 },
            { "conv": ["he", 3, 3, 4, 8] },
            { "relu": 0.1 },
            { "maxpool": [1, 2] },

            { "conv": ["he", 1, 1, 8, 4] },
            { "relu": 0.1 },
            { "conv": ["he", 3, 3, 4, 8] },
            { "relu": 0.1 },
            { "reshape": [ conv_end_size ] },
            { "fully_connected": [ "he", "zero", conv_end_size, 512 ] },
            { "relu": 0.1 },
            { "fully_connected": [ "he", "zero", 512, 256 ] },
            { "relu": 0.1 },
            { "fully_connected": [ "he", "zero", 256, 95 ] },
        ]
    }
}

print(json.dumps(config, indent=2))

