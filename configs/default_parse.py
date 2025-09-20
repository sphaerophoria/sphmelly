#!/usr/bin/env python3

import json
import math

batch_size = 40;
img_size = 380
conv_end_size = img_size / 4 * img_size / 4 * 8
rand_params = {
    "x_offs_range": [ -20 * 2 / 3, 20  * 2 / 3],
    "y_offs_range": [ -20 * 2 / 3, 20  * 2 / 3],
    "x_scale_range": [ 0.85, 1.0 ],
    "rot_range": [ -math.pi / 16, math.pi / 16 ],
    "aspect_range": [0.7, 1.3],
    "min_contrast": 0.5,
    "x_noise_multiplier_range": [ 1, 5],
    "y_noise_multiplier_range": [ 1, 5],
    "perlin_grid_size_range": [ 1, 100 ],
    "background_color_range": [ 0.0, 1.0 ],
    "blur_stddev_range": [ 0.0001, 9.0 ],
    "no_code_prob": 0.0,
}
config = {
    "data": {
        "batch_size": batch_size,
        "render_size": img_size,
        "label_in_frame": False,
        "confidence_metric": "none",
        "rand_params": rand_params,
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
            { "maxpool": 2 },

            { "conv": ["he", 3, 3, 4, 8] },
            { "relu": 0.1 },
            { "conv": ["he", 1, 1, 8, 4] },
            { "relu": 0.1 },
            { "conv": ["he", 3, 3, 4, 8] },
            { "relu": 0.1 },
            { "maxpool": 2 },

            { "conv": ["he", 1, 1, 8, 4] },
            { "relu": 0.1 },
            { "conv": ["he", 3, 3, 4, 8] },
            { "relu": 0.1 },
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

