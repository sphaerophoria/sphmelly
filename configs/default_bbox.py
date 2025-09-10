#!/usr/bin/env python3

import json
import math

batch_size = 64;
img_size = 256 * 0.75
conv_end_size = img_size / 16 * img_size / 16 * 4;
config = {
    "data": {
        "batch_size": batch_size,
        "img_size": img_size,
        "rand_params": {
            "x_offs_range": [ -40, 40 ],
            "y_offs_range": [ -40, 40 ],
            "x_scale_range": [ 0.40, 0.60 ],
            "rot_range": [ -math.pi / 2 , math.pi / 2 ],
            "aspect_range": [ 0.4, 1.0 ],
            "min_contrast": 0.5,
            "x_noise_multiplier_range": [ 0, 0 ],
            "y_noise_multiplier_range": [ 0, 0 ],
            "perlin_grid_size_range": [ 10, 100 ],
            "background_color_range": [ 0.0, 1.0 ],
            "blur_stddev_range": [ 0.0001, 0.15 ],
        },
        "enable_backgrounds": True,
    },
    "log_freq": 10,
    "val_freq": 100,
    "checkpoint_freq": 5000,
    "heal_orientations": True,
    "loss_multipliers": [ 1, 1, 1, 0.6, 0.05, 0.05 ],
    "train_target": "bbox",
    "network": {
        "lr": 0.004,
        "layers": [
            { "conv": ["he", 3, 3, 1, 4] },
            { "relu": 0.1 },
            { "maxpool": 2 },

            { "conv": ["he", 3, 3, 4, 4] },
            { "relu": 0.1 },
            { "conv": ["he", 1, 1, 4, 4] },
            { "relu": 0.1 },
            { "maxpool": 2 },

            { "conv": ["he", 3, 3, 4, 4] },
            { "relu": 0.1 },
            { "conv": ["he", 1, 1, 4, 4] },
            { "relu": 0.1 },
            { "maxpool": 2 },

            { "conv": ["he", 3, 3, 4, 4] },
            { "relu": 0.1 },
            { "conv": ["he", 1, 1, 4, 4] },
            { "relu": 0.1 },
            { "maxpool": 2 },

            { "conv": ["he", 3, 3, 4, 4] },
            { "relu": 0.1 },
            { "conv": ["he", 1, 1, 4, 4] },
            { "relu": 0.1 },

            { "reshape": [ conv_end_size ] },
            { "fully_connected": [ "he", "zero", conv_end_size, 128 ] },
            { "relu": 0.1 },
            { "fully_connected": [ "he", "zero", 128, 128 ] },
            { "relu": 0.1 },
            { "fully_connected": [ "he", "zero", 128, 6 ] },
        ]
    }
}

print(json.dumps(config, indent=2))

