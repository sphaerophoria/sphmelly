#!/usr/bin/env python3

import json

batch_size = 64;
img_size = 256
conv_end_size = img_size / 16 * img_size / 16 * 128;
config = {
    "batch_size": batch_size,
    "img_size": img_size,
    "network": {
        "lr": 0.004,
        "layers": [
            { "conv": ["he", 3, 3, 1, 4] },
            { "relu": {} },
            { "maxpool": 2 },

            { "conv": ["he", 3, 3, 4, 8] },
            { "relu": {} },
            { "conv": ["he", 1, 1, 8, 16] },
            { "relu": {} },
            { "maxpool": 2 },

            { "conv": ["he", 3, 3, 16, 32] },
            { "relu": {} },
            { "conv": ["he", 1, 1, 32, 32] },
            { "relu": {} },
            { "maxpool": 2 },

            { "conv": ["he", 3, 3, 32, 64] },
            { "relu": {} },
            { "conv": ["he", 1, 1, 64, 64] },
            { "relu": {} },
            { "maxpool": 2 },

            { "conv": ["he", 3, 3, 64, 128] },
            { "relu": {} },
            { "conv": ["he", 1, 1, 128, 128] },
            { "relu": {} },

            { "reshape": [ conv_end_size ] },

            { "fully_connected": [ "he", "zero", conv_end_size, 512 ] },
            { "relu": {} },
            { "fully_connected": [ "he", "zero", 512, 512 ] },
            { "relu": {} },
            { "fully_connected": [ "he", "zero", 512, 256 ] },
            { "relu": {} },
            { "fully_connected": [ "he", "zero", 256, 6 ]  }
        ]
    }
}

print(json.dumps(config, indent=2))

