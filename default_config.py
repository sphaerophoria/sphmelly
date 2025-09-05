#!/usr/bin/env python3

import json

batch_size = 64;
img_size = 256 * 0.75
conv_end_size = img_size / 16 * img_size / 16 * 4;
config = {
    "batch_size": batch_size,
    "img_size": img_size,
    "network": {
        "lr": 0.004,
        "layers": [
            { "conv": ["he", 3, 3, 1, 4] },
            { "relu": {} },
            { "maxpool": 2 },

            { "conv": ["he", 3, 3, 4, 4] },
            { "relu": {} },
            { "conv": ["he", 1, 1, 4, 4] },
            { "relu": {} },
            { "maxpool": 2 },

            { "conv": ["he", 3, 3, 4, 4] },
            { "relu": {} },
            { "conv": ["he", 1, 1, 4, 4] },
            { "relu": {} },
            { "maxpool": 2 },

            { "conv": ["he", 3, 3, 4, 4] },
            { "relu": {} },
            { "conv": ["he", 1, 1, 4, 4] },
            { "relu": {} },
            { "maxpool": 2 },

            { "conv": ["he", 3, 3, 4, 4] },
            { "relu": {} },
            { "conv": ["he", 1, 1, 4, 4] },
            { "relu": {} },

            { "reshape": [ conv_end_size ] },

            { "fully_connected": [ "he", "zero", conv_end_size, 128 ] },
            { "relu": {} },
            { "fully_connected": [ "he", "zero", 128, 128 ] },
            { "relu": {} },
            { "fully_connected": [ "he", "zero", 128, 6 ]  }
        ]
    }
}

print(json.dumps(config, indent=2))

