# ESA window system

This repo is used to simulate and test the CCSDS optical Pulse Position Modulation (PPM) communication protocol. The CCSDS recommended standard can be found [here](https://public.ccsds.org/Pubs/142x0b1.pdf), or via the [overview](https://public.ccsds.org/publications/BlueBooks.aspx) (Document number CCSDS 142.0-B-1). It should be noted that the standard is not followed completely, but rather used as a guideline for the encoding and decoding process.

## How to use the code
For convenience, there are three high level functions that can be used to encode and decode user data. To see how they work, one can refer to `encode_decode_image_example.py`. The three functions are:
- `encode`
- `decode`
- `demodulate`

