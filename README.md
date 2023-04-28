# ESA window system

This repo is used to simulate and test the CCSDS optical Pulse Position Modulation (PPM) communication protocol. The CCSDS recommended standard can be found [here](https://public.ccsds.org/Pubs/142x0b1.pdf), or via the [overview](https://public.ccsds.org/publications/BlueBooks.aspx) (Document number CCSDS 142.0-B-1). It should be noted that the standard is not followed completely, but rather used as a guideline for the encoding and decoding process. Most notably, iterative decoding is not used, due to the complexity of decoding. 

## Nomenclature
| Variable | Description                                                     |
|----------|-----------------------------------------------------------------|
| M        | PPM modulation order                                            |
| m        | 2log (log base 2) of M                                          |
| B        | Base length of the channel interleaver shift register           |
| N        | Number of parallel shift registers in the channel interleaver   |
| alpha    | Likelihood for the backward recursion                           |
| beta     | Likelihood for the forward recursion                            |
| gamma    | Weight assigned to each state in the Trellis                    |
| BER      | Bit Error Ratio                                                 |
| BCJR     | Decoding algorithm (named after Bahl, Cocke, Jelinek and Raviv) |
| CSM      | Codeword Synchronisation Marker                                 |
| PPM      | Pulse Position Modulation                                       |
| BPSK     | Binary Phase Shift Keying                                       |

## Some basics about Pulse Position Modulation
For the curious, or to bring you up to speed, I'd like to present a few basic concepts of PPM. Although this is not by any means a comprehensive overview, it should be enough to grasp the basic concept of PPM, convolutional encoders and BCJR. 

## How to use the code
For convenience, there are three high level functions that can be used to encode and decode user data. To see how they work, one can refer to `encode_decode_image_example.py`. The three functions are:
- `encode`: Encodes the bit sequence to a slot mapped sequence. 
- `decode`
- `demodulate`

