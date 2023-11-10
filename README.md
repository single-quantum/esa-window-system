# ESA window system

This repo is used to simulate and test the Pulse Position Modulation (PPM) communication protocol. It is in part based on the CCSDS standard. The CCSDS recommended standard can be found [here](https://public.ccsds.org/Pubs/142x0b1.pdf), or via the [overview](https://public.ccsds.org/publications/BlueBooks.aspx) (Document number CCSDS 142.0-B-1). It should be noted that the standard is not followed completely, but rather used as a guideline for the encoding and decoding process. Most notably, iterative decoding is not used.

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

## Resources on PPM and BCJR
In case anyone needs some helpful resources on the BCJR algorithm, here are some sources that helped me understand the concept:
- [This lecture](https://www.youtube.com/watch?v=NHkd9mz3aOQ) on BCJR by prof. Adrish Banerjee. Has some helpful step by step examples.
- Two lectures ([Part 1](https://www.youtube.com/watch?v=P5nwZQe4QYI) and [Part 2](https://www.youtube.com/watch?v=k5JwucVAwG0) by Prof. Andrew Thangaraj on BCJR and max log map decoding.
- To a lesser degree, the [paper](https://tmo.jpl.nasa.gov/progress_report/42-161/161T.pdf) referenced in the CCSDS standard (it's very densely written with a lot of implicit background knowledge).

## How to use the code
For convenience, there are three high level functions that can be used to encode and decode user data. To see how they work, one can refer to `encode_decode_image_example.py`. The three functions are:
- `encode`: Encodes a given bit sequence to a slot mapped sequence. To do this, two input parameters need to be known, the PPM order `M` and the `code_rate` (should be one of 1/3, 1/2, 2/3)
- `decode`: Decodes a slot mapped sequence, given M and a code rate.
- `demodulate`: This function is used to convert a sequence of timestamps to PPM symbols.

Further utility functions are:
- `payload_to_bit_sequence`: With this function, a given payload (string or image) can be converted to a bit stream that can then be encoded with the `encode` function. 
