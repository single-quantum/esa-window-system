# %%
import numpy as np
from scipy.ndimage import shift


class ShiftRegister:
    def __init__(self, length):
        self.state = np.zeros(length, dtype=int)
        self.sequence = []

    def next(self, input):
        if self.state.shape[0] == 0:
            self.sequence.append(input)
            return

        self.sequence.append(self.state[-1])
        self.state = shift(self.state, 1)
        self.state[0] = input


class CRC:
    def __init__(self, seed: list[int], xor_positions: list[int]):
        """Cyclic Redundancy Check. """
        # self.input = input
        self.seed = seed
        self.state = seed
        self.xor_positions = xor_positions

    def shift(self):
        """Shift the state by one and insert a 0 at the beginning. """
        self.state = shift(self.state, 1)

    def xor(self):
        """Perform the XOR operations and insert the resulting bit into the given position. """
        for pos in self.xor_positions:
            self.state[pos] = self.state[pos] ^ self.input

    def insert(self, input: int):
        self.input = input
        self.state[0] = input

    def next(self, input):
        self.shift()
        self.insert(input)
        self.xor()


# Generate a dummy Randomized Information Block
# rng = np.random.default_rng()
# RIB = rng.integers(low=0, high=2, size=5006)


# print('state', sr.state)
# print('RIB', RIB)
# print('input', RIB[-1] ^ sr.state[-1])

# print('state', sr.state)
