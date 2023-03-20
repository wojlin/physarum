import numpy.random
from numba import jit, cuda
from numba.experimental import jitclass
import numpy as np
import numba
from numba import int32, float32, int16

spec = [
    ('initial_circle_radius', int32),
    ('initial_cells_amount', int32),               # a simple scalar field
    ('cells_array', numba.int32[:, :]),          # an array field
]

@jitclass(spec)
class Physarium:
    def __init__(self, initial_circle_radius, initial_cells_amount):
        initial_circle_radius = 5
        self.initial_cells_amount = initial_cells_amount
        self.cells_array = np.zeros(shape=(3, initial_cells_amount), dtype=np.int32)

        """
        row 0 is cells x pos
        row 1 is cells y pos
        row 2 is cells rot
        """
    @staticmethod
    def __generate_random_rotation():
        return np.round(numpy.random.rand() * 360)

    @staticmethod
    def __generate_random_position():
        return np.round(numpy.random.rand() * 360)

    def generate_cells(self):
        for i in range(len(self.cells_array[2])):
            self.cells_array[2][i] = self.__generate_random_rotation()


if __name__ == "__main__":
    physarium = Physarium(5, 10)
    physarium.generate_cells()
    print(physarium.cells_array)