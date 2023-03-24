from numba.experimental import jitclass
import interfaces
from numba import int32
import numpy.random
import numpy as np
import numba


spec = [
    ('__simulation_resolution_x', int32),
    ('__simulation_resolution_y', int32),
    ('__initial_circle_radius', int32),
    ('__initial_cells_amount', int32),
    ('__cells_spawn_rate', int32),
    ('__trail_decay_factor', int32),
    ('__trail_evaporation_factor', int32),
    ('__sensors_distance', int32),
    ('__sensors_size', int32),
    ('__sensors_angle_span', int32),
    ('__movement_distance', int32),
    ('__movement_rotation', int32),
    ('__cells_array', numba.int32[:, :]),
    ('__matrix', numba.uint8[:, :]),
    ('__cells_amount', int32),
]


@jitclass(spec)
class Physarum(interfaces.Physarum):
    def __init__(self,
                 simulation_resolution_x: int,
                 simulation_resolution_y: int,
                 initial_circle_radius: int,
                 initial_cells_amount: int,
                 cells_spawn_rate: int,
                 trail_decay_factor: int,
                 trail_evaporation_factor: int,
                 sensors_distance: int,
                 sensors_size: int,
                 sensors_angle_span: int,
                 movement_distance: int,
                 movement_rotation: int
                 ):
        self.__simulation_resolution_x = simulation_resolution_x
        self.__simulation_resolution_y = simulation_resolution_y

        self.__initial_circle_radius = initial_circle_radius
        self.__initial_cells_amount = initial_cells_amount
        self.__cells_spawn_rate = cells_spawn_rate
        self.__trail_decay_factor = trail_decay_factor
        self.__trail_evaporation_factor = trail_evaporation_factor

        self.__sensors_distance = sensors_distance
        self.__sensors_size = sensors_size
        self.__sensors_angle_span = sensors_angle_span
        self.__movement_distance = movement_distance
        self.__movement_rotation = movement_rotation

        self.__cells_amount = 0
        self.__matrix = np.zeros(shape=(self.__simulation_resolution_y, self.__simulation_resolution_x), dtype=np.uint8)
        self.__cells_array = self.__generate_cells(self.__initial_cells_amount)


        self.iterate()

        """
        row 0 is cells x pos
        row 1 is cells y pos
        row 2 is cells rot
        """

    def get_cells_array(self):
        return self.__cells_array

    def get_cells_amount(self):
        return self.__cells_amount

    def get_matrix(self):
        return self.__matrix

    def iterate(self):
        self.__cells_array = np.concatenate((self.__cells_array, self.__generate_cells(self.__cells_spawn_rate)), axis=1)

        for i in range(self.__cells_amount):
            pos_x = self.__cells_array[0][i]
            pos_y = self.__cells_array[1][i]
            rot = self.__cells_array[2][i]
            if not self.__check_bounds(pos_x, pos_y):
                pos_x = 0
                pos_y = 0
            sensors_values = self.__calculate_sensors_values(pos_x, pos_y, rot)
            new_rot = self.__calculate_rotation_angle(rot, sensors_values)
            new_pos_x, new_pos_y = self.__calculate_cell_pos(pos_x, pos_y, new_rot)
            self.__update_cell_params(i, new_pos_x, new_pos_y, new_rot)

        self.__update_matrix()
        self.__evaporate_cells()
        self.__apply_gaussian_filter()

    def __check_bounds(self, pos_x, pos_y):
        if pos_x >= self.__simulation_resolution_x - self.__sensors_distance:
            return False
        elif pos_x <= self.__sensors_distance:
            return False
        elif pos_y >= self.__simulation_resolution_y - self.__sensors_distance:
            return False
        elif pos_y <= self.__sensors_distance:
            return False
        else:
            return True

    def __apply_gaussian_filter(self):
        neighborhood_size = self.__trail_decay_factor
        result = np.zeros_like(self.__matrix, dtype=np.uint8)

        radius = neighborhood_size // 2

        for i in range(radius, self.__simulation_resolution_y - radius):
            for j in range(radius, self.__simulation_resolution_x - radius):
                # Compute mean of neighborhood
                neighborhood_sum = 0
                for ii in range(i - radius, i + radius + 1):
                    for jj in range(j - radius, j + radius + 1):
                        neighborhood_sum += self.__matrix[ii, jj]
                mean = neighborhood_sum / (neighborhood_size ** 2)

                # Set new pixel value
                if mean > 255:
                    mean = 255
                if mean < 0:
                    mean = 0
                result[i, j] = int(mean)

        self.__matrix = result.astype(np.uint8)

    def __evaporate_cells(self):
        for y in numba.prange(self.__simulation_resolution_y):
            for x in range(self.__simulation_resolution_x):
                if self.__matrix[y][x] > self.__trail_evaporation_factor:
                    self.__matrix[y][x] = self.__matrix[y][x] - self.__trail_evaporation_factor
                else:
                    self.__matrix[y][x] = 0

        """for y in range(self.__simulation_resolution_y):
            for x in range(self.__simulation_resolution_x):
                if self.__matrix[y][x] > self.__trail_evaporation_factor:
                    self.__matrix[y][x] = self.__matrix[y][x] - self.__trail_evaporation_factor
                else:
                    self.__matrix[y][x] = 0"""

    def __update_cell_params(self, i, pos_x, pos_y, rot):
        self.__cells_array[0][i] = pos_x
        self.__cells_array[1][i] = pos_y
        self.__cells_array[2][i] = rot

    def __calculate_cell_pos(self, pos_x, pos_y, rot):
        alpha = np.deg2rad(270 - rot)
        pos_x = int(pos_x + round(np.cos(alpha)) * self.__movement_distance)
        pos_y = int(pos_y + round(np.sin(alpha)) * self.__movement_distance)
        return pos_x, pos_y

    def __calculate_rotation_angle(self, angle, sensors_values):
        multiply = numpy.random.rand()
        if sensors_values[0] == sensors_values[1] == sensors_values[2]:
            angle = angle + np.round((numpy.random.rand() * 2) - 1) * self.__movement_rotation
        elif sensors_values[0] > sensors_values[2]:
            angle -= self.__movement_rotation * multiply
        elif sensors_values[0] < sensors_values[2]:
            angle += self.__movement_rotation * multiply

        return angle

    def __calculate_sensors_values(self, pos_x: int, pos_y: int, rot: int):
        sensors_values = [0, 0, 0]
        offset = int(self.__sensors_size/2)
        for sensor in range(3):
            alpha = np.deg2rad(270 - rot - (self.__sensors_angle_span * (sensor - 1)))
            sensor_pos_x = pos_x + int(self.__sensors_distance * np.cos(alpha))
            sensor_pos_y = pos_y + int(self.__sensors_distance * np.sin(alpha))
            for x in range(self.__sensors_size):
                for y in range(self.__sensors_size):
                    sensors_values[sensor] += self.__matrix[sensor_pos_y + y - offset][sensor_pos_x + x - offset]
        return sensors_values

    def __update_matrix(self):
        for i in range(len(self.__cells_array[0])):
            self.__matrix[self.__cells_array[1][i]][self.__cells_array[0][i]] = 255

    def __generate_cells(self, amount):
        cells = np.zeros(shape=(3, amount), dtype=np.int32)

        theta = np.random.uniform(0, 2 * np.pi, amount)
        radius = np.random.uniform(0, self.__initial_circle_radius, amount)
        x = radius * np.cos(theta) + (self.__simulation_resolution_x / 2)
        y = radius * np.sin(theta) + (self.__simulation_resolution_y / 2)
        cells[0] = x
        cells[1] = y
        for i in range(len(cells[2])):
            cells[2][i] = np.round(numpy.random.rand() * 360)

        self.__cells_amount += amount

        return cells

