import numpy as np
import numpy.random
from numba.experimental import jitclass
from numba import int32
import numba


spec = [
    ('simulation_resolution_x', int32),
    ('simulation_resolution_y', int32),
    ('initial_circle_radius', int32),
    ('initial_cells_amount', int32),
    ('cells_spawn_rate', int32),
    ('trail_decay_factor', int32),
    ('trail_evaporation_factor', int32),
    ('sensors_distance', int32),
    ('sensors_size', int32),
    ('sensors_angle_span', int32),
    ('movement_distance', int32),
    ('movement_rotation', int32),
    ('cells_array', numba.int32[:, :]),
    ('matrix', numba.uint8[:, :]),
    ('cells_amount', int32),
]


@jitclass(spec)
class Physarium:
    def __init__(self,
                 simulation_resolution_x,
                 simulation_resolution_y,
                 initial_circle_radius,
                 initial_cells_amount,
                 cells_spawn_rate,
                 trail_decay_factor,
                 trail_evaporation_factor,
                 sensors_distance,
                 sensors_size,
                 sensors_angle_span,
                 movement_distance,
                 movement_rotation
                 ):
        self.simulation_resolution_x = simulation_resolution_x
        self.simulation_resolution_y = simulation_resolution_y

        self.initial_circle_radius = initial_circle_radius
        self.initial_cells_amount = initial_cells_amount
        self.cells_spawn_rate = cells_spawn_rate
        self.trail_decay_factor = trail_decay_factor
        self.trail_evaporation_factor = trail_evaporation_factor

        self.sensors_distance = sensors_distance
        self.sensors_size = sensors_size
        self.sensors_angle_span = sensors_angle_span
        self.movement_distance = movement_distance
        self.movement_rotation = movement_rotation

        self.cells_amount = self.initial_cells_amount
        self.matrix = np.zeros(shape=(self.simulation_resolution_y, self.simulation_resolution_x), dtype=np.uint8)
        self.cells_array = self.__generate_cells()

        """
        row 0 is cells x pos
        row 1 is cells y pos
        row 2 is cells rot
        """

    def get_matrix(self):
        return self.matrix

    def iterate(self):
        for i in range(self.cells_amount):
            pos_x = self.cells_array[0][i]
            pos_y = self.cells_array[1][i]
            rot = self.cells_array[2][i]
            sensors_values = self.__calculate_sensors_values(pos_x, pos_y, rot)
            new_rot = self.__calculate_rotation_angle(rot, sensors_values)
            new_pos_x, new_pos_y = self.__calculate_cell_pos(pos_x, pos_y, new_rot)
            self.__update_cell_params(i, new_pos_x, new_pos_y, new_rot)

        self.__update_matrix()

    def __update_cell_params(self, i, pos_x, pos_y, rot):
        self.cells_array[0][i] = pos_x
        self.cells_array[1][i] = pos_y
        self.cells_array[2][i] = rot

    def __calculate_cell_pos(self, pos_x, pos_y, rot):
        alpha = np.deg2rad(270 - rot)
        pos_x = int(pos_x + round(np.cos(alpha))) * self.movement_distance
        pos_y = int(pos_y + round(np.sin(alpha))) * self.movement_distance
        return pos_x, pos_y

    def __calculate_rotation_angle(self, angle, sensors_values):
        if sensors_values[0] == sensors_values[1] == sensors_values[2]:
            angle = angle + np.round(numpy.random.rand()) * self.movement_rotation
        elif sensors_values[0] > sensors_values[2]:
            angle -= self.movement_rotation
        elif sensors_values[0] < sensors_values[2]:
            angle += self.movement_rotation

        return angle

    def __calculate_sensors_values(self, pos_x: int, pos_y: int, rot: int):
        sensors_values = [0, 0, 0]
        for sensor in range(3):
            alpha = np.deg2rad(270 - rot - (self.sensors_angle_span * (sensor - 1)))
            sensor_pos_x = pos_x + int(self.sensors_distance * np.cos(alpha))
            sensor_pos_y = pos_y + int(self.sensors_distance * np.sin(alpha))
            for x in range(self.sensors_size):
                for y in range(self.sensors_size):
                    sensors_values[sensor] += self.matrix[sensor_pos_y + y][sensor_pos_x + x]
        return sensors_values

    def __update_matrix(self):
        for i in range(len(self.cells_array[0])):
            self.matrix[self.cells_array[0][i]][self.cells_array[0][i]] = 255

    def __generate_cells(self):
        cells = np.zeros(shape=(3, self.cells_amount), dtype=np.int32)

        theta = np.random.uniform(0, 2 * np.pi, self.cells_amount)
        radius = np.random.uniform(0, self.initial_circle_radius, self.cells_amount)
        x = radius * np.cos(theta) + (self.simulation_resolution_x / 2)
        y = radius * np.sin(theta) + (self.simulation_resolution_y / 2)
        cells[0] = x
        cells[1] = y
        for i in range(len(cells[2])):
            cells[2][i] = np.round(numpy.random.rand() * 360)
        return cells

