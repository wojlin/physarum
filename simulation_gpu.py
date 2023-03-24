import interfaces
import numpy as np
import numba
from numba import cuda


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

        print("child")

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

        self.__gpu = cuda.jit(restype=numba.uint32, argtypes=[])

        self.__cells_array, self.__cells_amount = self.__generate_cells(self.__cells_amount,
                                                                        self.__cells_spawn_rate,
                                                                        self.__simulation_resolution_x,
                                                                        self.__simulation_resolution_y,
                                                                        self.__initial_circle_radius)

        print("gpu")

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
        cells, self.__cells_amount = self.__generate_cells(self.__cells_amount,
                                      self.__cells_spawn_rate,
                                      self.__simulation_resolution_x,
                                      self.__simulation_resolution_y,
                                      self.__initial_circle_radius)

        self.__cells_array = np.concatenate((self.__cells_array, cells),
                                            axis=1)



        for i in range(self.__cells_amount):
            pos_x = self.__cells_array[0][i]
            pos_y = self.__cells_array[1][i]
            rot = self.__cells_array[2][i]
            if not self.__check_bounds(pos_x,
                                       pos_y,
                                       self.__simulation_resolution_x,
                                       self.__simulation_resolution_y,
                                       self.__sensors_distance):
                pos_x = 0
                pos_y = 0

            sensors_values = self.__calculate_sensors_values(self.__matrix,
                                                             pos_x,
                                                             pos_y,
                                                             rot,
                                                             self.__sensors_size,
                                                             self.__sensors_angle_span,
                                                             self.__sensors_distance)

            sensors_values = np.array(sensors_values)

            new_rot = self.__calculate_rotation_angle(rot,
                                                      sensors_values,
                                                      self.__movement_rotation)

            new_pos_x, new_pos_y = self.__calculate_cell_pos(pos_x,
                                                             pos_y,
                                                             rot,
                                                             self.__sensors_distance)

            self.__cells_array = self.__update_cell_params(self.__cells_array, i, new_pos_x, new_pos_y, new_rot)

        self.__update_matrix(self.__matrix,self.__cells_array)
        self.__apply_gaussian_filter(self.__matrix, self.__simulation_resolution_x, self.__simulation_resolution_y,
                                     self.__trail_decay_factor)
        self.__evaporate_cells(self.__matrix,self.__simulation_resolution_x,self.__simulation_resolution_y,self.__trail_evaporation_factor)



    @staticmethod
    @numba.cuda.jit(device= True, target_backend='cuda', nopython=True)
    def __check_bounds(pos_x, pos_y, sim_res_x, sim_res_y, sensor_dis):
        if pos_x >= sim_res_x - sensor_dis:
            return False
        elif pos_x <= sensor_dis:
            return False
        elif pos_y >= sim_res_y - sensor_dis:
            return False
        elif pos_y <= sensor_dis:
            return False
        else:
            return True

    @staticmethod
    @numba.cuda.jit(target_backend='cuda', nopython=True)
    def __apply_gaussian_filter(matrix, sim_res_x, sim_res_y, decay):
        neighborhood_size = decay
        result = np.zeros_like(matrix, dtype=np.int32)

        radius = neighborhood_size // 2

        print(radius)

        for i in numba.prange(radius, sim_res_y - radius):
            for j in numba.prange(radius, sim_res_x - radius):
                # Compute mean of neighborhood
                neighborhood_sum = 0
                for ii in range(i - radius, i + radius + 1):
                    for jj in range(j - radius, j + radius + 1):
                        neighborhood_sum = neighborhood_sum + matrix[ii, jj]
                mean = neighborhood_sum / (neighborhood_size * neighborhood_size)
                if mean > 255:
                    mean = 255
                if mean < 0:
                    mean = 0
                result[i, j] = int(mean)

        for y in range(sim_res_y):
            for x in range(sim_res_x):
                if matrix[y][x] + result[y][x] > 255:
                    matrix[y][x] = 255
                elif result[y][x] <= 50:
                    if matrix[y][x] > 10:
                        matrix[y][x] -= 10
                    else:
                        matrix[y][x] = 0
                else:
                    matrix[y][x] += result[y][x]
        #matrix = result.astype(np.uint8)

    @staticmethod
    @numba.cuda.jit(target_backend='cuda', nopython=True)
    def __evaporate_cells(matrix, sim_res_x, sim_res_y, evaporation):
        for y in numba.prange(sim_res_x):
            for x in range(sim_res_y):
                if matrix[y][x] > evaporation:
                    matrix[y][x] = matrix[y][x] - evaporation
                else:
                    matrix[y][x] = 0

    @staticmethod
    @numba.cuda.jit(target_backend='cuda', nopython=True)
    def __update_cell_params(cells, i, pos_x, pos_y, rot):
        cells[0][i] = pos_x
        cells[1][i] = pos_y
        cells[2][i] = rot
        return cells

    @staticmethod
    @numba.cuda.jit(target_backend='cuda', nopython=True)
    def __calculate_cell_pos(pos_x, pos_y, rot, movement_distance):
        alpha = np.deg2rad(270 - rot)
        pos_x = int(pos_x + round(np.cos(alpha)) * movement_distance)
        pos_y = int(pos_y + round(np.sin(alpha)) * movement_distance)
        return pos_x, pos_y

    @staticmethod
    @numba.cuda.jit(target_backend='cuda', nopython=True)
    def __calculate_rotation_angle(angle, sensors_values, movement_rotation):
        multiply = np.random.rand()
        if sensors_values[0] == sensors_values[1] == sensors_values[2]:
            angle = angle + np.round((np.random.rand() * 2) - 1) * movement_rotation
        elif sensors_values[0] > sensors_values[2]:
            angle -= movement_rotation * multiply
        elif sensors_values[0] < sensors_values[2]:
            angle += movement_rotation * multiply
        return angle

    @staticmethod
    @numba.cuda.jit(target_backend='cuda', nopython=True)
    def __calculate_sensors_values(matrix, pos_x: int, pos_y: int, rot: int, sensor_size, s_angle, s_dis):
        sensors_values = [0, 0, 0]
        offset = int(sensor_size / 2)
        for sensor in range(3):
            alpha = np.deg2rad(270 - rot - (s_angle * (sensor - 1)))
            sensor_pos_x = pos_x + int(s_dis * np.cos(alpha))
            sensor_pos_y = pos_y + int(s_dis * np.sin(alpha))
            for x in range(sensor_size):
                for y in range(sensor_size):
                    sensors_values[sensor] += matrix[sensor_pos_y + y - offset][sensor_pos_x + x - offset]
        return sensors_values

    @staticmethod
    @numba.cuda.jit(target_backend='cuda', nopython=True)
    def __update_matrix(matrix, cells):
        for i in range(len(cells[0])):
            matrix[cells[1][i]][cells[0][i]] = 255


    @staticmethod
    @numba.cuda.jit(device=True)
    def __generate_cells(cells_amount, amount, sim_res_x, sim_res_y, circle_radius):
        cells = numba.cuda.device_array((3, amount), dtype=numba.int32)

        theta = np.random.uniform(0, 2 * np.pi, amount)
        radius = np.random.uniform(0, circle_radius, amount)
        x = radius * np.cos(theta) + (sim_res_x / 2)
        y = radius * np.sin(theta) + (sim_res_y / 2)
        cells[0] = x
        cells[1] = y
        for i in range(len(cells[2])):
            cells[2][i] = np.round(np.random.rand() * 360)

        cells_amount += amount

        return cells, cells_amount
