import numpy.random
from numba import jit, cuda
from numba.experimental import jitclass
import numpy as np
import numba
from numba import int32, float32, int16
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import os
import sys
from config import DataAccessor

class Renderer:
    def __init__(self):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.set_cmap('bone')
        self.im = ax.imshow(data, animated=True)
        self.fps_val = ax.text(0.05, 1.05, "", transform=ax.transAxes, ha="center", fontsize=15)
        self.fps_val.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
        ax.text(0.15, 1.05, "FPS", transform=ax.transAxes, ha="center", fontsize=15)
        self.title = ax.text(1, 1.05, "", transform=ax.transAxes, ha="right")

        ani = animation.FuncAnimation(fig, self.update_image, interval=1)
        plt.show()

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        os.system(
            f"ffmpeg -framerate 30 -i sim_{self.output_dir}/frame_%05d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p sim_{self.output_dir}/sim.mp4")
        sys.exit(0)


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

        self.matrix = np.zeros(shape=(self.simulation_resolution_y, self.simulation_resolution_x), dtype=np.uint8)
        self.cells_array = self.__generate_cells()

        """
        row 0 is cells x pos
        row 1 is cells y pos
        row 2 is cells rot
        """




    def update_matrix(self):
        for i in range(len(self.cells_array[0])):
            self.matrix[self.cells_array[0][i]][self.cells_array[0][i]] = 255
    def __generate_cells(self):
        cells = np.zeros(shape=(3, self.initial_cells_amount), dtype=np.int32)

        theta = np.random.uniform(0, 2 * np.pi, self.initial_cells_amount)
        radius = np.random.uniform(0, self.initial_circle_radius, self.initial_cells_amount)
        x = radius * np.cos(theta) + (self.simulation_resolution_x / 2)
        y = radius * np.sin(theta) + (self.simulation_resolution_y / 2)
        cells[0] = x
        cells[1] = y
        for i in range(len(cells[2])):
            cells[2][i] = np.round(numpy.random.rand() * 360)
        return cells




def load_class(__data_accessor):
    __simulation_resolution_x = data_accessor.get_parameter("render_settings", "simulation_resolution_x")
    __simulation_resolution_y = data_accessor.get_parameter("render_settings", "simulation_resolution_y")

    __initial_circle_radius = data_accessor.get_parameter("initial_conditions", "initial_circle_radius")
    __initial_cells_amount = data_accessor.get_parameter("initial_conditions", "initial_cells_amount")
    __cells_spawn_rate = data_accessor.get_parameter("initial_conditions", "cells_spawn_rate")
    __trail_decay_factor = data_accessor.get_parameter("initial_conditions", "trail_decay_factor")
    __trail_evaporation_factor = data_accessor.get_parameter("initial_conditions", "trail_evaporation_factor")

    __sensors_distance = data_accessor.get_parameter("cell_settings", "sensors_distance")
    __sensors_size = data_accessor.get_parameter("cell_settings", "sensors_size")
    __sensors_angle_span = data_accessor.get_parameter("cell_settings", "sensors_angle_span")
    __movement_distance = data_accessor.get_parameter("cell_settings", "movement_distance")
    __movement_rotation = data_accessor.get_parameter("cell_settings", "movement_rotation")

    __physarum = Physarium(__simulation_resolution_x,
                           __simulation_resolution_y,
                           __initial_circle_radius,
                           __initial_cells_amount,
                           __cells_spawn_rate,
                           __trail_decay_factor,
                           __trail_evaporation_factor,
                           __sensors_distance,
                           __sensors_size,
                           __sensors_angle_span,
                           __movement_distance,
                           __movement_rotation
                           )
    return __physarum


if __name__ == "__main__":
    data_accessor = DataAccessor('config.json')
    physarum = load_class(data_accessor)

    print(physarum.cells_array)
