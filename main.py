import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib import animation
import shutil
import os
import sys
from datetime import datetime
from config import DataAccessor
import time
import signal

from simulation import Physarium


class Renderer:
    def __init__(self, simulation: Physarium):

        signal.signal(signal.SIGINT, self.signal_handler)

        #self.output_dir = self.manage_output()

        self.simulation = simulation

        self.im = None

    def simulate(self):
        fig = plt.figure()
        self.ax = fig.add_subplot(1, 1, 1)
        self.im = self.ax.imshow(self.simulation.get_matrix(), animated=True, interpolation='nearest', cmap="bone")
        self.fps_val = self.ax.text(0.05, 1.05, "", transform=self.ax.transAxes, ha="center", fontsize=15)
        self.fps_val.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
        self.ax.text(0.15, 1.05, "FPS", transform=self.ax.transAxes, ha="center", fontsize=15)
        self.title = self.ax.text(1, 1.05, "", transform=self.ax.transAxes, ha="right")
        ani = animation.FuncAnimation(fig, self.update_image, interval=1)
        plt.show()

    def update_image(self, i):
        start = time.time()
        self.simulation.iterate()
        print(f"{round(time.time() - start, 2)}s")
        self.im.set_array(self.simulation.get_matrix())
        #self.img.save(f'sim_{self.output_dir}/frame_{str(i).zfill(5)}.png')

        fps = int(1.0 / (time.time() - start))

        if fps < 10:
            color = "red"
        elif fps < 30:
            color = "yellow"
        else:
            color = "green"

        self.fps_val.set_text(f"{fps}")
        self.fps_val.set_color(color)
        self.title.set_text(f"frame: {i}  |  cells count: {self.simulation.get_cells_amount()}")
        print(f"frame: {i}  cells:{self.simulation.get_cells_amount()}")

    @staticmethod
    def manage_output():
        output_dir = datetime.now().strftime("%Y%m%d%H%M%S")
        try:
            shutil.rmtree(f'sim_{output_dir}')
        except Exception:  # noqa
            pass
        try:
            os.mkdir(f'sim_{output_dir}')
        except Exception:  # noqa
            pass
        return output_dir

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        os.system(
            f"ffmpeg -framerate 30 -i sim_{self.output_dir}/frame_%05d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p sim_{self.output_dir}/sim.mp4")
        sys.exit(0)


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
    renderer = Renderer(physarum)
    renderer.simulate()

