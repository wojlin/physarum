import numpy as np
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

        self.output_dir = self.manage_output()

        self.simulation = simulation

    def simulate(self):
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

    def update_image(self, i):
        start = time.time()

        #self.img = self.img.filter(ImageFilter.BoxBlur(config.DIS_BLUR))
        #self.img = ImageEnhance.Brightness(self.img).enhance(1 - config.DIS_EVAP)

        start = time.time()
        self.particles += self.circular_distribution(config.SPAWN_RATE, self.center_pos, 10)
        print(f"adding particles: {round(time.time() - start, 3)}s")

        start = time.time()
        for p in reversed(range(len(self.particles))):
            if type(self.particles[p]) == Particle:
                self.img.putpixel(self.particles[p].pos, 255)
        self.frame = i
        print(f"writing particles to image: {round(time.time() - start, 3)}s")

        self.img.save(f'sim_{self.output_dir}/frame_{str(i).zfill(5)}.png')

        start = time.time()
        npimg = np.asarray(self.img)
        for a, part in enumerate(self.particles):
            if not type(part) == Particle:
                continue
            if not part.process_particle(npimg):
                self.particles[a] = None

        print(f"processing particles: {round(time.time() - start, 3)}s")

        self.im.set_array(np.asarray(self.img))
        fps = int(1.0 / (time.time() - start))
        color = "white"
        if fps < 10:
            color = "red"
        elif fps < 20:
            color = "yellow"
        elif fps >= 20:
            color = "green"

        self.fps_val.set_text(f"{fps}")
        self.fps_val.set_color(color)
        self.title.set_text(f"frame: {i}  |  cells count: {len(self.particles)}")
        print(round(time.time() - start, 2))
        print(f"frame: {i}  cells:{len(self.particles)}")

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
    physarum.iterate()
    #renderer = Renderer(physarum)
    #renderer.simulate()

    print(physarum.cells_array)
