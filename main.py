import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.widgets import Slider, Button
from matplotlib import animation
import shutil
import os
import sys
from datetime import datetime
from config import DataAccessor
from PIL import Image
import time
import signal

from simulation_cpu import Physarum as Physarum_CPU
from simulation_gpu import Physarum as Physarum_GPU
from interfaces import Physarum


class Renderer:
    def __init__(self, simulation: Physarum, data_accessor: DataAccessor):
        self.data_accessor = data_accessor
        signal.signal(signal.SIGINT, self.signal_handler)
        self.save = save = data_accessor.get_parameter("program_settings", "save_to_disk")
        if self.save:
            self.output_dir = self.manage_output()

        self.fps = data_accessor.get_parameter("program_settings", "record_fps")

        self.simulation = simulation
        self.current_iter = 0

        self.im = None

    def simulate(self):
        fig = plt.figure(figsize=(15, 8))
        self.ax = fig.add_subplot(1, 1, 1)
        plt.subplots_adjust(left=0.5, right=0.9, top=0.9, bottom=0.1)
        self.im = self.ax.imshow(self.simulation.get_matrix(), animated=True, interpolation='nearest', cmap="bone")
        self.fps_val = self.ax.text(0.05, 1.05, "", transform=self.ax.transAxes, ha="center", fontsize=15)
        self.fps_val.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
        self.ax.text(0.15, 1.05, "FPS", transform=self.ax.transAxes, ha="center", fontsize=15)
        self.title = self.ax.text(1, 1.05, "", transform=self.ax.transAxes, ha="right")

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

        axes = plt.axes([0.2, 0.85, 0.15, 0.05])
        bnext = Button(axes, 'reset sim', color="red")
        bnext.on_clicked(self.restart)

        plt.figtext(0.2, 0.8, "render settings:")

        axfreq = fig.add_axes([0.2, 0.75, 0.15, 0.05])
        self.simulation_resolution_x_slider = Slider(
            ax=axfreq,
            label='simulation resolution x: ',
            valmin=500,
            valmax=2000,
            valinit=__simulation_resolution_x,
            valstep=100
        )
        self.simulation_resolution_x_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.7, 0.15, 0.05])
        self.simulation_resolution_y_slider = Slider(
            ax=axfreq,
            label='simulation resolution y: ',
            valmin=500,
            valmax=2000,
            valinit=__simulation_resolution_y,
            valstep=100
        )
        self.simulation_resolution_y_slider.on_changed(self.update)

        plt.figtext(0.2, 0.65, "initial conditions:")

        axfreq = fig.add_axes([0.2, 0.6, 0.15, 0.05])
        self.initial_circle_radius_slider = Slider(
            ax=axfreq,
            label='initial circle radius: ',
            valmin=1,
            valmax=1000,
            valinit=__initial_circle_radius,
            valstep=10
        )
        self.initial_circle_radius_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.55, 0.15, 0.05])
        self.initial_cells_amount_slider = Slider(
            ax=axfreq,
            label='initial cells amount: ',
            valmin=1000,
            valmax=10000000,
            valinit=__initial_cells_amount,
            valstep=1000
        )
        self.initial_cells_amount_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.5, 0.15, 0.05])
        self.cells_spawn_rate_slider = Slider(
            ax=axfreq,
            label='cells spawn rate: ',
            valmin=0,
            valmax=100000,
            valinit=__cells_spawn_rate,
            valstep=1000
        )
        self.cells_spawn_rate_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.45, 0.15, 0.05])
        self.trail_decay_factor_slider = Slider(
            ax=axfreq,
            label='trail decay factor: ',
            valmin=1,
            valmax=50,
            valinit=__trail_decay_factor,
            valstep=1
        )
        self.trail_decay_factor_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.4, 0.15, 0.05])
        self.trail_evaporation_factor_slider = Slider(
            ax=axfreq,
            label='trail evaporation factor: ',
            valmin=1,
            valmax=50,
            valinit=__trail_evaporation_factor,
            valstep=1
        )
        self.trail_evaporation_factor_slider.on_changed(self.update)

        plt.figtext(0.2, 0.35, "cell settings:")

        axfreq = fig.add_axes([0.2, 0.3, 0.15, 0.05])
        self.sensors_distance_slider = Slider(
            ax=axfreq,
            label='sensors distance: ',
            valmin=1,
            valmax=50,
            valinit=__sensors_distance,
            valstep=1
        )
        self.sensors_distance_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.25, 0.15, 0.05])
        self.sensors_size_slider = Slider(
            ax=axfreq,
            label='sensors size: ',
            valmin=1,
            valmax=10,
            valinit=__sensors_size,
            valstep=1
        )
        self.sensors_size_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.2, 0.15, 0.05])
        self.sensors_angle_span_slider = Slider(
            ax=axfreq,
            label='sensors angle span: ',
            valmin=1,
            valmax=180,
            valinit=__sensors_angle_span,
            valstep=1
        )
        self.sensors_angle_span_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.15, 0.15, 0.05])
        self.movement_distance_slider = Slider(
            ax=axfreq,
            label='movement distance: ',
            valmin=1,
            valmax=5,
            valinit=__movement_distance,
            valstep=1
        )
        self.movement_distance_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.1, 0.15, 0.05])
        self.movement_rotation_slider = Slider(
            ax=axfreq,
            label='movement rotation: ',
            valmin=1,
            valmax=90,
            valinit=__movement_rotation,
            valstep=1
        )
        self.movement_rotation_slider.on_changed(self.update)



        ani = animation.FuncAnimation(fig, self.update_image, interval=1)
        plt.show()

    def restart(self, *args, **kwargs):
        self.simulation.restart()
    def update(self, val):

        self.set_parameter("simulation_resolution_x", int(self.simulation_resolution_x_slider.val))
        self.set_parameter("simulation_resolution_y", int(self.simulation_resolution_y_slider.val))

        self.set_parameter("initial_circle_radius", int(self.initial_circle_radius_slider.val))
        self.set_parameter("initial_cells_amount", int(self.initial_cells_amount_slider.val))
        self.set_parameter("cells_spawn_rate", int(self.cells_spawn_rate_slider.val))
        self.set_parameter("trail_decay_factor", int(self.trail_decay_factor_slider.val))
        self.set_parameter("trail_evaporation_factor", int(self.trail_evaporation_factor_slider.val))

        self.set_parameter("sensors_distance", int(self.sensors_distance_slider.val))
        self.set_parameter("sensors_size", int(self.sensors_size_slider.val))
        self.set_parameter("movement_distance", int(self.movement_distance_slider.val))
        self.set_parameter("movement_rotation", int(self.movement_rotation_slider.val))
        self.set_parameter("sensors_angle_span", int(self.sensors_angle_span_slider.val))
        self.simulation.iterate()

    def set_parameter(self, name, value):
        setattr(self.simulation, name, value)

    def update_image(self, i):
        self.current_iter += 1
        start = time.time()
        self.simulation.iterate()
        print(f"{round(time.time() - start, 2)}s")
        self.im.set_array(self.simulation.get_matrix())

        if self.save:
            im = Image.fromarray(self.simulation.get_matrix())
            im.save(f'sim_{self.output_dir}/frame_{str(self.current_iter).zfill(5)}.png')

        fps = int(1.0 / (time.time() - start))

        #if i == 100:
        #    self.set_parameter("movement_distance", 5)

        if fps < 10:
            color = "red"
        elif fps < 30:
            color = "yellow"
        else:
            color = "green"

        self.fps_val.set_text(f"{fps}")
        self.fps_val.set_color(color)
        self.title.set_text(f"frame: {self.current_iter}  |  cells count: {self.simulation.get_cells_amount()}")
        print(f"frame: {self.current_iter}  cells:{self.simulation.get_cells_amount()}")

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
            f"ffmpeg -framerate {self.fps} -i sim_{self.output_dir}/frame_%05d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p sim_{self.output_dir}/sim.mp4")
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

    __sim_type = data_accessor.get_parameter("program_settings", "simulation_type")
    if __sim_type == "cpu":
        __physarum = Physarum_CPU(__simulation_resolution_x,
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
    else:
        __physarum = Physarum_GPU(__simulation_resolution_x,
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
    renderer = Renderer(physarum, data_accessor)
    renderer.simulate()
