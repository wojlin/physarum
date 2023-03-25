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

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
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

    def __cmap(self, __low_threshold,
               __high_threshold,
               __low_red,
               __low_green,
               __low_blue,
               __high_red,
               __high_green,
               __high_blue):

        __low_threshold = __low_threshold / 100
        __high_threshold = __high_threshold / 100

        __low_red = __low_red / 100
        __low_green = __low_green / 100
        __low_blue = __low_blue / 100

        __high_red = __high_red / 100
        __high_green = __high_green / 100
        __high_blue = __high_blue / 100

        _low_threshold = __high_threshold - 1 if __low_threshold >= __high_threshold else __low_threshold
        __high_threshold = __low_threshold + 1 if __high_threshold <= __low_threshold else __high_threshold

        __low_red = __high_red - 1 if __low_red >= __high_red else __low_red
        __low_green = __high_green - 1 if __low_green >= __high_green else __low_green
        __low_blue = __high_blue - 1 if __low_blue >= __high_blue else __low_blue

        __high_red = __low_red + 1 if __high_red <= __low_red else __high_red
        __high_green = __low_green + 1 if __high_green <= __low_green else __high_green
        __high_blue = __low_blue + 1 if __high_blue <= __low_blue else __high_blue


        __red_jump = (__high_threshold - __low_threshold) / (__high_red - __low_red)
        __green_jump = (__high_threshold - __low_threshold) / (__high_green - __low_green)
        __blue_jump = (__high_threshold - __low_threshold) / (__high_blue - __low_blue)


        color_dict = {
            'red': (
                (0.0, 0.0, 0.0),
                (__low_threshold, 0, 0),
                (__low_threshold, 0, 0),
                (__high_threshold, __red_jump, __red_jump),
                (1, __red_jump, __red_jump),
            ),
            'green': (
                (0.0, 0.0, 0.0),
                (__low_threshold, 0, 0),
                (__low_threshold, 0, 0),
                (__high_threshold, __green_jump, __green_jump),
                (1, __green_jump, __green_jump),
            ),
            'blue': (
                (0.0, 0.0, 0.0),
                (__low_threshold, 0, 0),
                (__low_threshold, 0, 0),
                (__high_threshold, __blue_jump, __blue_jump),
                (1, __blue_jump, __blue_jump),
            )
        }

        color_map = LinearSegmentedColormap("custom", color_dict)
        mpl.colormaps.unregister("custom")
        mpl.colormaps.register(color_map)


    def simulate(self):

        __low_threshold = data_accessor.get_parameter("color_settings", "low_threshold")
        __high_threshold = data_accessor.get_parameter("color_settings", "high_threshold")

        __low_red = data_accessor.get_parameter("color_settings", "low_red")
        __low_green = data_accessor.get_parameter("color_settings", "low_green")
        __low_blue = data_accessor.get_parameter("color_settings", "low_blue")

        __high_red = data_accessor.get_parameter("color_settings", "high_red")
        __high_green = data_accessor.get_parameter("color_settings", "high_green")
        __high_blue = data_accessor.get_parameter("color_settings", "high_blue")

        self.__cmap(__low_threshold, __high_threshold, __low_red, __low_green, __low_blue, __high_red, __high_green, __high_blue)

        fig = plt.figure(num="physarum simulation", figsize=(15, 8))
        self.ax = fig.add_subplot(1, 1, 1)
        plt.subplots_adjust(left=0.5, right=0.9, top=0.9, bottom=0.1)
        self.im = self.ax.imshow(self.simulation.get_matrix(), animated=True, interpolation='nearest', cmap="custom")
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(self.im, cax=cax, orientation="vertical")
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


        height = 0.04

        axes = plt.axes([0.2, 0.85, 0.15, 0.05])
        bnext = Button(axes, 'reset sim', color="red")
        bnext.on_clicked(self.restart)

        plt.figtext(0.2, 0.72, "color settings:")

        axfreq = fig.add_axes([0.2, 0.675, 0.15, height])
        self.low_threshold_slider = Slider(
            ax=axfreq,
            label='low threshold: ',
            valmin=0,
            valmax=100,
            valinit=__low_threshold,
            valstep=1
        )
        self.low_threshold_slider.on_changed(self.update_colormap)

        axfreq = fig.add_axes([0.2, 0.65, 0.15, height])
        self.high_threshold_slider = Slider(
            ax=axfreq,
            label='high threshold: ',
            valmin=0,
            valmax=100,
            valinit=__high_threshold,
            valstep=1
        )
        self.high_threshold_slider.on_changed(self.update_colormap)

        axfreq = fig.add_axes([0.2, 0.625, 0.15, height])
        self.low_red_slider = Slider(
            ax=axfreq,
            label='low red color: ',
            valmin=0,
            valmax=255,
            valinit=__low_red,
            valstep=1
        )
        self.low_red_slider.on_changed(self.update_colormap)

        axfreq = fig.add_axes([0.2, 0.6, 0.15, height])
        self.low_green_slider = Slider(
            ax=axfreq,
            label='low green color: ',
            valmin=0,
            valmax=255,
            valinit=__low_green,
            valstep=1
        )
        self.low_green_slider.on_changed(self.update_colormap)

        axfreq = fig.add_axes([0.2, 0.575, 0.15, height])
        self.low_blue_slider = Slider(
            ax=axfreq,
            label='low blue color: ',
            valmin=0,
            valmax=255,
            valinit=__low_blue,
            valstep=1
        )
        self.low_blue_slider.on_changed(self.update_colormap)

        axfreq = fig.add_axes([0.2, 0.55, 0.15, height])
        self.high_red_slider = Slider(
            ax=axfreq,
            label='high red color: ',
            valmin=0,
            valmax=255,
            valinit=__high_red,
            valstep=1
        )
        self.high_red_slider.on_changed(self.update_colormap)

        axfreq = fig.add_axes([0.2, 0.525, 0.15, height])
        self.high_green_slider = Slider(
            ax=axfreq,
            label='high green color: ',
            valmin=0,
            valmax=255,
            valinit=__high_green,
            valstep=1
        )
        self.high_green_slider.on_changed(self.update_colormap)

        axfreq = fig.add_axes([0.2, 0.5, 0.15, height])
        self.high_blue_slider = Slider(
            ax=axfreq,
            label='high blue color: ',
            valmin=0,
            valmax=255,
            valinit=__high_blue,
            valstep=1
        )
        self.high_blue_slider.on_changed(self.update_colormap)


        plt.figtext(0.2, 0.46, "render settings:")

        axfreq = fig.add_axes([0.2, 0.425, 0.15, height])
        self.simulation_resolution_x_slider = Slider(
            ax=axfreq,
            label='simulation resolution x: ',
            valmin=500,
            valmax=2000,
            valinit=__simulation_resolution_x,
            valstep=100
        )
        self.simulation_resolution_x_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.4, 0.15, height])
        self.simulation_resolution_y_slider = Slider(
            ax=axfreq,
            label='simulation resolution y: ',
            valmin=500,
            valmax=2000,
            valinit=__simulation_resolution_y,
            valstep=100
        )
        self.simulation_resolution_y_slider.on_changed(self.update)

        plt.figtext(0.2, 0.39, "initial conditions:")

        axfreq = fig.add_axes([0.2, 0.35, 0.15, height])
        self.initial_circle_radius_slider = Slider(
            ax=axfreq,
            label='initial circle radius: ',
            valmin=1,
            valmax=1000,
            valinit=__initial_circle_radius,
            valstep=10
        )
        self.initial_circle_radius_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.325, 0.15, height])
        self.initial_cells_amount_slider = Slider(
            ax=axfreq,
            label='initial cells amount: ',
            valmin=1000,
            valmax=10000000,
            valinit=__initial_cells_amount,
            valstep=1000
        )
        self.initial_cells_amount_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.3, 0.15, height])
        self.cells_spawn_rate_slider = Slider(
            ax=axfreq,
            label='cells spawn rate: ',
            valmin=0,
            valmax=1000,
            valinit=__cells_spawn_rate,
            valstep=10
        )
        self.cells_spawn_rate_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.275, 0.15, height])
        self.trail_decay_factor_slider = Slider(
            ax=axfreq,
            label='trail decay factor: ',
            valmin=1,
            valmax=50,
            valinit=__trail_decay_factor,
            valstep=1
        )
        self.trail_decay_factor_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.25, 0.15, height])
        self.trail_evaporation_factor_slider = Slider(
            ax=axfreq,
            label='trail evaporation factor: ',
            valmin=1,
            valmax=50,
            valinit=__trail_evaporation_factor,
            valstep=1
        )
        self.trail_evaporation_factor_slider.on_changed(self.update)

        plt.figtext(0.2, 0.24, "cell settings:")

        axfreq = fig.add_axes([0.2, 0.2, 0.15, height])
        self.sensors_distance_slider = Slider(
            ax=axfreq,
            label='sensors distance: ',
            valmin=1,
            valmax=50,
            valinit=__sensors_distance,
            valstep=1
        )
        self.sensors_distance_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.175, 0.15, height])
        self.sensors_size_slider = Slider(
            ax=axfreq,
            label='sensors size: ',
            valmin=1,
            valmax=10,
            valinit=__sensors_size,
            valstep=1
        )
        self.sensors_size_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.15, 0.15, height])
        self.sensors_angle_span_slider = Slider(
            ax=axfreq,
            label='sensors angle span: ',
            valmin=1,
            valmax=180,
            valinit=__sensors_angle_span,
            valstep=1
        )
        self.sensors_angle_span_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.125, 0.15, height])
        self.movement_distance_slider = Slider(
            ax=axfreq,
            label='movement distance: ',
            valmin=1,
            valmax=5,
            valinit=__movement_distance,
            valstep=1
        )
        self.movement_distance_slider.on_changed(self.update)

        axfreq = fig.add_axes([0.2, 0.1, 0.15, height])
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


    def update_colormap(self, val):
        __low_threshold = self.low_threshold_slider.val
        __high_threshold = self.high_threshold_slider.val
        __low_red = self.low_red_slider.val
        __low_green = self.low_green_slider.val
        __low_blue = self.low_blue_slider.val
        __high_red = self.high_red_slider.val
        __high_green = self.high_green_slider.val
        __high_blue = self.high_blue_slider.val

        self.__cmap(__low_threshold, __high_threshold, __low_red, __low_green, __low_blue, __high_red, __high_green,
                    __high_blue)

        self.im.set_cmap("custom")

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
