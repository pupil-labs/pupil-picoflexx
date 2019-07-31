import logging
from builtins import NotImplementedError
from typing import Optional, Tuple

import cv2
import numpy as np
from OpenGL.GL import *
from pyglui import ui
from pyglui.pyfontstash import fontstash

import gl_utils
import glfw
from methods import denormalize, normalize
from plugin import Plugin
from .frames import DepthFrame, IRFrame
from .utils import clamp

logger = logging.getLogger(__name__)


def indicators_for(near, far, width, ignore_clip: bool = False):
    INDICATOR_PER_PX = 50
    STEPS = (1.0, 0.75, 0.5, 0.25, 0.1, 0.05)

    max_indicators = width // INDICATOR_PER_PX + 1
    selected_step = 2
    distance_range = far - near

    # find the best step to use
    for p in reversed(STEPS):
        if distance_range / p < max_indicators:
            selected_step = p
            break

    yield near

    indicator_gap = distance_range / max_indicators  # min gap between indicator labels
    cur = (near // selected_step) * selected_step
    while True:
        cur += selected_step

        if cur >= far:
            break

        # enforce min label gap around the near and far indicators
        if not ignore_clip and (near + indicator_gap > cur or far - indicator_gap < cur):
            continue

        yield cur

    yield far


class PicoflexxCommon(Plugin):
    def __init__(self, g_pool, *args, **kwargs):
        super(PicoflexxCommon, self).__init__(g_pool)

        self.hue_near = kwargs.get('hue_near', 0.0)
        self.hue_far = kwargs.get('hue_far', 0.75)
        self.dist_near = kwargs.get('dist_near', 0.14)
        self.dist_far = kwargs.get('dist_far', 5.0)
        self.preview_true_depth = kwargs.get('preview_true_depth', False)
        self.preview_depth = kwargs.get('preview_depth', True)

        self._recent_depth_frame = None  # type: Optional[DepthFrame]
        self._recent_frame = None  # type: Optional[IRFrame]
        # capture doesn't provide this in g_pool like player does, main_window isn't present during export
        self._camera_render_size = glfw.glfwGetWindowSize(g_pool.main_window) if hasattr(g_pool, 'main_window') else None
        self.current_exposure = 0  # type: int

        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", ui.get_opensans_font_path())
        self.glfont.set_color_float((1.0, 1.0, 1.0, 0.8))
        self.glfont.set_align_string(v_align="center", h_align="top")

        self._colorbar_offs = (6, 30, 30, 30)
        self._colorbar_pos = [520, 12]
        self._colorbar_size = (300, 20)
        self._tex_id_color_bar = None
        self._current_opts = None
        self._mouse_drag_pos = None  # type: Optional[Tuple[int, int]]

    def on_window_resize(self, window, w, h):
        self._camera_render_size = w, h

    def _render_color_bar(self):
        if self.dist_far == self.dist_near:
            return

        if (self.hue_near, self.hue_far, self.dist_near, self.dist_far) != self._current_opts:
            self._generate_color_bar_texture(self._colorbar_size[0])

        x, y = self._colorbar_pos
        w, h = self._colorbar_size
        offs = self._colorbar_offs

        glColor4f(0, 0, 0, 0.5)
        glRecti(x - offs[2], y - offs[0], x + w + offs[3], y + h + offs[1])
        glColor3f(1, 1, 1)

        try:
            if self._tex_id_color_bar is None:
                self._generate_color_bar_texture()

            glBindTexture(GL_TEXTURE_1D, self._tex_id_color_bar)

            glBegin(GL_QUADS)
            glTexCoord1i(0)
            glVertex2i(x, y)
            glVertex2i(x, y + h)
            glTexCoord1i(1)
            glVertex2i(x + w, y + h)
            glVertex2i(x + w, y)
            glEnd()
        finally:
            glBindTexture(GL_TEXTURE_1D, 0)  # Ensure we unbind the 1D texture

        gap, marker_h = 10, 5
        glTranslate(x, y + h + gap, 0)
        glLineWidth(1)
        glColor3f(1, 1, 1)

        glBegin(GL_LINE_STRIP)
        glVertex2i(0, 0)
        glVertex2i(w, 0)
        glEnd()

        near, far = self.dist_near, self.dist_far
        dist_scale = w / (far - near)

        glBegin(GL_LINES)
        for i in indicators_for(near, far, self._colorbar_size[0], ignore_clip=True):
            marker_x = round((i - near) * dist_scale)
            glVertex2i(marker_x, 0)
            glVertex2i(marker_x, -marker_h)
        glEnd()

        def draw_indicator(dist: float, geq: bool = False, leq: bool = False):
            marker_x = int((dist - near) * dist_scale)

            if dist == int(dist):
                dist = str(int(dist))
            else:
                dist = '{:.2f}'.format(dist)

            text = '{}{}m'.format('\u2264 ' if leq else '\u2265 ' if geq else '', dist)
            self.glfont.draw_text(marker_x, 0, text)

        self.glfont.set_size(18)

        for i in indicators_for(near, far, self._colorbar_size[0]):
            draw_indicator(i, leq=near == i, geq=far == i)

        glTranslate(-x, -(y + h + gap), 0)

    def camera_to_screen_coords(self, pos: Tuple) -> Tuple[int, int]:
        pos = denormalize(pos, self._camera_render_size)
        pos = normalize(pos, self.g_pool.capture.frame_size)
        return int(pos[0]), int(pos[1])

    @property
    def _colorbar_bounds(self):
        u, d, l, r = self._colorbar_offs
        x, y = self._colorbar_pos
        w, h = self._colorbar_size

        return x - l, y - u, x + w + r, y + h + d

    def on_click(self, pos, button, action):
        super().on_click(pos, button, action)

        if self.preview_depth:
            pos = self.camera_to_screen_coords(pos)
            x1, y1, x2, y2 = self._colorbar_bounds

            if x1 <= pos[0] <= x2 and y1 <= pos[1] <= y2:
                if button == glfw.GLFW_MOUSE_BUTTON_LEFT and action == glfw.GLFW_PRESS:
                    self._mouse_drag_pos = pos

        if button == glfw.GLFW_MOUSE_BUTTON_LEFT and action == glfw.GLFW_RELEASE:
            self._mouse_drag_pos = None

    def on_pos(self, pos):
        super().on_pos(pos)

        pos = self.camera_to_screen_coords(pos)
        w, h = self._camera_render_size

        if self._mouse_drag_pos is not None:
            new_x = self._colorbar_pos[0] + pos[0] - self._mouse_drag_pos[0]
            new_y = self._colorbar_pos[1] + pos[1] - self._mouse_drag_pos[1]

            self._colorbar_pos[0] = clamp(self._colorbar_offs[2], new_x, w - self._colorbar_size[0] - self._colorbar_offs[3])
            self._colorbar_pos[1] = clamp(self._colorbar_offs[0], new_y, h - self._colorbar_size[1] - self._colorbar_offs[1])

            self._mouse_drag_pos = pos

    def deinit_ui(self):
        super().deinit_ui()

        if self.menu is not None:
            self.remove_menu()

        if self._tex_id_color_bar is not None:
            glDeleteTextures([self._tex_id_color_bar])
            self._tex_id_color_bar = None

    @property
    def recent_frame(self) -> Optional[IRFrame]:
        return self._recent_frame

    @property
    def recent_depth_frame(self) -> Optional[DepthFrame]:
        return self._recent_depth_frame

    @property
    def frame_size(self):
        return (
            (self.recent_frame.width, self.recent_frame.height)
            if self.recent_frame
            else (1280, 720)
        )

    def gl_display(self):
        if self.online:
            if self.preview_depth and self.recent_depth_frame is not None:
                self.g_pool.image_tex.update_from_ndarray(self.recent_depth_frame.get_color_mapped(
                    self.hue_near, self.hue_far, self.dist_near, self.dist_far, self.preview_true_depth
                ))
            elif self.recent_frame is not None:
                self.g_pool.image_tex.update_from_ndarray(self.recent_frame.img)
            gl_utils.glFlush()
            gl_utils.make_coord_system_norm_based()
            self.g_pool.image_tex.draw(interpolation=False)
        else:
            super().gl_display()

        if self.preview_depth:
            gl_utils.adjust_gl_view(*self._camera_render_size)
            self._render_color_bar()

        gl_utils.make_coord_system_pixel_based(
            (self.frame_size[1], self.frame_size[0], 3)
        )

    @property
    def online(self):
        raise NotImplementedError()

    def get_init_dict(self):
        return {
            "preview_depth": self.preview_depth,
            "hue_near": self.hue_near,
            "hue_far": self.hue_far,
            "dist_near": self.dist_near,
            "dist_far": self.dist_far,
            "preview_true_depth": self.preview_true_depth,
        }

    def append_depth_preview_menu(self):
        text = ui.Info_Text(
            "Enabling Preview Depth will display a colourised version of the data "
            "based on the depth. Disabling the option will display the "
            "according IR image." +
            ("Independent of which option is selected, the IR image stream will "
             "be stored to `world.mp4` during a recording."
             if self.g_pool.app == 'capture' else "")
        )
        self.menu.append(text)
        self.menu.append(ui.Switch("preview_depth", self, label="Preview Depth"))

        depth_preview_menu = ui.Growing_Menu("Depth preview settings")
        depth_preview_menu.collapsed = True
        depth_preview_menu.append(
            ui.Info_Text("Set hue and distance ranges for the depth preview.")
        )
        depth_preview_menu.append(
            ui.Slider("hue_near", self, min=0.0, max=1.0, label="Near Hue")
        )
        depth_preview_menu.append(
            ui.Slider("hue_far", self, min=0.0, max=1.0, label="Far Hue")
        )
        depth_preview_menu.append(ui.Button("Fit distance (15th and 85th percentile)", lambda: self._fit_distance()))
        depth_preview_menu.append(
            ui.Slider("dist_near", self, min=0.0, max=4.8, label="Near Distance (m)")
        )
        depth_preview_menu.append(
            ui.Slider("dist_far", self, min=0.2, max=5.0, label="Far Distance (m)")
        )
        depth_preview_menu.append(ui.Switch("preview_true_depth", self, label="Preview using linalg distance"))

        self.menu.append(depth_preview_menu)

    def _fit_distance(self):
        if not self.recent_depth_frame:
            logger.warning("No recent frame, can't fit hue.")
            return

        if self.preview_true_depth:
            depth_data = self.recent_depth_frame.true_depth
        else:
            depth_data = self.recent_depth_frame._data.z

        near, far = np.percentile(depth_data, (15, 85))
        self.dist_near, self.dist_far = float(near), float(far)

    def _generate_color_bar_texture(self, width: int = 300):
        glEnable(GL_TEXTURE_1D)

        if self._tex_id_color_bar is None:
            self._tex_id_color_bar = glGenTextures(1)

        near, far = int(self.hue_near * 360), int(self.hue_far * 360)

        if near == far:
            return

        glBindTexture(GL_TEXTURE_1D, self._tex_id_color_bar)
        hues = np.arange(near, far, (far - near) / width, dtype=np.float32).reshape((1, width))
        hues = hues[:, :, np.newaxis]

        sv = np.ones(hues.shape, dtype=np.float32)  # saturation/value
        hsv = np.concatenate((hues, sv, sv), axis=2)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        image *= 255
        image = image.astype(np.uint8)

        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB8, width, 0, GL_RGB, GL_UNSIGNED_BYTE, image)

        self._current_opts = (self.hue_near, self.hue_far, self.dist_near, self.dist_far)
