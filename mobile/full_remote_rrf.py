import logging
import struct
import time
from typing import Optional

import ndsi
import zstd
from ndsi.sensor import Sensor, NotDataSubSupportedError

from camera_models import load_intrinsics
from picoflexx import utils
from picoflexx.common import PicoflexxCommon
from picoflexx.frames import IRFrame, DepthFrame
from picoflexx.mobile import rrf_ndsi_spec
from video_capture import Base_Source, Base_Manager, manager_classes

# Patch ZreMsg._put_number1(n) as number1 SHOULD be unsigned
utils.monkeypatch_zre_msg_number1()
utils.monkeypatch_pyre_peer_status()
logger = logging.getLogger(__name__)


class Full_Remote_RRF_Source(PicoflexxCommon, Base_Source):
    sensor: Optional[Sensor]

    def __init__(
            self,
            g_pool,
            frame_size,
            frame_rate,
            network=None,
            source_id=None,
            host_name=None,
            sensor_name=None,
            **kwargs,
    ):
        super().__init__(g_pool, **kwargs)
        self.sensor = None
        self._source_id = source_id
        self._sensor_name = sensor_name
        self._host_name = host_name
        self._frame_size = frame_size
        self._frame_rate = frame_rate
        self.has_ui = False
        self.control_id_ui_mapping = {}
        self.get_frame_timeout = 16  # ms
        self.ghost_mode_timeout = 10  # sec
        self._initial_refresh = True
        self.last_update = self.g_pool.get_timestamp()
        self.record_pointcloud = False

        if not network:
            logger.debug(
                "No network reference provided. Capture is started "
                + "in ghost mode. No images will be supplied."
            )
            return

        self.recover(network)

        if not self.sensor or not self.sensor.supports_data_subscription:
            logger.error(
                "Init failed. Capture is started in ghost mode. "
                + "No images will be supplied."
            )
            self.cleanup()

        logger.debug("NDSI Source Sensor: %s" % self.sensor)

    def recover(self, network):
        logger.debug(
            "Trying to recover with %s, %s, %s"
            % (self._source_id, self._sensor_name, self._host_name)
        )
        if self._source_id:
            try:
                # uuid given
                self.sensor = network.sensor(
                    self._source_id, callbacks=(self.on_notification,)
                )
            except ValueError:
                pass

        if self.online:
            self._sensor_name = self.sensor.name
            self._host_name = self.sensor.host_name
            return
        if self._host_name and self._sensor_name:
            for sensor in network.sensors.values():
                if (
                        sensor["host_name"] == self._host_name
                        and sensor["sensor_name"] == self._sensor_name
                ):
                    self.sensor = network.sensor(
                        sensor["sensor_uuid"], callbacks=(self.on_notification,)
                    )
                    if self.online:
                        self._sensor_name = self.sensor.name
                        self._host_name = self.sensor.host_name
                        break
        else:
            for s_id in network.sensors:
                self.sensor = network.sensor(s_id, callbacks=(self.on_notification,))
                if self.online:
                    self._sensor_name = self.sensor.name
                    self._host_name = self.sensor.host_name
                    break

    @property
    def name(self):
        return "{}".format(self._sensor_name)

    @property
    def host(self):
        return "{}".format(self._host_name)

    @property
    def online(self):
        return bool(self.sensor)

    def poll_notifications(self):
        while self.sensor.has_notifications:
            self.sensor.handle_notification()

    def get_newest_data_frame(self, timeout=None):
        if not self.sensor.supports_data_subscription:
            raise NotDataSubSupportedError()

        if self.sensor.data_sub.poll(timeout=timeout):
            print("Poll")
            while self.sensor.has_data:
                data_msg = self.sensor.get_data(copy=False)
                meta_data = struct.unpack("<LLLLdLL", data_msg[1])
                print(meta_data)

                decmp = zstd.decompress(data_msg[2])

                return meta_data, rrf_ndsi_spec.decode_frame(meta_data[0], decmp)
        else:
            raise ndsi.StreamError('Operation timed out.')

    def recent_events(self, events):
        if self.online:
            self.poll_notifications()

            frame, depth_frame = None, None
            try:
                # data = self.sensor.get_data()
                # print(data)
                meta, data = self.get_newest_data_frame(
                    timeout=self.get_frame_timeout
                )
                print(meta)
                flags, w, h, idx, timestamp = meta[:5]
                frame = IRFrame({
                    "data": data["ir"],
                    "timestamp": timestamp,
                    "width": w,
                    "height": h,
                    "exposure_times": [0, 0, 0],
                })
                frame.index = idx

                depth_frame = DepthFrame({
                    "_data": data,
                    "timestamp": timestamp,
                    "width": w,
                    "height": h,
                    "exposure_times": [0, 0, 0],
                })
                # raise ndsi.StreamError("nop")
            except ndsi.StreamError:
                frame = None
            except Exception:
                frame = None
                import traceback

                logger.error(traceback.format_exc())

            if frame is not None:
                self._frame_size = (frame.width, frame.height)
                self.last_update = self.g_pool.get_timestamp()
                self._recent_frame = events["frame"] = frame

                if depth_frame is not None:
                    self._recent_depth_frame = events["depth_frame"] = depth_frame
            elif (
                    self.g_pool.get_timestamp() - self.last_update > self.ghost_mode_timeout
            ):
                logger.info("Entering gost mode")
                if self.online:
                    self.sensor.unlink()
                self.sensor = None
                self._source_id = None
                self._initial_refresh = True
                self.update_control_menu()
                self.last_update = self.g_pool.get_timestamp()
        else:
            time.sleep(self.get_frame_timeout / 1e3)

    # remote notifications
    def on_notification(self, sensor, event):
        # should only called if sensor was created
        if self._initial_refresh:
            self.sensor.set_control_value("streaming", True)
            self.sensor.refresh_controls()
            self._initial_refresh = False
        if event["subject"] == "error":
            # if not event['error_str'].startswith('err=-3'):
            logger.warning("Error {}".format(event["error_str"]))
            if "control_id" in event and event["control_id"] in self.sensor.controls:
                logger.debug(str(self.sensor.controls[event["control_id"]]))
        elif self.has_ui and (
                event["control_id"] not in self.control_id_ui_mapping
                or event["changes"].get("dtype") == "strmapping"
                or event["changes"].get("dtype") == "intmapping"
        ):
            self.update_control_menu()

    # local notifications
    def on_notify(self, notification):
        subject = notification["subject"]
        if subject.startswith("remote_recording.") and self.online:
            if "should_start" in subject and self.online:
                session_name = notification["session_name"]
                self.sensor.set_control_value("capture_session_name", session_name)
                self.sensor.set_control_value("local_capture", True)
            elif "should_stop" in subject:
                self.sensor.set_control_value("local_capture", False)

    @property
    def intrinsics(self):
        if self._intrinsics is None or self._intrinsics.resolution != self.frame_size:
            self._intrinsics = load_intrinsics(
                self.g_pool.user_dir, self.name, self.frame_size
            )
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, model):
        self._intrinsics = model

    @property
    def frame_size(self):
        return self._frame_size

    @property
    def frame_rate(self):
        if self.online:
            # FIXME: Hacky way to calculate frame rate. Depends on control option's caption
            fr_ctrl = self.sensor.controls.get("CAM_FRAME_RATE_CONTROL")
            if fr_ctrl:
                current_fr = fr_ctrl.get("value")
                map_ = {
                    mapping["value"]: mapping["caption"]
                    for mapping in fr_ctrl.get("map", [])
                }
                current_fr_cap = map_[current_fr].replace("Hz", "").strip()
                return float(current_fr_cap)

        return self._frame_rate

    @property
    def jpeg_support(self):
        return isinstance(self._recent_frame, ndsi.frame.JPEGFrame)

    def get_init_dict(self):
        settings = super().get_init_dict()
        settings["frame_rate"] = self.frame_rate
        settings["frame_size"] = self.frame_size
        if self.online:
            settings["sensor_name"] = self.sensor.name
            settings["host_name"] = self.sensor.host_name
        else:
            settings["sensor_name"] = self._sensor_name
            settings["host_name"] = self._host_name
        return settings

    def init_ui(self):
        self.add_menu()
        self.menu.label = "NDSI Source: {} @ {}".format(
            self._sensor_name, self._host_name
        )

        self.has_ui = True
        self.update_control_menu()

    def deinit_ui(self):
        self.remove_menu()
        self.has_ui = False

    def add_controls_to_menu(self, menu, controls):
        from pyglui import ui

        # closure factory
        def make_value_change_fn(ctrl_id):
            def initiate_value_change(val):
                logger.debug("{}: {} >> {}".format(self.sensor, ctrl_id, val))
                self.sensor.set_control_value(ctrl_id, val)

            return initiate_value_change

        for ctrl_id, ctrl_dict in controls:
            try:
                dtype = ctrl_dict["dtype"]
                ctrl_ui = None
                if dtype == "string":
                    ctrl_ui = ui.Text_Input(
                        "value",
                        ctrl_dict,
                        label=ctrl_dict["caption"],
                        setter=make_value_change_fn(ctrl_id),
                    )
                elif dtype == "integer" or dtype == "float":
                    convert_fn = int if dtype == "integer" else float
                    ctrl_ui = ui.Slider(
                        "value",
                        ctrl_dict,
                        label=ctrl_dict["caption"],
                        min=convert_fn(ctrl_dict.get("min", 0)),
                        max=convert_fn(ctrl_dict.get("max", 100)),
                        step=convert_fn(ctrl_dict.get("res", 0.0)),
                        setter=make_value_change_fn(ctrl_id),
                    )
                elif dtype == "bool":
                    ctrl_ui = ui.Switch(
                        "value",
                        ctrl_dict,
                        label=ctrl_dict["caption"],
                        on_val=ctrl_dict.get("max", True),
                        off_val=ctrl_dict.get("min", False),
                        setter=make_value_change_fn(ctrl_id),
                    )
                elif dtype == "strmapping" or dtype == "intmapping":
                    desc_list = ctrl_dict["map"]
                    labels = [desc["caption"] for desc in desc_list]
                    selection = [desc["value"] for desc in desc_list]
                    ctrl_ui = ui.Selector(
                        "value",
                        ctrl_dict,
                        label=ctrl_dict["caption"],
                        labels=labels,
                        selection=selection,
                        setter=make_value_change_fn(ctrl_id),
                    )
                if ctrl_ui:
                    ctrl_ui.read_only = ctrl_dict.get("readonly", False)
                    self.control_id_ui_mapping[ctrl_id] = ctrl_ui
                    menu.append(ctrl_ui)
                else:
                    logger.error("Did not generate UI for {}".format(ctrl_id))
            except:
                logger.error("Exception for control:\n{}".format(ctrl_dict))
                import traceback as tb

                tb.print_exc()
        if len(menu) == 0:
            menu.append(ui.Info_Text("No {} settings found".format(menu.label)))
        return menu

    def update_control_menu(self):
        if not self.has_ui:
            return
        from pyglui import ui

        del self.menu[:]
        self.control_id_ui_mapping = {}
        if not self.sensor:
            self.menu.append(
                ui.Info_Text(
                    ("Sensor %s @ %s not available. " + "Running in ghost mode.")
                    % (self._sensor_name, self._host_name)
                )
            )
            return

        other_controls = []
        for entry in iter(sorted(self.sensor.controls.items())):
            other_controls.append(entry)

        self.add_controls_to_menu(self.menu, other_controls)

        self.menu.append(
            ui.Button("Reset to default values", self.sensor.reset_all_control_values)
        )

    def cleanup(self):
        if self.online:
            self.sensor.unlink()
        self.sensor = None


class Full_Remote_RRF_Manager(Base_Manager):
    gui_name = "Remote RRF (Full)"
    group = 'pupil-picoflexx-v1'

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.network = ndsi.Network(callbacks=(self.on_event,))
        self.network.start()
        self._switch_network_group()
        self.selected_host = None
        self._recover_in = 3
        self._rejoin_in = 400
        logger.warning("Make sure the `time_sync` plugin is loaded!")

    def _switch_network_group(self):
        self.network.pyre_node.leave(self.group)
        self.network.pyre_node.leave(self.network.group)
        self.network.pyre_node.join(self.group)

    def cleanup(self):
        self.network.pyre_node.leave(self.network.group)
        self.network.stop()

    def init_ui(self):
        self.add_menu()
        self.re_build_ndsi_menu()

    def deinit_ui(self):
        self.remove_menu()

    def re_build_ndsi_menu(self):
        del self.menu[1:]
        from pyglui import ui

        ui_elements = []
        ui_elements.append(ui.Info_Text("Remote Pupil Mobile sources"))

        def host_selection_list():
            devices = {
                s["host_uuid"]: s["host_name"]  # removes duplicates
                for s in self.network.sensors.values()
            }
            devices = [pair for pair in devices.items()]  # create tuples
            # split tuples into 2 lists
            return zip(*(devices or [(None, "No hosts found")]))

        def view_host(host_uuid):
            if self.selected_host != host_uuid:
                self.selected_host = host_uuid
                self.re_build_ndsi_menu()

        host_sel, host_sel_labels = host_selection_list()
        ui_elements.append(
            ui.Selector(
                "selected_host",
                self,
                selection=host_sel,
                labels=host_sel_labels,
                setter=view_host,
                label="Remote host",
            )
        )

        self.menu.extend(ui_elements)
        if not self.selected_host:
            return
        ui_elements = []

        host_menu = ui.Growing_Menu("Remote Host Information")
        ui_elements.append(host_menu)

        def source_selection_list():
            default = (None, "Select to activate")
            # self.poll_events()
            sources = [default] + [
                (s["sensor_uuid"], s["sensor_name"])
                for s in self.network.sensors.values()
                if (
                        s["sensor_type"] == "royale_full" and s["host_uuid"] == self.selected_host
                )
            ]
            return zip(*sources)

        def activate(source_uid):
            if not source_uid:
                return
            settings = {
                "frame_size": self.g_pool.capture.frame_size,
                "frame_rate": self.g_pool.capture.frame_rate,
                "source_id": source_uid,
            }
            self.activate_source(settings)

        src_sel, src_sel_labels = source_selection_list()
        host_menu.append(
            ui.Selector(
                "selected_source",
                selection=src_sel,
                labels=src_sel_labels,
                getter=lambda: None,
                setter=activate,
                label="Source",
            )
        )

        self.menu.extend(ui_elements)

    def poll_events(self):
        while self.network.has_events:
            self.network.handle_event()

    def recent_events(self, events):
        self.poll_events()

        if (
                isinstance(self.g_pool.capture, Full_Remote_RRF_Source)
                and not self.g_pool.capture.sensor
        ):
            if self._recover_in <= 0:
                self.recover()
                self._recover_in = int(2 * 1e3 / self.g_pool.capture.get_frame_timeout)
            else:
                self._recover_in -= 1

            if self._rejoin_in <= 0:
                logger.debug("Rejoining network...")
                self.network.rejoin()
                self._switch_network_group()
                self._rejoin_in = int(10 * 1e3 / self.g_pool.capture.get_frame_timeout)
            else:
                self._rejoin_in -= 1

    def on_event(self, caller, event):
        if event["subject"] == "detach":
            logger.debug("detached: %s" % event)
            sensors = [s for s in self.network.sensors.values()]
            if self.selected_host == event["host_uuid"]:
                if sensors:
                    self.selected_host = sensors[0]["host_uuid"]
                else:
                    self.selected_host = None
                self.re_build_ndsi_menu()

        elif event["subject"] == "attach":
            if event["sensor_type"] == "video":
                logger.debug("attached: {}".format(event))
                self.notify_all({"subject": "capture_manager.source_found"})
            if not self.selected_host:
                self.selected_host = event["host_uuid"]
            self.re_build_ndsi_menu()

    def activate_source(self, settings={}):
        settings["network"] = self.network
        if hasattr(self.g_pool, "plugins"):
            self.g_pool.plugins.add(Full_Remote_RRF_Source, args=settings)
        else:
            self.g_pool.replace_source(Full_Remote_RRF_Source.__name__, source_settings=settings)

    def recover(self):
        self.g_pool.capture.recover(self.network)
        self._switch_network_group()

    def on_notify(self, n):
        """Provides UI for the capture selection

        Reacts to notification:
            ``capture_manager.source_found``: Check if recovery is possible

        Emmits notifications:
            ``capture_manager.source_found``
        """
        if (
                n["subject"].startswith("capture_manager.source_found")
                and isinstance(self.g_pool.capture, Full_Remote_RRF_Source)
                and not self.g_pool.capture.sensor
        ):
            self.recover()
        elif n["subject"] == "ndsi_recover" and self.g_pool.capture.class_name == "Full_Remote_RRF_Source":
            self.recover()


manager_classes.append(Full_Remote_RRF_Manager)
