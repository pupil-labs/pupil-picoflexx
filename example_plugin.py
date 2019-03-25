"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from plugin import Plugin


class Example_Picoflexx_Plugin(Plugin):
    def recent_events(self, events):
        if "frame" in events:
            pass
            # IR image
            # frame = events["frame"]
            # frame.gray

        if "depth_frame" in events and hasattr(
            events["depth_frame"], "dense_pointcloud"
        ):
            pass
            # depth_frame = events["depth_frame"]

            # access point cloud, dimensions: (height*width) x 3, dtype: float [meter]
            # depth_frame.dense_pointcloud

            # access depth values only, dimensions: height x width, dtype: float [meter]
            # depth_frame.dense_pointcloud[:, 3].reshape(depth_frame.height, depth_frame.width)

            # cloud point confidence and noise, dimensions: height x width
            # depth_frame.noise  # dtype: float [meter]
            # depth_frame.depthConfidence  # dtype: uint8 [0: invalid, 255: full confidence]

            # 2d visualization using cumulative histogram mapping
            # dimensions: height x width x 3, dtype: uint8
            # depth_frame.img
