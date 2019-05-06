class LensParamaters:
    def __init__(self, data):
        self.principal_point = data["principalPoint"]
        self.focal_length = data["focalLength"]
        self.distortion_tangential = data["distortionTangential"]
        self.distortion_radial = data["distortionRadial"]

    def __repr__(self) -> str:
        return f"LensParamaters(" \
            f"principal_point={self.principal_point!r}, " \
            f"focal_length={self.focal_length!r}, " \
            f"distortion_tangential={self.distortion_tangential!r}, " \
            f"distortion_radial={self.distortion_radial!r}" \
            f")"
