from ldm.systems.car.dynamics.single_track_params import DefaultSingleTrackParameters

# by now the default ones are okay
class F1TenthSingleTrackParameters(DefaultSingleTrackParameters):
    def __init__(self) -> None:
        super(F1TenthSingleTrackParameters, self).__init__()