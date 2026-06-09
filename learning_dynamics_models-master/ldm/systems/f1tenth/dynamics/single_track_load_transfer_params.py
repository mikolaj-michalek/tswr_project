# by now the default ones are okay
from ldm.systems.car.dynamics.single_track_load_transfer_params import DefaultSingleTrackLoadTransferParameters


class F1TenthSingleTrackLoadTransferParameters(DefaultSingleTrackLoadTransferParameters):
    def __init__(self) -> None:
        super(F1TenthSingleTrackLoadTransferParameters, self).__init__()