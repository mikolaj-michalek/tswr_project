class CAR:
    MIN_VELOCITY = 0.5   # [m/s]
    MAX_VELOCITY = 20.0  # [m/s]
    MAX_OMEGA_REF = 8.0  # [m/s]
    MAX_STEERING = 0.5  # [rad]
    MAX_OMEGA_DOT = 5.0  # [m/s^2]
    WIDTH = 0.3  # [m]
    LENGTH = 0.43  # [m]

class TRACK:
    MAX_WIDTH = 5.0  # [m]
    MAX_SIZE = 750  # [dm] Maximum track size in decimeters
    OFF_TRACK_DISTANCE = 0.0  # [m]
    DEFAULT_FRICTION = 0.8  # Default friction value for the track
    MAX_FRICTION = 1.0  # Maximum friction value for the track
    MIN_FRICTION = 0.5  # Minimum friction value for the track

class OBSERVATION:
    FORESIGHT_SIZE = 200  # Number of future track points to consider in the foresight window
    FORESIGHT_SPACING = 0.1  # Spacing between foresight points in meters
    HISTORY_SIZE = 100  # Number of past observations to keep in the history

class STATE:
    SIZE = 13  # Default state dimension for the simulator