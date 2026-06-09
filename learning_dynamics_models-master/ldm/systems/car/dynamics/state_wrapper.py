from collections import namedtuple

WX = namedtuple('WX', ['v_x', 'v_y', 'r',
                       'friction_front', 'friction_rear',
                       'omega_wheels_rear',
                       'omega_wheels_front', 'delta'])

def StateWrapper(xu):
    return WX(v_x=xu[..., 0],
              v_y=xu[..., 1],
              r=xu[..., 2],
              friction_front=xu[..., 3],
              friction_rear=xu[..., 4],
              omega_wheels_rear=xu[..., 5],
              omega_wheels_front=xu[..., 6],
              delta=xu[..., 7])

WEX = namedtuple('WEX', ['v_x', 'v_y', 'r', 'dFz',
                       'omega_wheels_rear',
                       'omega_wheels_front', 'delta'])

def ExtendedStateWrapper(xu):
    return WEX(v_x=xu[..., 0],
              v_y=xu[..., 1],
              r=xu[..., 2],
              dFz=xu[..., 3],
              omega_wheels_rear=xu[..., 4],
              omega_wheels_front=xu[..., 5],
              delta=xu[..., 6])