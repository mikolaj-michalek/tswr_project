from collections import namedtuple

EP = namedtuple('EP', ['friction'])

def ExternalParamsWrapper(p):
    return EP(friction=p[..., 0],)