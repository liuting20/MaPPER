from .mapper_vg import MaPPER


def build_model(args,config):
    return MaPPER(args,config)
