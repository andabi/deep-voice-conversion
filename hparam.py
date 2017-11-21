# -*- coding: utf-8 -*-
#!/usr/bin/env python

import yaml
# import pprint

# path
## local
data_path_base = './datasets'
logdir_path = './logdir'

## remote
# data_path_base = '/data/private/vc/datasets'
# logdir_path = '/data/private/vc/logdir'



def load_hparam(filename):
    stream = open(filename, 'r')
    docs = yaml.load_all(stream)
    hparam_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = v
    return hparam_dict


def merge_dict(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_dict(user[k], v)
    return user


class Dotdict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value


class Hparam():
    default_hparam_file = 'hparams/default.yaml'
    user_hparams_file = 'hparams/hparams.yaml'
    global_hparam = None

    def __init__(self, case):
        default_hp = load_hparam(Hparam.default_hparam_file)
        user_hp = load_hparam(Hparam.user_hparams_file)
        if case in user_hp:
            hp = merge_dict(user_hp[case], default_hp)
        else:
            hp = default_hp
        self.hparam = Dotdict(hp)

    def __call__(self):
        return self.hparam

    def set_as_global_hparam(self):
        Hparam.global_hparam = self.hparam

    @staticmethod
    def get_global_hparam():
        return Hparam.global_hparam

# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(hparam)