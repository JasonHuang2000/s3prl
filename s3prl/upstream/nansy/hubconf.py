import os

from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def nansy_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
            feature_selection (str): 'c' (default) or 'z'
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)

def nansy_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return nansy_local(_urls_to_filepaths(ckpt, refresh=refresh, agent='gdown'), *args, **kwargs)

def nansy(refresh=False, *args, **kwargs):
    """
        The apc standard model on 360hr
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://drive.google.com/uc?id=18cDJRmAxIbV3TzSTS70Vat6JVkuVH4Wq'
    return nansy_url(refresh=refresh, *args, **kwargs)