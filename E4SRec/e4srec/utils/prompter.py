import os
import json
from typing import Union, List


class Prompter(object):
    __slots__ = ("template", "_verbose")
    
    def __init__(self, template_name: str = "", verbose: bool = False):
        pass