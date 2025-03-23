# Import submodules
from . import submodule1
from . import submodule2

# Import specific functions
from .submodule1 import save_model, load_model

# Package metadata
__version__ = '1.0.0'
__author__ = 'alluethrenn'

# Initialization code
def initialize():
    print("Initializing the Supa package")

initialize()

# Define the public API
__all__ = ['save_model', 'load_model', 'submodule1', 'submodule2'], 'initialize'


