"""General package utilities
"""

import os
from pathlib import Path

# define input and working directories depending on platform
ON_KAGGLE: bool = os.environ['PWD'] == '/kaggle/working'
INPUT_DIR = Path('/kaggle/input/titanic') if ON_KAGGLE else Path('/Users/herman/Dropbox/ml_practice/classic_ml/titanic/data/raw')
WORKING_DIR = Path('/kaggle/working') if ON_KAGGLE else Path('/Users/herman/Dropbox/ml_practice/classic_ml/titanic')
