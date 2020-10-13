"""
ProcessEnvData.py
Author: Maggie Jacoby, October 2020

This is script to run the data cleaning on env params
Use in combination with HomeDataClasses.py and cleanData.py
"""




import os
import sys
import csv
import ast
import json
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import HomeDataClasses as HD
from cleanData import *

from my_functions import *
from gen_argparse import *




if __name__ == '__main__':

    # root_path = path.strip('/').split('/')[:-1]

    data = HD.ReadEnv(house_entry=home_system, pi=True, root_dir=path)
    data.main()
    data.clean_data()