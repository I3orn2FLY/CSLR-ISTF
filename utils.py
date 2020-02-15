import time
import numpy as np
from config import *


def print_progress(cur_idx, L, start_time):
    time_left = int((time.time() - start_time) * 1.0 / cur_idx * (L - cur_idx))

    hours = time_left // 3600

    minutes = time_left % 3600 // 60

    seconds = time_left % 60

    print("\rProgress: %.2f" % (cur_idx * 100 / L) + "% " \
          + str(hours) + " hours " \
          + str(minutes) + " minutes " \
          + str(seconds) + " seconds left",
          end=" ")


