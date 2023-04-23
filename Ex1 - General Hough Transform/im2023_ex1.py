import numpy as np
from collections import defaultdict
from skimage.feature import canny
from scipy.ndimage import sobel
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog

MIN_CANNY_THRESHOLD = 10
MAX_CANNY_THRESHOLD = 50
letters = {
    'M': {45.0: [(41, 48), (41, 11), (41, -9), (40, -7), (39, 9), (39, -46), (38, -7), (37, -6), (36, 44), (36, 14),
                 (36, -11), (36, -40), (35, -6), (35, -11), (34, -5), (33, -10), (32, -9), (31, -5), (30, -4),
                 (29, -9),
                 (28, -8), (27, -4), (26, -3), (25, -8), (24, -3), (24, -7), (23, -2), (21, -7), (20, -2), (20, -6),
                 (19, -1), (18, -6), (18, -21), (17, -5), (17, -23), (16, -1), (15, 0), (14, -5), (13, 2), (13, -4),
                 (13, -23), (12, -22), (10, -4), (9, -3), (9, -20), (8, -19), (7, -3), (6, -2), (5, -19), (4, -18),
                 (3, -2), (2, -1), (2, -18), (1, -17), (-1, -1), (-2, 0), (-2, -17), (-3, -16), (-6, -16), (-7, -15),
                 (-9, -15), (-10, -14), (-13, -14), (-14, -13), (-17, -13), (-18, -12), (-19, 19), (-20, -12),
                 (-21, 18),
                 (-21, -11), (-22, -16), (-23, -15), (-24, -11), (-25, -10), (-26, -15), (-27, -14), (-28, -10),
                 (-29, -9), (-30, -14), (-31, -9), (-31, -13), (-32, -8), (-33, -42), (-34, 44), (-34, -13),
                 (-34, -41),
                 (-35, 25), (-35, -7), (-35, -12), (-36, 27), (-36, -24), (-36, -41), (-37, -12), (-38, 14),
                 (-38, -11),
                 (-39, 48), (-39, 21), (-39, -18), (-40, 47), (-40, 12), (-40, -11), (-40, -19), (-40, -46)],
          0.0: [(41, 47), (41, 46), (41, 45), (41, 44), (41, 43), (41, 42), (41, 41), (41, 40), (41, 39), (41, 38),
                (41, 37), (41, 36), (41, 35), (41, 34), (41, 33), (41, 32), (41, 31), (41, 30), (41, 29), (41, 28),
                (41, 27), (41, 26), (41, 25), (41, 24), (41, 23), (41, 22), (41, 21), (41, 20), (41, 19), (41, 18),
                (41, 17), (41, 16), (41, 15), (41, 14), (41, 13), (41, 12), (41, 10), (41, -10), (41, -11), (41, -12),
                (41, -13), (41, -14), (41, -15), (41, -16), (41, -17), (41, -18), (41, -19), (41, -20), (41, -21),
                (41, -22), (41, -23), (41, -24), (41, -25), (41, -26), (41, -27), (41, -28), (41, -29), (41, -30),
                (41, -31), (41, -32), (41, -33), (41, -34), (41, -35), (41, -36), (41, -37), (41, -38), (41, -39),
                (41, -40), (41, -41), (41, -42), (41, -43), (41, -44), (40, 9), (40, -46), (39, 48), (38, 48), (38, 9),
                (38, -46), (37, 48), (37, -46), (36, 48), (36, 43), (36, 42), (36, 41), (36, 40), (36, 39), (36, 38),
                (36, 37), (36, 36), (36, 35), (36, 34), (36, 33), (36, 32), (36, 31), (36, 30), (36, 29), (36, 28),
                (36, 27), (36, 26), (36, 25), (36, 24), (36, 23), (36, 22), (36, 21), (36, 20), (36, 19), (36, 18),
                (36, 17), (36, 16), (36, 15), (36, 13), (36, 8), (36, -6), (36, -12), (36, -13), (36, -14), (36, -15),
                (36, -16), (36, -17), (36, -18), (36, -19), (36, -20), (36, -21), (36, -22), (36, -23), (36, -24),
                (36, -25), (36, -26), (36, -27), (36, -28), (36, -29), (36, -30), (36, -31), (36, -32), (36, -33),
                (36, -34), (36, -35), (36, -36), (36, -37), (36, -38), (36, -39), (36, -41), (36, -46), (35, 48),
                (35, 13), (35, 8), (35, -41), (35, -46), (34, 48), (34, 44), (34, 12), (34, -42), (34, -46), (33, 48),
                (33, 44), (33, 12), (33, 7), (33, -5), (33, -42), (33, -46), (32, 48), (32, 44), (32, 12), (32, 7),
                (32, -5), (32, -42), (32, -46), (31, 48), (31, 44), (31, 7), (31, -9), (31, -42), (31, -46), (30, 48),
                (30, 44), (30, 11), (30, 7), (30, -9), (30, -42), (30, -46), (29, 48), (29, 44), (29, 11), (29, -4),
                (29, -42), (29, -46), (28, 48), (28, 44), (28, 11), (28, 6), (28, -4), (28, -42), (28, -46), (27, 48),
                (27, 44), (27, 6), (27, -8), (27, -42), (27, -46), (26, 48), (26, 44), (26, 10), (26, -8), (26, -42),
                (26, -46), (25, 48), (25, 44), (25, 10), (25, 5), (25, -3), (25, -42), (25, -46), (24, 48), (24, 44),
                (24, 5), (24, -42), (24, -46), (23, 48), (23, 44), (23, 9), (23, 5), (23, -7), (23, -42), (23, -46),
                (22, 48), (22, 44), (22, 9), (22, -2), (22, -7), (22, -42), (22, -46), (21, 48), (21, 44), (21, 9),
                (21, 4), (21, -2), (21, -42), (21, -46), (20, 48), (20, 44), (20, 4), (20, -42), (20, -46), (19, 48),
                (19, 44), (19, 8), (19, -6), (19, -42), (19, -46), (18, 48), (18, 44), (18, 8), (18, 3), (18, -1),
                (18, -22), (18, -23), (18, -42), (18, -46), (17, 48), (17, 44), (17, 25), (17, 23), (17, 8), (17, 3),
                (17, -1), (17, -42), (17, -46), (16, 48), (16, 44), (16, 25), (16, 23), (16, 3), (16, -5), (16, -21),
                (16, -23), (16, -42), (16, -46), (15, 48), (15, 44), (15, 25), (15, 23), (15, 7), (15, -5), (15, -21),
                (15, -23), (15, -42), (15, -46), (14, 48), (14, 44), (14, 25), (14, 23), (14, 7), (14, 2), (14, 0),
                (14, -23), (14, -42), (14, -46), (13, 48), (13, 44), (13, 25), (13, 23), (13, 0), (13, -42), (13, -46),
                (12, 48), (12, 44), (12, 25), (12, 6), (12, 1), (12, -4), (12, -20), (12, -42), (12, -46), (11, 48),
                (11, 44), (11, 25), (11, 22), (11, 6), (11, -4), (11, -20), (11, -22), (11, -42), (11, -46), (10, 48),
                (10, 44), (10, 25), (10, 22), (10, 6), (10, -20), (10, -22), (10, -42), (10, -46), (9, 48), (9, 44),
                (9, 25), (9, 22), (9, -22), (9, -42), (9, -46), (8, 48), (8, 44), (8, 25), (8, 5), (8, -3), (8, -22),
                (8, -42), (8, -46), (7, 48), (7, 44), (7, 25), (7, 21), (7, 5), (7, -19), (7, -22), (7, -42), (7, -46),
                (6, 48), (6, 44), (6, 25), (6, 21), (6, 5), (6, -19), (6, -22), (6, -42), (6, -46), (5, 48), (5, 44),
                (5, 25), (5, 21), (5, -2), (5, -22), (5, -42), (5, -46), (4, 48), (4, 44), (4, 25), (4, 4), (4, -2),
                (4, -22), (4, -42), (4, -46), (3, 48), (3, 44), (3, 25), (3, 20), (3, 4), (3, -18), (3, -22), (3, -42),
                (3, -46), (2, 48), (2, 44), (2, 25), (2, 20), (2, -22), (2, -42), (2, -46), (1, 48), (1, 44), (1, 25),
                (1, 20), (1, 3), (1, -1), (1, -22), (1, -42), (1, -46), (0, 48), (0, 44), (0, 25), (0, 3), (0, -1),
                (0, -17), (0, -22), (0, -42), (0, -46), (-1, 48), (-1, 44), (-1, 25), (-1, 19), (-1, 3), (-1, -17),
                (-1, -22), (-1, -42), (-1, -46), (-2, 48), (-2, 44), (-2, 25), (-2, 19), (-2, -22), (-2, -42),
                (-2, -46),
                (-3, 48), (-3, 44), (-3, 25), (-3, 19), (-3, 2), (-3, 0), (-3, -22), (-3, -42), (-3, -46), (-4, 48),
                (-4, 44), (-4, 25), (-4, 2), (-4, 0), (-4, -16), (-4, -22), (-4, -42), (-4, -46), (-5, 48), (-5, 44),
                (-5, 25), (-5, 18), (-5, 2), (-5, 0), (-5, -16), (-5, -22), (-5, -42), (-5, -46), (-6, 48), (-6, 44),
                (-6, 25), (-6, 18), (-6, 2), (-6, 0), (-6, -22), (-6, -42), (-6, -46), (-7, 48), (-7, 44), (-7, 25),
                (-7, -22), (-7, -42), (-7, -46), (-8, 48), (-8, 44), (-8, 25), (-8, 17), (-8, -15), (-8, -22),
                (-8, -42),
                (-8, -46), (-9, 48), (-9, 44), (-9, 25), (-9, 17), (-9, -22), (-9, -42), (-9, -46), (-10, 48),
                (-10, 44),
                (-10, 25), (-10, 17), (-10, -22), (-10, -42), (-10, -46), (-11, 48), (-11, 44), (-11, 25), (-11, -14),
                (-11, -22), (-11, -42), (-11, -46), (-12, 48), (-12, 44), (-12, 25), (-12, 16), (-12, -14), (-12, -22),
                (-12, -42), (-12, -46), (-13, 48), (-13, 44), (-13, 25), (-13, 16), (-13, -22), (-13, -42), (-13, -46),
                (-14, 48), (-14, 44), (-14, 25), (-14, 16), (-14, -22), (-14, -42), (-14, -46), (-15, 48), (-15, 44),
                (-15, 25), (-15, -13), (-15, -22), (-15, -42), (-15, -46), (-16, 48), (-16, 44), (-16, 25), (-16, 21),
                (-16, 19), (-16, 15), (-16, -13), (-16, -19), (-16, -22), (-16, -42), (-16, -46), (-17, 48), (-17, 44),
                (-17, 25), (-17, 21), (-17, 19), (-17, 15), (-17, -22), (-17, -42), (-17, -46), (-18, 48), (-18, 44),
                (-18, 25), (-18, 21), (-18, 19), (-18, -16), (-18, -18), (-18, -22), (-18, -42), (-18, -46), (-19, 48),
                (-19, 44), (-19, 25), (-19, 21), (-19, 14), (-19, -12), (-19, -16), (-19, -18), (-19, -22), (-19, -42),
                (-19, -46), (-20, 48), (-20, 44), (-20, 25), (-20, 18), (-20, 14), (-20, -16), (-20, -18), (-20, -22),
                (-20, -42), (-20, -46), (-21, 48), (-21, 44), (-21, 25), (-21, 20), (-21, 14), (-21, -16), (-21, -18),
                (-21, -22), (-21, -42), (-21, -46), (-22, 48), (-22, 44), (-22, 25), (-22, 20), (-22, 18), (-22, -11),
                (-22, -18), (-22, -22), (-22, -42), (-22, -46), (-23, 48), (-23, 44), (-23, 25), (-23, 20), (-23, 18),
                (-23, 13), (-23, -11), (-23, -18), (-23, -22), (-23, -42), (-23, -46), (-24, 48), (-24, 44), (-24, 25),
                (-24, 20), (-24, 13), (-24, -15), (-24, -18), (-24, -22), (-24, -42), (-24, -46), (-25, 48), (-25, 44),
                (-25, 25), (-25, 20), (-25, 17), (-25, 13), (-25, -15), (-25, -18), (-25, -22), (-25, -42), (-25, -46),
                (-26, 48), (-26, 44), (-26, 25), (-26, 20), (-26, 17), (-26, -10), (-26, -18), (-26, -22), (-26, -42),
                (-26, -46), (-27, 48), (-27, 44), (-27, 25), (-27, 20), (-27, 17), (-27, 12), (-27, -10), (-27, -18),
                (-27, -22), (-27, -42), (-27, -46), (-28, 48), (-28, 44), (-28, 25), (-28, 20), (-28, 12), (-28, -14),
                (-28, -18), (-28, -22), (-28, -42), (-28, -46), (-29, 48), (-29, 44), (-29, 25), (-29, 20), (-29, 16),
                (-29, -14), (-29, -18), (-29, -22), (-29, -42), (-29, -46), (-30, 48), (-30, 44), (-30, 25), (-30, 20),
                (-30, 16), (-30, 11), (-30, -9), (-30, -18), (-30, -22), (-30, -42), (-30, -46), (-31, 48), (-31, 44),
                (-31, 25), (-31, 20), (-31, 16), (-31, 11), (-31, -18), (-31, -22), (-31, -42), (-31, -46), (-32, 48),
                (-32, 44), (-32, 25), (-32, 20), (-32, 11), (-32, -13), (-32, -18), (-32, -22), (-32, -42), (-32, -46),
                (-33, 48), (-33, 44), (-33, 25), (-33, 20), (-33, 15), (-33, -8), (-33, -13), (-33, -18), (-33, -22),
                (-33, -46), (-34, 48), (-34, 25), (-34, 20), (-34, 15), (-34, 10), (-34, -8), (-34, -18), (-34, -46),
                (-35, 48), (-35, 20), (-35, -18), (-35, -23), (-35, -41), (-35, -46), (-36, 48), (-36, 41), (-36, 40),
                (-36, 39), (-36, 38), (-36, 37), (-36, 36), (-36, 35), (-36, 34), (-36, 33), (-36, 32), (-36, 31),
                (-36, 30), (-36, 29), (-36, 28), (-36, 20), (-36, 14), (-36, 7), (-36, 6), (-36, 5), (-36, 4),
                (-36, 3),
                (-36, 2), (-36, 1), (-36, 0), (-36, -1), (-36, -2), (-36, -3), (-36, -4), (-36, -5), (-36, -12),
                (-36, -18), (-36, -25), (-36, -26), (-36, -27), (-36, -28), (-36, -29), (-36, -30), (-36, -31),
                (-36, -32), (-36, -33), (-36, -34), (-36, -35), (-36, -36), (-36, -37), (-36, -38), (-36, -39),
                (-36, -46), (-37, 48), (-37, 20), (-37, 14), (-37, -18), (-37, -46), (-38, 48), (-38, 20), (-38, -18),
                (-38, -46), (-39, 13), (-39, -11), (-39, -46), (-40, 46), (-40, 45), (-40, 44), (-40, 43), (-40, 42),
                (-40, 41), (-40, 40), (-40, 39), (-40, 38), (-40, 37), (-40, 36), (-40, 35), (-40, 34), (-40, 33),
                (-40, 32), (-40, 31), (-40, 30), (-40, 29), (-40, 28), (-40, 27), (-40, 26), (-40, 25), (-40, 24),
                (-40, 23), (-40, 11), (-40, 10), (-40, 9), (-40, 8), (-40, 7), (-40, 6), (-40, 5), (-40, 4), (-40, 3),
                (-40, 2), (-40, 1), (-40, 0), (-40, -1), (-40, -2), (-40, -3), (-40, -4), (-40, -5), (-40, -6),
                (-40, -7), (-40, -8), (-40, -9), (-40, -20), (-40, -21), (-40, -22), (-40, -23), (-40, -24),
                (-40, -25),
                (-40, -26), (-40, -27), (-40, -28), (-40, -29), (-40, -30), (-40, -31), (-40, -32), (-40, -33),
                (-40, -34), (-40, -35), (-40, -36), (-40, -37), (-40, -38), (-40, -39), (-40, -40), (-40, -41),
                (-40, -42), (-40, -43), (-40, -44)],
          89.0: [(40, 48), (40, 10), (39, -7), (37, 9), (35, 44), (34, 8), (34, -10), (31, 12), (29, 7), (27, 11),
                 (26, 6), (24, 10), (22, 5), (20, 9), (19, 4), (17, -21), (16, 8), (15, 3), (13, 7), (13, -20),
                 (12, 23),
                 (11, 2), (9, 6), (8, 22), (5, 5), (4, 21), (2, 4), (0, 20), (-2, 3), (-4, 19), (-7, 18), (-11, 17),
                 (-15, 16), (-18, 15), (-20, 21), (-22, 14), (-24, 18), (-26, 13), (-28, 17), (-29, 12), (-32, 16),
                 (-33, 11), (-34, -22), (-35, 44), (-35, 15), (-35, 10), (-36, 42), (-36, 8), (-36, -6), (-36, -23),
                 (-36, -40), (-39, 14), (-40, 48), (-40, 22), (-40, 13), (-40, -10), (-40, -18), (-40, -45)],
          90.0: [(40, -8), (40, -45), (12, 2), (12, 0), (-20, 19), (-35, 43), (-35, 26), (-35, 9)],
          44.0: [(35, -10), (14, -20)], 71.0: [(14, -21), (-35, -8), (-39, 20)], 18.0: [(11, 0), (-36, -7), (-40, 21)]},
    'K': {45.0: [(42, 46), (42, 16), (41, -2), (41, -3), (40, 47), (40, 14), (40, -1), (40, -2), (39, 0), (39, -1),
                 (39, -44), (39, -45), (38, 42), (38, 20), (38, 1), (38, 0), (38, -35), (38, -43), (38, -44), (37, 19),
                 (37, 2), (37, 1), (37, -5), (37, -6), (37, -42), (37, -43), (36, 2), (36, -4), (36, -5), (36, -35),
                 (36, -41), (36, -42), (35, -3), (35, -4), (35, -40), (35, -41), (34, 4), (34, -2), (34, -3), (34, -32),
                 (34, -39), (34, -40), (33, 5), (33, 4), (33, -1), (33, -2), (33, -38), (33, -39), (32, 6), (32, 5),
                 (32, 0), (32, -1), (32, -32), (32, -37), (32, -38), (31, 7), (31, 6), (31, 1), (31, 0), (31, -29),
                 (31, -30), (31, -36), (31, -37), (30, 8), (30, 7), (30, 2), (30, 1), (30, -28), (30, -29), (30, -35),
                 (30, -36), (29, 9), (29, 8), (29, 3), (29, 2), (29, -27), (29, -28), (29, -34), (29, -35), (28, 10),
                 (28, 9), (28, 4), (28, 3), (28, -26), (28, -27), (28, -33), (28, -34), (27, 10), (27, 4), (27, -25),
                 (27, -26), (27, -32), (27, -33), (26, -24), (26, -25), (26, -31), (26, -32), (25, 14), (25, 12),
                 (25, 6), (25, -23), (25, -24), (25, -30), (25, -31), (24, 7), (24, 6), (24, -22), (24, -23), (24, -29),
                 (24, -30), (23, 8), (23, 7), (23, -21), (23, -22), (23, -28), (23, -29), (22, 9), (22, 8), (22, -20),
                 (22, -21), (22, -27), (22, -28), (21, 10), (21, 9), (21, -19), (21, -20), (21, -26), (21, -27),
                 (20, 11), (20, 10), (20, -18), (20, -19), (20, -25), (20, -26), (19, 12), (19, 11), (19, -17),
                 (19, -18), (19, -24), (19, -25), (18, 13), (18, 12), (18, -16), (18, -17), (18, -23), (18, -24),
                 (17, 13), (17, -15), (17, -16), (17, -22), (17, -23), (16, -14), (16, -15), (16, -21), (16, -22),
                 (15, 15), (15, -13), (15, -14), (15, -20), (15, -21), (14, 16), (14, 15), (14, -12), (14, -13),
                 (14, -19), (14, -20), (13, 16), (13, -11), (13, -12), (12, -11), (12, -17), (10, -10), (9, -16),
                 (-6, 9), (-6, 8), (-7, 10), (-7, 9), (-8, 11), (-8, 10), (-9, 12), (-9, 11), (-10, 13), (-10, 12),
                 (-11, 14), (-11, 13), (-12, 15), (-12, 14), (-13, 16), (-13, 15), (-14, 17), (-14, 16), (-14, 10),
                 (-14, 9), (-15, 17), (-15, 11), (-15, 10), (-16, 12), (-16, 11), (-17, 19), (-17, 13), (-17, 12),
                 (-18, 13), (-33, 42), (-34, 41), (-34, 19), (-34, -10), (-34, -38), (-37, 47), (-37, -45), (-38, 14),
                 (-39, 45), (-39, 16), (-39, -45)],
          0.2249: [(42, 45), (42, -4), (42, -46), (38, 41), (38, -7), (-33, 19), (-37, 14), (-38, -45)],
          0.0: [(42, 44), (42, 43), (42, 42), (42, 41), (42, 40), (42, 39), (42, 38), (42, 37), (42, 36), (42, 35),
                (42, 34), (42, 33), (42, 32), (42, 31), (42, 30), (42, 29), (42, 28), (42, 27), (42, 26), (42, 25),
                (42, 24), (42, 23), (42, 22), (42, 21), (42, 20), (42, 19), (42, 18), (42, 17), (42, -5), (42, -6),
                (42, -7), (42, -8), (42, -9), (42, -10), (42, -11), (42, -12), (42, -13), (42, -14), (42, -15),
                (42, -16), (42, -17), (42, -18), (42, -19), (42, -20), (42, -21), (42, -22), (42, -23), (42, -24),
                (42, -25), (42, -26), (42, -27), (42, -28), (42, -29), (42, -30), (42, -31), (42, -32), (42, -33),
                (42, -34), (42, -35), (42, -36), (42, -37), (42, -38), (42, -39), (42, -40), (42, -41), (42, -42),
                (42, -43), (42, -44), (42, -45), (41, 46), (39, 47), (39, 14), (38, 47), (38, 40), (38, 39), (38, 38),
                (38, 37), (38, 36), (38, 35), (38, 34), (38, 33), (38, 32), (38, 31), (38, 30), (38, 29), (38, 28),
                (38, 27), (38, 26), (38, 25), (38, 24), (38, 23), (38, 22), (38, 21), (38, 14), (38, -8), (38, -9),
                (38, -10), (38, -11), (38, -12), (38, -13), (38, -14), (38, -15), (38, -16), (38, -17), (38, -18),
                (38, -19), (38, -20), (38, -21), (38, -22), (38, -23), (38, -24), (38, -25), (38, -26), (38, -27),
                (38, -28), (38, -29), (38, -30), (38, -31), (38, -32), (38, -33), (38, -34), (37, 47), (37, 14),
                (36, 47), (36, 42), (36, 19), (36, 14), (35, 47), (35, 42), (35, 19), (35, 14), (35, 3), (34, 47),
                (34, 42), (34, 19), (34, 14), (33, 47), (33, 42), (33, 19), (33, 14), (33, -32), (32, 47), (32, 42),
                (32, 19), (32, 14), (32, -31), (31, 47), (31, 42), (31, 19), (31, 14), (30, 47), (30, 42), (30, 19),
                (30, 14), (29, 47), (29, 42), (29, 19), (29, 14), (28, 47), (28, 42), (28, 19), (28, 14), (27, 47),
                (27, 42), (27, 19), (27, 14), (26, 47), (26, 42), (26, 19), (26, 14), (26, 11), (26, 5), (25, 47),
                (25, 42), (25, 19), (24, 47), (24, 42), (24, 19), (24, 13), (23, 47), (23, 42), (23, 19), (22, 47),
                (22, 42), (22, 19), (21, 47), (21, 42), (21, 19), (20, 47), (20, 42), (20, 19), (19, 47), (19, 42),
                (19, 19), (18, 47), (18, 42), (18, 19), (17, 47), (17, 42), (17, 19), (16, 47), (16, 42), (16, 19),
                (16, 14), (15, 47), (15, 42), (15, 19), (14, 47), (14, 42), (14, 19), (13, 47), (13, 42), (13, 19),
                (13, -18), (12, 47), (12, 42), (12, 19), (11, 47), (11, 42), (11, 19), (11, 17), (10, 47), (10, 42),
                (10, -16), (9, 47), (9, 42), (8, 47), (8, 42), (8, -11), (7, 47), (7, 42), (7, -17), (6, 47), (6, 42),
                (5, 47), (5, 42), (5, -13), (4, 47), (4, 42), (4, -19), (3, 47), (3, 42), (2, 47), (2, 42), (2, -15),
                (1, 47), (1, 42), (1, -21), (0, 47), (0, 42), (-1, 47), (-1, 42), (-1, -17), (-1, -22), (-2, 47),
                (-2, 42), (-3, 47), (-3, 42), (-3, -18), (-4, 47), (-4, 42), (-4, -24), (-5, 47), (-5, 42), (-5, 7),
                (-6, 47), (-6, 42), (-6, 6), (-6, -20), (-7, 47), (-7, 42), (-7, -26), (-8, 47), (-8, 42), (-8, 5),
                (-9, 47), (-9, 42), (-9, -22), (-10, 47), (-10, 42), (-10, -28), (-11, 47), (-11, 42), (-11, 3),
                (-12, 47), (-12, 42), (-12, -24), (-13, 47), (-13, 42), (-13, 8), (-13, 2), (-13, -30), (-14, 47),
                (-14, 42), (-15, 47), (-15, 42), (-15, 6), (-15, 1), (-15, -26), (-15, -31), (-16, 47), (-16, 42),
                (-16, 18), (-17, 47), (-17, 42), (-17, 5), (-17, 0), (-17, -27), (-18, 47), (-18, 42), (-18, 19),
                (-18, -33), (-19, 47), (-19, 42), (-19, 19), (-19, 4), (-19, -1), (-20, 47), (-20, 42), (-20, 19),
                (-20, 14), (-20, -29), (-21, 47), (-21, 42), (-21, 19), (-21, 14), (-21, 3), (-21, -2), (-22, 47),
                (-22, 42), (-22, 19), (-22, 14), (-22, -36), (-23, 47), (-23, 42), (-23, 19), (-23, 14), (-23, 2),
                (-23, -31), (-24, 47), (-24, 42), (-24, 19), (-24, 14), (-24, -4), (-24, -37), (-25, 47), (-25, 42),
                (-25, 19), (-25, 14), (-26, 47), (-26, 42), (-26, 19), (-26, 14), (-26, 0), (-26, -5), (-26, -33),
                (-27, 47), (-27, 42), (-27, 19), (-27, 14), (-27, -39), (-28, 47), (-28, 42), (-28, 19), (-28, 14),
                (-28, -1), (-28, -6), (-29, 47), (-29, 42), (-29, 19), (-29, 14), (-29, -35), (-30, 47), (-30, 42),
                (-30, 19), (-30, 14), (-30, -2), (-30, -7), (-30, -41), (-31, 47), (-31, 42), (-31, 19), (-31, 14),
                (-32, 47), (-32, 42), (-32, 19), (-32, 14), (-32, -3), (-33, 47), (-33, 14), (-33, -9), (-33, -43),
                (-34, 47), (-34, 40), (-34, 39), (-34, 38), (-34, 37), (-34, 36), (-34, 35), (-34, 34), (-34, 33),
                (-34, 32), (-34, 31), (-34, 30), (-34, 29), (-34, 28), (-34, 27), (-34, 26), (-34, 25), (-34, 24),
                (-34, 23), (-34, 22), (-34, 21), (-34, 14), (-34, -4), (-34, -11), (-34, -12), (-34, -13), (-34, -14),
                (-34, -15), (-34, -16), (-34, -17), (-34, -18), (-34, -19), (-34, -20), (-34, -21), (-34, -22),
                (-34, -23), (-34, -24), (-34, -25), (-34, -26), (-34, -27), (-34, -28), (-34, -29), (-34, -30),
                (-34, -31), (-34, -32), (-34, -33), (-34, -34), (-34, -35), (-34, -36), (-35, 47), (-35, 14),
                (-35, -44), (-36, 47), (-36, 14), (-37, -6), (-39, 44), (-39, 43), (-39, 42), (-39, 41), (-39, 40),
                (-39, 39), (-39, 38), (-39, 37), (-39, 36), (-39, 35), (-39, 34), (-39, 33), (-39, 32), (-39, 31),
                (-39, 30), (-39, 29), (-39, 28), (-39, 27), (-39, 26), (-39, 25), (-39, 24), (-39, 23), (-39, 22),
                (-39, 21), (-39, 20), (-39, 19), (-39, 18), (-39, 17), (-39, -9), (-39, -10), (-39, -11), (-39, -12),
                (-39, -13), (-39, -14), (-39, -15), (-39, -16), (-39, -17), (-39, -18), (-39, -19), (-39, -20),
                (-39, -21), (-39, -22), (-39, -23), (-39, -24), (-39, -25), (-39, -26), (-39, -27), (-39, -28),
                (-39, -29), (-39, -30), (-39, -31), (-39, -32), (-39, -33), (-39, -34), (-39, -35), (-39, -36),
                (-39, -37), (-39, -38), (-39, -39), (-39, -40), (-39, -41), (-39, -42), (-39, -43)],
          0.2266: [(42, 15), (-13, 7)], 45.25: [(42, -3), (40, -45), (38, -6), (32, -30), (13, -17), (-5, 8), (-13, 9)],
          89.56: [(41, 15), (24, 14), (6, -12), (5, -18), (3, -14), (2, -20), (0, -16), (-3, -23), (-5, -19), (-6, -25),
                  (-8, -21), (-9, -27), (-10, 4), (-11, -23), (-12, -29), (-14, 7), (-14, -25), (-17, -32), (-19, -28),
                  (-20, -34), (-21, -35), (-22, -30), (-23, -3), (-25, 1), (-25, -32), (-26, -38), (-28, -34),
                  (-29, -40), (-31, -36), (-32, -8), (-32, -37), (-32, -42), (-34, 42), (-34, -9), (-36, -5),
                  (-38, -6)],
          0.6743: [(41, 14), (9, -11), (8, -17), (6, -13), (5, -19), (3, -15), (2, -21), (0, -17), (0, -22), (-2, -18),
                   (-3, -24), (-5, -20), (-6, -26), (-7, 5), (-8, -22), (-9, -28), (-10, 3), (-11, -24), (-12, 2),
                   (-12, -30), (-14, 6), (-14, 1), (-14, -26), (-14, -31), (-16, 5), (-16, 0), (-16, -27), (-17, -33),
                   (-18, 4), (-18, -1), (-19, -29), (-20, 3), (-20, -2), (-21, -36), (-22, 2), (-22, -31), (-23, -4),
                   (-23, -37), (-25, 0), (-25, -5), (-25, -33), (-26, -39), (-27, -1), (-27, -6), (-28, -35), (-29, -2),
                   (-29, -7), (-29, -41), (-31, -3), (-32, -9), (-32, -38), (-32, -43), (-33, -4), (-34, -44),
                   (-36, -6), (-36, -45)],
          90.0: [(41, -47), (40, -46), (37, -36), (35, -34), (34, -33), (-33, -38), (-38, 15), (-38, -7)],
          0.4512: [(38, 19), (38, -36), (7, -12), (6, -18), (4, -14), (3, -20), (1, -16), (-2, -23), (-4, -19), (-5, 6),
                   (-5, -25), (-7, -21), (-8, -27), (-9, 4), (-10, -23), (-11, -29), (-13, -25), (-16, -32), (-18, -28),
                   (-19, -34), (-20, -35), (-21, -30), (-22, -3), (-24, 1), (-24, -32), (-25, -38), (-27, -34),
                   (-28, -40), (-30, -36), (-31, -8), (-31, -37), (-31, -42), (-35, -5), (-38, 46)],
          89.8: [(37, 42), (12, 17), (11, -10), (11, -16), (9, -10), (8, -16), (7, -11), (6, -17), (4, -13), (3, -19),
                 (1, -15), (0, -21), (-2, -17), (-2, -22), (-4, -18), (-5, -24), (-7, 6), (-7, -20), (-8, -26), (-9, 5),
                 (-10, -22), (-11, -28), (-12, 3), (-13, -24), (-14, 2), (-14, -30), (-16, 6), (-16, 1), (-16, -26),
                 (-16, -31), (-18, 5), (-18, 0), (-18, -27), (-19, 14), (-19, -33), (-20, 4), (-20, -1), (-21, -29),
                 (-22, 3), (-22, -2), (-23, -36), (-24, 2), (-24, -31), (-25, -4), (-25, -37), (-27, 0), (-27, -5),
                 (-27, -33), (-28, -39), (-29, -1), (-29, -6), (-30, -35), (-31, -2), (-31, -7), (-31, -41), (-33, -3),
                 (-34, 20), (-34, -37), (-34, -43), (-35, -4), (-36, -44), (-38, 47), (-39, -8), (-39, -44)],
          44.78: [(36, 3), (27, 11), (27, 5), (17, 14), (13, 17), (12, -10), (12, -16), (-15, 18), (-18, 14)],
          71.6: [(36, -36), (34, 3), (25, 11), (25, 5), (15, 14)], 18.44: [(35, -35), (24, 12), (13, -19)],
          89.3: [(-39, 46)],
          0.224: [],
          0.226: []}

}


def gradient_orientation(image):
    """Calculate the gradient orientation for edge point in the image"""
    dx = sobel(image, axis=0, mode='constant')
    dy = sobel(image, axis=1, mode='constant')
    gradient = np.arctan2(dy, dx) * 180 / np.pi

    return gradient


def build_r_table(image, origin):
    """Build the R-table from the given shape image and a reference point"""
    edges = canny(image, low_threshold=MIN_CANNY_THRESHOLD,
                  high_threshold=MAX_CANNY_THRESHOLD)
    gradient = gradient_orientation(edges)

    r_table = defaultdict(list)
    for (i, j), value in np.ndenumerate(edges):
        if value:
            r_table[gradient[i, j]].append((origin[0] - i, origin[1] - j))

    return r_table


def accumulate_gradients(r_table, grayImage):
    """Perform a General Hough Transform with the given image and R-table"""
    edges = canny(grayImage)
    gradient = gradient_orientation(edges)

    accumulator = np.zeros(grayImage.shape)
    for i, j in [(i, j) for i, row in enumerate(edges) for j, value in enumerate(row) if value]:
        try:
            for r in r_table[gradient[i, j]]:
                accum_i, accum_j = i + r[0], j + r[1]
                if accum_i < accumulator.shape[0] and accum_j < accumulator.shape[1]:
                    accumulator[accum_i, accum_j] += 1
        except KeyError:
            pass

    return accumulator


def general_hough(query_image):
    """ Uses a accumulate_gradients to detect shapes in an image and create nice output """
    accumulator = accumulate_gradients(letters['M'], query_image)
    plt.subplot(121)
    plt.title('Pinpoint M')
    plt.imshow(query_image, cmap='gray')
    max_accumulator = np.max(accumulator)
    indices = np.argwhere(accumulator == max_accumulator)
    i, j = indices[0]
    plt.scatter([j], [i], marker='o', color='blue')
    accumulator = accumulate_gradients(letters['K'], query_image)
    plt.subplot(122)
    plt.title('Pinpoint K')
    plt.imshow(query_image, cmap='gray')
    max_accumulator = np.max(accumulator)
    indices = np.argwhere(accumulator == max_accumulator)
    i, j = indices[0]
    plt.scatter([j], [i], marker='o', color='red')
    plt.show()


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=
                                           (("image type", "*.jpg"), ("image type", "*.PNG")))
    image = plt.imread(file_path)
    image = image[:, :, 0]
    fig, ax = plt.subplots()
    im = ax.imshow(image, cmap='gray')
    ax.axis('off')
    general_hough(image)
