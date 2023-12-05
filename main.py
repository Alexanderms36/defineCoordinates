import cv2
from solver import Solver
impath = "data/0.0/00.pgm"
img = cv2.imread(impath)
solver = Solver()
res = solver.solve(img)
print(res)