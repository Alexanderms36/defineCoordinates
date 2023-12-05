import numpy as np
import cv2

class Solver:
    @staticmethod
    def find_triangle_vertices(image: np.ndarray) -> list:
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            if len(approx) == 3:
                return approx
        return None

    def find(self, image: np.ndarray) -> list:
        img = cv2.fastNlMeansDenoising(image, None, 20, 7, 21)
        threshold1 = 100
        threshold2 = 200
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1, threshold2)
        verts = Solver.find_triangle_vertices(edges)
        coords = np.asarray(verts)
        return coords

    def __init__(self):
        self.alg = self.find


    def solve(self, image: np.ndarray) -> np.ndarray:
        result = self.alg(image)
        return result