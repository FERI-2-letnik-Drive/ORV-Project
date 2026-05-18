import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent
UTILS = BASE / 'utils'


if __name__ == '__main__':
    img = cv2.imread(str(UTILS.joinpath("man-face-test.jpg")))

    if img is None:
        raise FileNotFoundError("Image not found")

    cv2.imshow("Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()