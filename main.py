from relaxation_score_calculator import RelaxationScoreCalculator
from print_logger import PrintLogger
import numpy as np
import cv2

logger = PrintLogger()
relaxation_score_calculator = RelaxationScoreCalculator()
cumulative_score = 0
cumulative_score_ = 0
MAX_SCORE = 300
# cv2.namedWindow('score', cv2.WINDOW_NORMAL)
cv2.namedWindow('score')

while True:

    relaxation_score, feature_vector = relaxation_score_calculator.calc_score()
    logger.log(relaxation_score, feature_vector)

    cumulative_score_ = np.clip(cumulative_score_ + relaxation_score, 0, MAX_SCORE)
    cumulative_score = int(cumulative_score_)

    text_color = (255,255,255)   
    bar_color =  (255,255,255)  
    img_size = 500
    img = np.zeros((img_size, img_size, 3), np.uint8)


    bar_y1 = 400
    bar_y0 = bar_y1 - cumulative_score + 1
    bar_x0 = 250
    bar_x1 = 350

    img[bar_y0:bar_y1, bar_x0:bar_x1, :] = bar_color

    text = "Relaxation score: {}".format(cumulative_score)
    # cv2.putText(z,ss,(int(xx[ii]),int((yy[ii]))),
    cv2.putText(img, text,(img_size //6, int(0.1 * img_size)),
                cv2.FONT_HERSHEY_PLAIN, 1, text_color)
    cv2.imshow('score',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
