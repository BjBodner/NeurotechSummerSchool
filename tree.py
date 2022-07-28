import cv2
import numpy as np

empty_tree_img = cv2.imread(r'pictures\empty.png')
full_tree_img = cv2.imread(r'pictures\full.png')

cv2.namedWindow('score')

cumulative_score = 0
while True:
    
    text_color = (0,0,0)   
    bar_color =  (255,255,255)  
    img_size = 500
    empty_tree_img = cv2.resize(empty_tree_img, (img_size, img_size))
    full_tree_img = cv2.resize(full_tree_img, (img_size, img_size))




    # img = np.zeros((img_size, img_size, 3), np.uint8)

    np.sum(full_tree_img != 255, (1,2)) > img_size
    threshold = 200

    y_tree_start = np.where(np.sum(full_tree_img != 255, (1,2)) > threshold)[0][-1]
    y_tree_end = np.where(np.sum(full_tree_img != 255, (1,2)) > threshold)[0][0]

    y_full_start = y_tree_start - cumulative_score
    y_full_start = np.clip(y_full_start, y_tree_end, y_tree_start)

    mask = np.zeros((img_size, img_size, 3), np.uint8)
    mask[y_full_start:, :, :] = 1

    img = mask * full_tree_img + (1 - mask) * empty_tree_img

    cumulative_score += 1
    text = "Relaxation score: {}".format(cumulative_score)

    cv2.putText(img, text,(img_size //6, int(0.05 * img_size)),
                cv2.FONT_HERSHEY_PLAIN, 1, text_color)
    cv2.imshow('score',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
