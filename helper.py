import cv2

def show_images(titles, images, wait=True):
    """Display multiple images with one line of code"""

    for (title, image) in zip(titles, images):
        cv2.imshow(title, image)

    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()