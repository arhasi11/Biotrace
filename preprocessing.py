import cv2

IMG_SIZE = 128

def preprocess_image(path):
    img = cv2.imread(path)

    # ✅ Convert to RGB (3 channels)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    return img