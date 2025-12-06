

import cv2
from sharpen_image import sharpen_image

# Wczytanie obrazu
image = cv2.imread('brudne/202511190035/48001F003202511190035.bmp', 0)  # 0 = grayscale

# Użycie:
enhanced_image = sharpen_image(image)
# lub z customowymi parametrami:
# enhanced_image = sharpen_image(image, clip_limit=7.0, sharpen_amount=3.0)

# Zapisanie wyniku
cv2.imwrite('enhanced_output.bmp', enhanced_image)

#gotowa aplikacja