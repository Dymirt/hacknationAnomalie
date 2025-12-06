import cv2
import numpy as np

class XRayFilters:
    """
    Klasa wykonująca pełny pipeline:
        Brighten whites -> CLAHE -> Unsharp Mask

    Zachowuje się TAK SAMO jak:
        - funkcja sharpen_image()
        - klasa XRayPreprocessor
    """

    def __init__(self,
                 clip_limit=4.0,
                 grid_size=(8, 8),
                 sharpen_amount=2.5,
                 sharpen_blur_sigma=1.0,
                 white_threshold=240):
        
        # Parametry CLAHE
        self.clip_limit = clip_limit
        self.grid_size = grid_size

        # Parametry Unsharp Mask
        self.sharpen_amount = sharpen_amount
        self.sharpen_blur_sigma = sharpen_blur_sigma

        # Próg rozjaśniania
        self.white_threshold = white_threshold

    # ------------------------------------------------------------
    # ELEMENT 0: konwersja do grayscale (tak jak w oryginale)
    # ------------------------------------------------------------
    def _ensure_gray(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    # ------------------------------------------------------------
    # KROK 1: Brighten whites (identycznie jak w Twoim kodzie)
    # ------------------------------------------------------------
    def brighten_whites(self, image):
        result = image.copy()
        result[result >= self.white_threshold] = 255
        return result

    # ------------------------------------------------------------
    # KROK 2: CLAHE
    # ------------------------------------------------------------
    def apply_clahe(self, image):
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.grid_size
        )
        return clahe.apply(image)

    # ------------------------------------------------------------
    # KROK 3: Unsharp Mask (dokładna kopia Twojego kodu)
    # ------------------------------------------------------------
    def apply_unsharp_mask(self, image):
        gaussian_blur = cv2.GaussianBlur(image, (0, 0), self.sharpen_blur_sigma)
        sharpened = cv2.addWeighted(
            image,
            self.sharpen_amount,
            gaussian_blur,
            1 - self.sharpen_amount,
            0
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------
    # GŁÓWNY PIPELINE — identyczny jak funkcja sharpen_image()
    # ------------------------------------------------------------
    def process(self, image):
        """
        Wykonuje pełny pipeline:
        Brighten whites -> CLAHE -> Unsharp Mask
        """
        gray = self._ensure_gray(image)
        bright = self.brighten_whites(gray)
        contrasted = self.apply_clahe(bright)
        sharpened = self.apply_unsharp_mask(contrasted)
        return sharpened
