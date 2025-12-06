import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# --- FUNKCJE POMOCNICZE DO UŻYCIA W INNYCH SKRYPTACH ---

def sharpen_image(image, clip_limit=4.0, grid_size=(8, 8), 
                  sharpen_amount=2.5, sharpen_blur_sigma=1.0,
                  white_threshold=240):
    """
    Funkcja do wyostrzania obrazów RTG.
    
    Pipeline: Brighten whites -> CLAHE -> Unsharp Mask
    
    Args:
        image: Obraz wejściowy (grayscale numpy array lub BGR - zostanie skonwertowany)
        clip_limit: Parametr CLAHE dla kontrastu (domyślnie 4.0)
        grid_size: Rozmiar siatki CLAHE (domyślnie (8, 8))
        sharpen_amount: Siła wyostrzenia (domyślnie 2.5)
        sharpen_blur_sigma: Sigma dla Gaussian blur w unsharp mask (domyślnie 1.0)
        white_threshold: Próg od którego jasne piksele stają się białe (domyślnie 240)
    
    Returns:
        Wyostrzony obraz (numpy array, uint8)
    """
    # Konwersja na grayscale jeśli potrzeba
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Krok 0: Ustawienie bardzo jasnych pikseli na białe (255)
    gray[gray >= white_threshold] = 255
    
    # Krok 1: CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    contrasted = clahe.apply(gray)
    
    # Krok 2: Unsharp Mask
    gaussian_blur = cv2.GaussianBlur(contrasted, (0, 0), sharpen_blur_sigma)
    sharpened = cv2.addWeighted(contrasted, sharpen_amount, gaussian_blur, 1 - sharpen_amount, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened


class XRayPreprocessor:
    """
    Klasa do zaawansowanego przetwarzania obrazów RTG (Industrial/Security X-Ray).
    Realizuje pipeline: Brighten whites -> CLAHE -> Sharpen.
    """

    def __init__(self, clip_limit=6.0, grid_size=(8, 8),
                 sharpen_amount=2.5, sharpen_blur_sigma=1.0,
                 white_threshold=240, output_dir='output'):
        # Parametry CLAHE
        self.clip_limit = clip_limit
        self.grid_size = grid_size

        # Parametry wyostrzania
        self.sharpen_amount = sharpen_amount
        self.sharpen_blur_sigma = sharpen_blur_sigma
        
        # Próg dla jasnych pikseli
        self.white_threshold = white_threshold

        # Katalog wyjściowy
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def brighten_whites(self, image):
        """
        KROK 0: Ustawienie bardzo jasnych pikseli na kompletnie białe (255).
        Pomaga wyciągnąć anomalie na tle jasnych obszarów.
        """
        result = image.copy()
        result[result >= self.white_threshold] = 255
        return result

    def apply_clahe(self, image):
        """
        KROK 1: Adaptacyjne wyrównanie histogramu.
        Wyciąga detale z cieni (np. gęste elementy silnika/ramy).
        """
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
        return clahe.apply(image)

    def apply_unsharp_mask(self, image):
        """
        KROK 2: Silne wyostrzanie poprzez odjęcie rozmycia Gaussa (Unsharp Masking).
        Wzór: dst = image * amount + blur * (1-amount)
        """
        gaussian_blur = cv2.GaussianBlur(image, (0, 0), self.sharpen_blur_sigma)
        # Mocne wyostrzenie: amount > 1 wzmacnia wysokie częstotliwości (krawędzie, detale)
        sharpened = cv2.addWeighted(image, self.sharpen_amount, gaussian_blur, 1 - self.sharpen_amount, 0)
        # Ograniczenie do zakresu 0-255
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return sharpened

    def align_images_ecc(self, im_scan, im_reference):
        """
        OPCJONALNIE: Wyrównanie geometryczne (Image Alignment) algorytmem ECC.
        Przydatne przy porównywaniu par obrazów (Scan vs Wzorzec).
        """
        print(" -> Rozpoczynam wyrównywanie obrazów (ECC)...")
        # Konwersja na grayscale float32
        im1_gray = np.float32(im_reference)
        im2_gray = np.float32(im_scan)

        # Inicjalizacja macierzy warp (identyczność)
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Kryteria zatrzymania: 500 iteracji lub precyzja 1e-5
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-5)

        try:
            # Znajdowanie transformacji (może potrwać kilka sekund dla dużych obrazów)
            (cc, warp_matrix) = cv2.findTransformECC(
                im1_gray, im2_gray, warp_matrix, cv2.MOTION_TRANSLATION, criteria
            )
            print(f" -> Znaleziono korelację: {cc:.4f}")

            # Przekształcenie obrazu badanego
            aligned_scan = cv2.warpAffine(
                im_scan, warp_matrix, (im_reference.shape[1], im_reference.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            return aligned_scan
        except cv2.error:
            print(" -> Ostrzeżenie: Algorytm ECC nie zbiegł się (zbyt duże różnice). Zwracam oryginał.")
            return im_scan

    def process_pipeline(self, image_path):
        """
        Uruchamia pełny pipeline przetwarzania na jednym obrazie.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Nie znaleziono pliku: {image_path}")

        # Wczytanie (Grayscale)
        original = cv2.imread(image_path, 0)
        if original is None:
            raise ValueError("Nie udało się wczytać obrazu (format nieobsługiwany?).")

        # Pipeline
        brightened = self.brighten_whites(original)
        contrasted = self.apply_clahe(brightened)
        sharpened = self.apply_unsharp_mask(contrasted)

        # Zapisywanie wyników
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(self.output_dir, f"{base_name}_1_original.bmp"), original)
        cv2.imwrite(os.path.join(self.output_dir, f"{base_name}_2_brightened.bmp"), brightened)
        cv2.imwrite(os.path.join(self.output_dir, f"{base_name}_3_clahe.bmp"), contrasted)
        cv2.imwrite(os.path.join(self.output_dir, f"{base_name}_4_final_enhanced.bmp"), sharpened)

        print(f"Zapisano wyniki do katalogu: {self.output_dir}")

        return {
            "original": original,
            "brightened": brightened,
            "clahe": contrasted,
            "final_enhanced": sharpened
        }

    def visualize_results(self, results_dict):
        """
        Wizualizacja wyników przy użyciu Matplotlib.
        """
        plt.figure(figsize=(16, 4))

        titles = ["1. Oryginał", "2. Jasne -> Białe", "3. Po CLAHE", "4. Finalny (Wyostrzony)"]

        keys = ["original", "brightened", "clahe", "final_enhanced"]

        for i, key in enumerate(keys):
            plt.subplot(1, 4, i + 1)
            plt.imshow(results_dict[key], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()


# --- PRZYKŁAD UŻYCIA ---
if __name__ == "__main__":
    # Plik wejściowy
    input_file = 'brudne/202511190035/48001F003202511190035.bmp'

    # Utworzenie pliku testowego (gdyby nie istniał), żeby skrypt zadziałał od razu
    if not os.path.exists(input_file):
        print("Tworzę testowy obraz (szum)...")
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        dummy_img = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
        cv2.imwrite(input_file, dummy_img)

    # Inicjalizacja procesora z parametrami dla RTG - wyostrzanie dla detekcji anomalii
    processor = XRayPreprocessor(
        clip_limit=6.0,  # Bardzo mocny kontrast dla ukrytych detali
        grid_size=(8, 8),  # Standardowa siatka
        sharpen_amount=2.5,  # Silne wyostrzenie (domyślnie 2.5)
        sharpen_blur_sigma=1.0,  # Mniejszy blur = ostrzejsze krawędzie
        white_threshold=240,  # Piksele >= 240 stają się białe (255)
        output_dir='output'  # Katalog na pliki wyjściowe
    )

    try:
        # Przetwarzanie
        print(f"Przetwarzanie pliku: {input_file}...")
        results = processor.process_pipeline(input_file)

        # Wizualizacja
        print("Wyświetlanie wyników...")
        processor.visualize_results(results)

    except Exception as e:
        print(f"Błąd: {e}")