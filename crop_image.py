import cv2
import numpy as np
import os
import shutil
from PIL import Image

def apply_hardcore_processing(pil_img):
    # Konwersja na numpy
    img = np.array(pil_img)

    # Upewniamy się że mamy grayscale do obliczeń
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 1. CLAHE (Kontrast)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    final = clahe.apply(img)

    # 2. Wyostrzanie
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    final_sharpened = cv2.filter2D(final, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    
    # Zwracamy obraz PIL w trybie "L" (Grayscale)
    return Image.fromarray(final_sharpened)

def get_object_coordinates(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # Zakładamy: tło jasne, obiekt ciemny -> THRESH_BINARY_INV
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    margin = 20
    h_img, w_img = img.shape
    x_start = max(x - margin, 0)
    y_start = max(y - margin, 0)
    x_end = min(x + w + margin, w_img)
    y_end = min(y + h + margin, h_img)
    
    return (x_start, y_start, x_end, y_end)

def process_with_pil(src_path, output_folder, coords, tile_size=256, overlap_ratio=0.5):
    # 1. Wczytanie
    img = Image.open(src_path).convert("L")
    cropped_img = img.crop(coords)
    
    print(f"Wycięto obiekt: {cropped_img.size}. Przetwarzam...")

    # 2. Hardcore Processing na całości
    processed_full_img = apply_hardcore_processing(cropped_img)

    # 3. Reset folderu
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    w, h = processed_full_img.size
    count = 0
    stride = int(tile_size * (1 - overlap_ratio))
    stride = max(1, stride)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            box = (x, y, x + tile_size, y + tile_size)
            tile = processed_full_img.crop(box)
            
            # --- FIX NA CZARNY PADDING (The Nuclear Option) ---
            # Zamiast walczyć z paletami Grayscale, tworzymy kafelek RGB.
            # (255, 255, 255) to ZAWSZE biały.
            
            if tile.size != (tile_size, tile_size):
                # Tworzymy puste BIAŁE tło w RGB
                new_tile = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
                
                # Konwertujemy nasz szary wycinek na RGB, żeby pasował do tła
                tile_rgb = tile.convert("RGB")
                
                # Wklejamy
                new_tile.paste(tile_rgb, (0, 0))
                tile = new_tile
            else:
                # Jeśli rozmiar jest OK, i tak konwertujemy na RGB dla spójności plików
                tile = tile.convert("RGB")

            # Zapisz
            filename = f"tile_{count}_x{x}_y{y}.bmp"
            tile.save(os.path.join(output_folder, filename))
            count += 1
            
    print(f"Gotowe. Utworzono {count} kafelków w '{output_folder}'.")

if __name__ == "__main__":
    plik_wejsciowy = r"C:\Users\jakub\Desktop\NAUKA\czyste\202511180023\48001F003202511180023 czarno.bmp"
    folder_wyjsciowy = "test_tiled"
    
    print("Szukam obiektu...")
    coordinates = get_object_coordinates(plik_wejsciowy)
    
    if coordinates:
        process_with_pil(plik_wejsciowy, folder_wyjsciowy, coordinates, overlap_ratio=0.5)
    else:
        print("Błąd detekcji.")