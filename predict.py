from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

CKPT_PATH = "results/Patchcore/contraband_xray/latest/weights/lightning/model.ckpt"

# 1) ŚCIEŻKA DO OBRAZU LUB FOLDERU
#   - może być pojedynczy plik .bmp
#   - albo katalog z wieloma plikami
DATA_PATH = "cropped_images_bad/202511190100crop_0040_x1152_y256.bmp"

def main():
    # 2) Inicjalizacja modelu i engine
    #    Parametry możesz zostawić domyślne – wagi zostaną nadpisane z checkpointu.
    model = Patchcore(
        backbone="resnet18",				# Model do ekstrakcji cech z obrazów
        pre_trained=True,					# Użyj wstępnie wytrenowanych wag
        coreset_sampling_ratio=0.01,
        )
    engine = Engine()

    # 3) Dataset predykcyjny
    dataset = PredictDataset(
        path=DATA_PATH,
    )

    # 4) Predykcja
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=CKPT_PATH,
    )

    # 5) Obsługa wyników
    if predictions is None:
        print("Brak predykcji.")
        return

    for i, prediction in enumerate(predictions, start=1):
        # API wg aktualnej dokumentacji:
        image_path = prediction.image_path
        anomaly_map = prediction.anomaly_map      # mapa cieplna (pikselowo)
        pred_label = prediction.pred_label        # 0 = normalne, 1 = anomalia
        pred_score = float(prediction.pred_score) # score anomalii

        print(f"[{i}] {image_path}")
        print(f"    label : {pred_label} (0=normal, 1=anomalia)")
        print(f"    score : {pred_score:.4f}")

        # Tu możesz dodać zapis heatmapy do pliku, np. przez OpenCV/matplotlib.

if __name__ == "__main__":
    main()
