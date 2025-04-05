import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, \
    Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import cv2


TARGET_SIZE = (224, 224)
IMG_HEIGHT, IMG_WIDTH = TARGET_SIZE
FOLDER_IMAGINI = r"C:\Users\alex2\Desktop\dataset\dataset\train_dataset\train_dataset"
CSV_FILE = r"C:\Users\alex2\Desktop\train-data.csv"
OUTPUT_CSV_PATH = r"C:\Users\alex2\Desktop\final_predictions.csv"
EPOCHS_CLS = 10
EPOCHS_REG_PIXEL = 10
EPOCHS_REG_BBOX = 20
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42


def preproceseaza_imaginea(cale_imagine):
    img = load_img(cale_imagine, target_size=TARGET_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return img_array


print("Se încarcă datele...")
df = pd.read_csv(CSV_FILE)
print("Coloanele din CSV:", df.columns.tolist())

datapoint_ids = []
imagini = []
pixel_counts_true = []
bounding_boxes_true_str = []
bounding_boxes_true_norm = []
duck_labels_true = []

for index, row in df.iterrows():
    image_id = str(row['DatapointID'])
    cale_imagine = os.path.join(FOLDER_IMAGINI, f"{image_id}.png")
    if os.path.exists(cale_imagine):
        img_array = preproceseaza_imaginea(cale_imagine)
        imagini.append(img_array)
        datapoint_ids.append(image_id)

        duck_label = int(row['DuckOrNoDuck'])
        duck_labels_true.append(duck_label)

        pixel_counts_true.append(float(row['PixelCount']))

        bbox_str = str(row['BoundingBox'])
        bounding_boxes_true_str.append(bbox_str)

        coords = list(map(int, bbox_str.split()))
        if duck_label == 1 and len(coords) == 4:
            norm_coords = [
                coords[0] / IMG_WIDTH,
                coords[1] / IMG_HEIGHT,
                coords[2] / IMG_WIDTH,
                coords[3] / IMG_HEIGHT
            ]
            bounding_boxes_true_norm.append(norm_coords)
        else:
            bounding_boxes_true_norm.append([0.0, 0.0, 0.0, 0.0])
    else:
        print(f"Imaginea {cale_imagine} nu a fost găsită.")

imagini = np.array(imagini)
pixel_counts_true = np.array(pixel_counts_true, dtype=np.float32)
duck_labels_true = np.array(duck_labels_true, dtype=np.int32)
bounding_boxes_true_norm = np.array(bounding_boxes_true_norm, dtype=np.float32)

print(f"Numărul de imagini preprocesate: {imagini.shape}")
print(f"Numărul de etichete DuckOrNoDuck: {duck_labels_true.shape}")
print(f"Numărul de etichete PixelCount: {pixel_counts_true.shape}")
print(f"Numărul de etichete BoundingBox (normalizate): {bounding_boxes_true_norm.shape}")

print("\n--- Antrenare Model Clasificare (Subtask 1) ---")
X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(
    imagini, duck_labels_true, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE, stratify=duck_labels_true
)

model_cls = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_cls.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model_cls.summary()

history_cls = model_cls.fit(X_train_cls, y_train_cls,
                            validation_data=(X_val_cls, y_val_cls),
                            epochs=EPOCHS_CLS,
                            batch_size=BATCH_SIZE)

val_loss_cls, val_acc_cls = model_cls.evaluate(X_val_cls, y_val_cls, verbose=0)
print(f"Acuratețea pe setul de validare (clasificare): {val_acc_cls * 100:.2f}%")
model_cls.save("duck_or_no_duck_model.keras")

print("\n--- Antrenare Model Regresie Pixel Count (Subtask 2) ---")
X_train_pix, X_val_pix, y_train_pix, y_val_pix = train_test_split(
    imagini, pixel_counts_true, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE
)

model_reg_pixel = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='relu')
])

model_reg_pixel.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
model_reg_pixel.summary()

history_reg_pixel = model_reg_pixel.fit(X_train_pix, y_train_pix,
                                        validation_data=(X_val_pix, y_val_pix),
                                        epochs=EPOCHS_REG_PIXEL,
                                        batch_size=BATCH_SIZE)

val_loss_pix, val_mae_pix = model_reg_pixel.evaluate(X_val_pix, y_val_pix, verbose=0)
print(f"Mean Absolute Error pe setul de validare (Pixel Count): {val_mae_pix:.2f}")
model_reg_pixel.save("duck_pixel_count_model.keras")


print("\n--- Antrenare Model Regresie Bounding Box (Subtask 3) ---")
X_train_bbox, X_val_bbox, y_train_bbox, y_val_bbox = train_test_split(
    imagini, bounding_boxes_true_norm, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE
)

model_reg_bbox = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='sigmoid')  # Output normalizat între 0 și 1
])

model_reg_bbox.compile(optimizer=Adam(learning_rate=0.0003), loss='mean_squared_error', metrics=['mae'])
model_reg_bbox.summary()

history_reg_bbox = model_reg_bbox.fit(X_train_bbox, y_train_bbox,
                                      validation_data=(X_val_bbox, y_val_bbox),
                                      epochs=EPOCHS_REG_BBOX,
                                      batch_size=BATCH_SIZE)

val_loss_bbox, val_mae_bbox = model_reg_bbox.evaluate(X_val_bbox, y_val_bbox, verbose=0)
print(f"Mean Absolute Error pe setul de validare (Bounding Box Norm): {val_mae_bbox:.4f}")
model_reg_bbox.save("duck_bounding_box_model.keras")

print("\n--- Generare Predicții Finale pe întregul set de date încărcat ---")
preds_cls_prob = model_cls.predict(imagini)
predicted_duck_labels = (preds_cls_prob > 0.5).astype(int).flatten()

preds_pixel_counts = model_reg_pixel.predict(imagini).flatten()
preds_bboxes_norm = model_reg_bbox.predict(imagini)

final_pixel_counts = []
final_bounding_boxes_str = []

for i in range(len(imagini)):
    datapoint_id = datapoint_ids[i]
    predicted_duck = predicted_duck_labels[i]

    if predicted_duck == 0:
        final_pixel_counts.append(0)
        final_bounding_boxes_str.append("0 0 0 0")
    else:
        pred_pixels = max(0, int(round(preds_pixel_counts[i])))
        final_pixel_counts.append(pred_pixels)

        pred_bbox_norm = preds_bboxes_norm[i]
        x1_norm, y1_norm, x2_norm, y2_norm = pred_bbox_norm
        x1 = int(round(x1_norm * IMG_WIDTH))
        y1 = int(round(y1_norm * IMG_HEIGHT))
        x2 = int(round(x2_norm * IMG_WIDTH))
        y2 = int(round(y2_norm * IMG_HEIGHT))

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(IMG_WIDTH - 1, x2)
        y2 = min(IMG_HEIGHT - 1, y2)

        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        bbox_final_str = f"{x1} {y1} {x2} {y2}"
        final_bounding_boxes_str.append(bbox_final_str)

df_output = pd.DataFrame({
    "DatapointID": datapoint_ids,
    "DuckOrNoDuck": predicted_duck_labels,
    "PixelCount": final_pixel_counts,
    "BoundingBox": final_bounding_boxes_str
})

df_output = df_output[["DatapointID", "DuckOrNoDuck", "PixelCount", "BoundingBox"]]
df_output.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\nAm salvat fișierul {OUTPUT_CSV_PATH} cu predicțiile finale.")
print("\nExemple de predicții:")
print(df_output.head())
