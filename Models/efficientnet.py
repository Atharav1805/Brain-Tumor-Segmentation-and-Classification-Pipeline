# =============================================
# Models/efficientnet_transfer.py
# =============================================
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3, EfficientNetB4
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

def build_efficientnet(model_variant="B0", input_shape=(224, 224, 3), num_classes=3, 
                       trainable_layers=30, dropout_rate=0.3):
    """
    Build EfficientNet (B0/B3/B4) transfer learning model for brain tumor classification.
    """
    model_map = {
        "B0": EfficientNetB0,
        "B3": EfficientNetB3,
        "B4": EfficientNetB4
    }
    if model_variant not in model_map:
        raise ValueError(f"Invalid variant {model_variant}. Choose from B0, B3, B4.")

    base_model = model_map[model_variant](weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze all layers initially
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(dropout_rate)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model
