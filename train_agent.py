import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

print("=" * 60)
print("BREAST CANCER IMAGING AGENT TRAINER")
print("=" * 60)

# Path to your dataset
dataset_path = r"C:\Users\yami\Downloads\archive (1)\Dataset_BUSI_with_GT"

# Check if dataset exists
if not os.path.exists(dataset_path):
    print(f"❌ ERROR: Dataset not found at: {dataset_path}")
    print("\nPlease update the dataset_path variable with the correct path.")
    print("The dataset should have 3 folders: benign, malignant, normal")
    exit(1)

print(f"✓ Dataset found at: {dataset_path}")

# Check dataset structure
expected_folders = ['benign', 'malignant', 'normal']
found_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
print(f"✓ Found folders: {found_folders}")

# 1️⃣ Data preprocessing with augmentation
print("\n" + "=" * 60)
print("STEP 1: Preparing Data")
print("=" * 60)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

try:
    train_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    print(f"✓ Training samples: {train_data.samples}")
    print(f"✓ Validation samples: {val_data.samples}")
    print(f"✓ Classes: {train_data.class_indices}")

except Exception as e:
    print(f"❌ ERROR loading data: {e}")
    print("\nMake sure your dataset structure is:")
    print("Dataset_BUSI_with_GT/")
    print("├── benign/")
    print("│   ├── image1.png")
    print("│   ├── image2.png")
    print("├── malignant/")
    print("│   ├── image1.png")
    print("│   ├── image2.png")
    print("└── normal/")
    print("    ├── image1.png")
    print("    ├── image2.png")
    exit(1)

# 2️⃣ Build model with transfer learning
print("\n" + "=" * 60)
print("STEP 2: Building Model")
print("=" * 60)

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

print("✓ Loaded MobileNetV2 base model")

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(3, activation='softmax')(x)  # 3 classes

model = Model(inputs=base_model.input, outputs=output)

print("✓ Added custom classification layers")

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled")
print(f"✓ Total parameters: {model.count_params():,}")

# 3️⃣ Setup callbacks
print("\n" + "=" * 60)
print("STEP 3: Setting up Training Callbacks")
print("=" * 60)

checkpoint = ModelCheckpoint(
    'best_imaging_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("✓ Callbacks configured")

# 4️⃣ Train model
print("\n" + "=" * 60)
print("STEP 4: Training Model")
print("=" * 60)
print("This may take 10-30 minutes depending on your hardware...")
print()

try:
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    # 5️⃣ Save final model
    print("\n" + "=" * 60)
    print("STEP 5: Saving Model")
    print("=" * 60)

    model.save("final_imaging_model.keras")
    print("✓ Final model saved as 'final_imaging_model.keras'")

    # Print training summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    
    print("\n✅ TRAINING COMPLETE!")
    print("\nYou can now use the model with your FastAPI application.")
    print("Run: uvicorn agents.imaging_agent.app:app --reload --port 5010")

except KeyboardInterrupt:
    print("\n⚠ Training interrupted by user")
    model.save("interrupted_model.keras")
    print("✓ Model saved as 'interrupted_model.keras'")

except Exception as e:
    print(f"\n❌ ERROR during training: {e}")
    import traceback
    traceback.print_exc()
