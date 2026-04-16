#import some usefull librairies

import argparse
import os
import time

# command line argument parsing

parser = argparse.ArgumentParser(description="Train Intel Image Classifier")
parser.add_argument(
    "--model",
    type=str,
    choices=["pytorch", "tensorflow"],
    required=True,
    help="Choose which model to train: 'pytorch' or 'tensorflow'"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=15,
    help="Number of training epochs (default: 15)"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for training (default: 32)"
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="../intel_dataset",
    help="Path to the Intel dataset folder"
)
args = parser.parse_args()

# dataset paths

TRAIN_DIR = os.path.join(args.data_dir, "seg_train", "seg_train")
TEST_DIR  = os.path.join(args.data_dir, "seg_test",  "seg_test")

# dataset config

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = 6
IMG_SIZE = 150

print(f"\n{'='*55}")
print(f"  Training with: {args.model.upper()} model")
print(f"  Epochs:        {args.epochs}")
print(f"  Batch size:    {args.batch_size}")
print(f"  Image size:    {IMG_SIZE}x{IMG_SIZE}")
print(f"  Classes:       {CLASSES}")
print(f"{'='*55}\n")

# pytorch training

if args.model == "pytorch":

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms

    # device setup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PyTorch] Using device: {device}")
    if device.type == "cuda":
        print(f"[PyTorch] GPU name: {torch.cuda.get_device_name(0)}")

    # preprocessing

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load dataset

    print("[PyTorch] Loading dataset...")
    full_train = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

    total = len(full_train)
    val_size = int(0.2 * total)
    train_size = total - val_size

    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"[PyTorch] Train:      {train_size} images")
    print(f"[PyTorch] Validation: {val_size} images")
    print(f"[PyTorch] Test:       {len(test_dataset)} images")
    print(f"[PyTorch] Classes:    {full_train.classes}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    # model

    class IntelCNN_PyTorch(nn.Module):
        def __init__(self, num_classes=6):
            super(IntelCNN_PyTorch, self).__init__()

            self.block1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )

            self.block2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )

            self.block3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )

            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(128 * 18 * 18, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    model = IntelCNN_PyTorch(num_classes=NUM_CLASSES).to(device)
    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PyTorch] Total trainable parameters: {total_params:,}")

    # loss and optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
    )

    # training functions

    def train_one_epoch(model, loader, criterion, optimizer, device):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(f"Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def evaluate(model, loader, criterion, device):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    print(f"[PyTorch] Starting training...")
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/Tikome_Nana_model.pth")

    model.load_state_dict(torch.load("models/Tikome_Nana_model.pth", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"FINAL TEST ACCURACY:  {test_acc:.2f}%")

# tensorflow training

elif args.model == "tensorflow":

    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # gpu setup

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # preprocessing

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # load dataset

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="training",
        seed=42
    )

    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="validation",
        seed=42
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False
    )

    # model

    def build_keras_model(num_classes=6, img_size=150):
        model = keras.Sequential([
            keras.Input(shape=(img_size, img_size, 3)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model

    model = build_keras_model(NUM_CLASSES, IMG_SIZE)
    model.summary()

    # compile

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # callbacks

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="models/Tikome_Nana_model.keras",
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    # train

    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    # evaluate
    
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)

    print(f"FINAL TEST ACCURACY:  {test_acc * 100:.2f}%")
