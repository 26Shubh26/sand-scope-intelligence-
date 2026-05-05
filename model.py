import segmentation_models_pytorch as smp

def get_model(num_classes=10):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    return model

if __name__ == "__main__":
    model = get_model()
    print(model)
