from torchvision import transforms

def get_train_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

def get_eval_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

def get_transforms(split_type):
    match split_type:
        case "train":
            return get_train_transform
        case "eval":
            return get_eval_transform