import torch
import folder_s
import torchvision.transforms as transforms


def get_dut_omron(
        root, root_gt, use_pretrained=True, image_size=(224, 224),
        batch_size=16, shuffle=True, num_workers=2
        ):

    if use_pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    if image_size is not None:
        transform = transforms.Compose([
            transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

    dataset = folder_s.ImageFolderS(
        root=root,
        root_gt=root_gt,
        transform=transform,
        transform_gt=transform
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
        )

    return data_loader
