import torchvision.transforms as transforms
from constants import image_size, batch_size
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def init_dataloader():
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Absollute path
    # Don't look at the names xd 
    PATH_TO_EPI = '/Users/mingtongyuan/june/python/side_project/epi_generator/epi'
    
    dataset = ImageFolder(PATH_TO_EPI, transform=transform)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
