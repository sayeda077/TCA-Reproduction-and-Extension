import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

CIFAR100_CLASSES = """apple,aquarium fish,baby,bear,beaver,bed,bee,beetle,bicycle,bottle,bowl,boy,bridge,bus,butterfly,camel,can,castle,caterpillar,cattle,chair,chimpanzee,clock,cloud,cockroach,couch,crab,crocodile,cup,dinosaur,dolphin,elephant,flatfish,forest,fox,girl,hamster,house,kangaroo,keyboard,lamp,lawn mower,leopard,lion,lizard,lobster,man,maple tree,motorcycle,mountain,mouse,mushroom,oak tree,orange,orchid,otter,palm tree,pear,pickup truck,pine tree,plain,plate,poppy,porcupine,possum,rabbit,raccoon,ray,road,rocket,rose,sea,seal,shark,shrew,skunk,skyscraper,snail,snake,spider,squirrel,streetcar,sunflower,sweet pepper,table,tank,telephone,television,tiger,tractor,train,trout,tulip,turtle,wardrobe,whale,willow tree,wolf,woman,worm""".split(",")

TEMPLATE = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
]

class CIFAR100C(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.corruption = os.environ.get(
            "CIFAR100C_CORRUPTION", "contrast"
        ).lower()

        self.severity = int(
            os.environ.get("CIFAR100C_SEVERITY", "1")
        )

        if self.severity not in range(1, 6):
            raise ValueError("CIFAR100C_SEVERITY must be 1-5")

        image_file = os.path.join(
            root, self.corruption + ".npy"
        )
        label_file = os.path.join(root, "labels.npy")

        self.images = np.load(image_file, mmap_mode="r")
        self.labels = np.load(label_file, mmap_mode="r")

        self.start = (self.severity - 1) * 10000
        self.end = self.severity * 10000

        self.classnames = CIFAR100_CLASSES
        self.template = TEMPLATE

        print(
            f"CIFAR-100-C: corruption={self.corruption}, "
            f"severity={self.severity}, "
            f"samples={self.end - self.start}"
        )

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        real_idx = self.start + idx

        array = np.array(self.images[real_idx], copy=True)
        image = Image.fromarray(array).convert("RGB")
        label = int(self.labels[real_idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, label
