from utils.dataset import BreastDataset

dataset = BreastDataset("data/train")

print("Total images:", len(dataset))

img, label = dataset[0]
print("Image shape:", img.shape)
print("Label:", label)