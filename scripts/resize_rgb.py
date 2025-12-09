from PIL import Image
import glob
import os

src = "./images/"
dst = "./images_resized/"
os.makedirs(dst, exist_ok=True)

for f in glob.glob(src + "*.jpeg"):
    img = Image.open(f)
    img = img.resize((1600, 900))  
    img.save(dst + os.path.basename(f))
print(glob.glob(src + "*.jpeg"))