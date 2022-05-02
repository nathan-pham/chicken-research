from chicken import Chicken

import random
import uuid
import os
import cv2
import imutils

__dirname = os.path.dirname(os.path.realpath(__file__))
resolve = lambda *p: os.path.join(__dirname, *p)

"""
BUILD DATASET
https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5
"""
def get_images(image_dir):
    return [resolve(image_dir, path) for path in os.listdir(resolve(image_dir)) if not "seg" in path]


def get_lbs(filename):
    return filename.split('#').pop().lower().replace('.jpg', '')

def build_dataset(image_dir, export_file):
    images = get_images(image_dir)
    export_file = resolve(export_file)

    results = []

    for i in range(len(images)):
        current_image = f" {i + 1}/{len(images)}"
        try:
            image = images[i]
            lbs = get_lbs(image)
            chicken = Chicken(image)

            if chicken.ratio > 30 and chicken.ratio < 200:
                results.append([chicken.ratio, lbs]) # hard cap ratio at 200
                print(f"completed (seg): {current_image}")
            else:
                os.remove(image)
                print(f"invalid (seg): {current_image}")
        except:
            os.remove(image)
            print(f"failed (seg): {current_image}")


    with open(export_file, "w") as f:
        f.write("ratio,lbs\n")
        for result in results:
            f.write(",".join([str(n) for n in result]) + "\n")

    print(f"finished {image_dir}, exported to {export_file}")


"""
AUGMENTATION
"""
def fill(img, h, w):
    return cv2.resize(img, (h, w), cv2.INTER_CUBIC)

def horizontal_shift(img, ratio=random.random()):
    if random.random() > 0.5:
        ratio = random.uniform(-ratio, ratio)
        h, w = img.shape[:2]
        to_shift = w*ratio
        if ratio > 0:
            img = img[:, :int(w-to_shift), :]
        if ratio < 0:
            img = img[:, int(-1*to_shift):, :]
        img = fill(img, h, w)
    return img

def vertical_shift(img, ratio=random.random()):
    if random.random() > 0.5:
        ratio = random.uniform(-ratio, ratio)
        h, w = img.shape[:2]
        to_shift = h*ratio
        if ratio > 0:
            img = img[:int(h-to_shift), :, :]
        if ratio < 0:
            img = img[int(-1*to_shift):, :, :]
        img = fill(img, h, w)
    return img

def translate(img):
    return imutils.translate(img, random.randint(-300, 300), random.randint(-300, 300))

def zoom(img, value=min(random.random() + 0.75, 0.99)):
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = fill(img, h, w)
    return img

def horizontal_flip(img):
    return cv2.flip(img, 1)

def vertical_flip(img):
    return cv2.flip(img, 0)

def rotation(img, angle=random.randint(0, 360)):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def augment(image_dir, export_dir, expansion=10):
    images = get_images(image_dir)

    methods = [
        translate,
        zoom,
        horizontal_shift,
        vertical_shift,
        horizontal_flip,
        vertical_flip,
    ]

    for i in range(len(images)):
        image = images[i]
        pure_img = cv2.imread(image)
        lbs = float(get_lbs(image))

        for _ in range(expansion):
            export_filename = resolve(export_dir, f"{str(uuid.uuid4())}#{str(lbs)}.jpg")
            img = pure_img.copy()

            # apply random transformations
            random.shuffle(methods)
            for method in methods:
                if random.random() > 0.25:
                    img = method(img)

            img = imutils.resize(img, width=300)
            cv2.imwrite(export_filename, img)

        for angle in [0, 90, 180, 270, 360]:
            img = pure_img.copy()
            img = imutils.rotate(img, angle)
            img = imutils.resize(img, width=300)

            export_filename = resolve(export_dir, f"{str(uuid.uuid4())}#{str(lbs)}.jpg")

            cv2.imwrite(export_filename , img)
        
        print(f"completed (aug): {i}/{len(images)}")
