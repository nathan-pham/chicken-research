import utils

def main():
    utils.augment(
        image_dir = "./images/chicken_train",
        export_dir = "./images/chicken_augmented"
    )

    utils.build_dataset(
        image_dir = "./images/chicken_augmented",
        export_file = "./datasets/train.csv"
    )

if __name__ == "__main__":
    main()