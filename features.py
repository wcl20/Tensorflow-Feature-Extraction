import argparse
import glob
import numpy as np
import os
import random
import tqdm
from core.io import HDF5Writer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="Path to dataset")
    parser.add_argument("-o", "--output", required=True, help="Output HDF5 file")
    parser.add_argument("-b", "--buffer", type=int, default=1000, help="Buffer size for feature extraction")
    args = parser.parse_args()

    # Shuffle paths before storing into file
    img_paths = glob.glob(f"{args.dataset}/*/*.jpg")
    random.shuffle(img_paths)

    # Get labels
    label_encoder = LabelEncoder()
    labels = [img_path.split(os.path.sep)[-2] for img_path in img_paths]
    labels = label_encoder.fit_transform(labels)

    # Output dims of VGG16 without top
    dims = (len(img_paths), 512 * 7 * 7)
    dataset = HDF5Writer(args.output, dims, buffer_size=args.buffer)
    dataset.store_class_names(label_encoder.classes_)

    # Feature extraction model
    model = VGG16(weights="imagenet", include_top=False)

    batch_size = 32
    for i in tqdm.tqdm(np.arange(0, len(img_paths), batch_size)):
        # Get batch
        batch_img_paths = img_paths[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        # Process each path
        batch_images = []
        for img_path in batch_img_paths:
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)
            batch_images.append(image)
        batch_images = np.vstack(batch_images)

        # Extract features
        batch_features = model.predict(batch_images, batch_size=batch_size)
        batch_features = batch_features.reshape((batch_features.shape[0], -1))

        # Add to dataset
        dataset.add(batch_features, batch_labels)

    dataset.close()


if __name__ == '__main__':
    main()
