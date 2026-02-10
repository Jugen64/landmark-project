import csv
from pathlib import Path
from collections import defaultdict
import random


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# READ-IN FILES:
DATASET_DIR = Path("~/Documents/Code/projects/datasets").expanduser()
IMAGE_DIR = DATASET_DIR / "gldv2_micro/images"
GLDV2_CSV_PATH = DATASET_DIR  / "gldv2_micro/gldv2_micro.csv"
LANDMARK_TO_COUNTRY_CSV = DATASET_DIR / "landmark_to_country/landmark_to_country.csv"

# READ+WRITE FILES:
METADATA_PATH = Path("~/Documents/Code/projects/landmark_project/data/processed/metadata.csv").expanduser()


# WRITE FILES:
SPLIT_DIR = Path("~/Documents/Code/projects/landmark_project/data/splits").expanduser()
TVTs = ["train", "val", "test"]
IDs = ["images", "landmarks", "countries"]

MIN_LANDMARKS = 20

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

RANDOM_SEED = 64
random.seed(RANDOM_SEED)

################ WRITE-TO-METADATA FUNCTIONS ################

def load_landmark_to_country():
    # IN:
    #
    # READS-FROM: LANDMARK_TO_COUNTRY_CSV
    #
    # OUT: landmark_to_country, dict: str -> str
    #     > maps landmark_id to its country
    landmark_to_country = {}
    with open(LANDMARK_TO_COUNTRY_CSV, newline='') as lm_t_c:
        landmark_to_country_reader = csv.reader(lm_t_c)
        header = next(landmark_to_country_reader)
        for row in landmark_to_country_reader:
            country = row[1]
            landmark_id = row[0]
            landmark_to_country[landmark_id] = country
    return landmark_to_country

def load_country_to_landmarks(landmark_to_country):
    # IN: landmark_to_country, dict: str -> str
    #     > maps landmark_id to its country
    #
    # READS FROM: GLDV2_CSV_PATH, IMAGE_DIR
    #
    # OUT: country_to_landmark_map, dict: str -> set(str)
    #     > maps a country name to a set of its landmarks by landmark id
    country_to_landmarks = defaultdict(set)
    with open(GLDV2_CSV_PATH, newline='') as csv_in:
        reader = csv.reader(csv_in)
        header = next(reader)
        
        for row in reader:
            image_filename = row[0]
            landmark_id = row[1]
            if landmark_id not in landmark_to_country:
                continue

            image_path = IMAGE_DIR / f"{image_filename}"
            if not image_path.exists():
                missing_images += 1
                continue
            
            country_name = landmark_to_country[landmark_id]
            country_to_landmarks[country_name].add(landmark_id)
    return country_to_landmarks

def filter_countries(country_to_landmark):
    # IN: country_to_landmark_map, dict: str -> set(str)
    #     > maps a country name to a set of its landmarks by landmark id
    #
    # OUT: remaining_countries, list[str]
    #     > list of country names with at least MIN_LANDMARKS landmarks
    remaining_countries = {x for x in country_to_landmark.keys() if len(country_to_landmark[x]) > MIN_LANDMARKS }
    return remaining_countries

def landmark_to_single_photo():
    # IN:
    #
    # READS FROM: GLDV2_CSV_PATH
    #
    # OUT: landmark_to_example, dict: str -> str
    #     > maps landmarks by landmark_id to a single image of that landmark
    landmark_to_example = {}
    filtered_landmarks = 0
    with open(GLDV2_CSV_PATH, 'r', newline='') as csv_in:
        reader = csv.reader(csv_in)
        header = next(reader)
        for row in reader:
            image_filename = row[0]
            landmark_id = row[1]
            if landmark_id not in landmark_to_example:
                # print(f"landmark {landmark_id} associated with image {image_filename}")
                landmark_to_example[landmark_id] = image_filename
            else:
                filtered_landmarks += 1
    return landmark_to_example, filtered_landmarks

def landmark_to_photoset():
    # IN:
    #
    # READS FROM: GLDV2_CSV_PATH
    #
    # OUT: landmark_to_example, dict: str -> set(str)
    #     > maps landmarks by landmark_id to a set of all images of that landmark
    landmark_to_photoset = defaultdict(set)
    filtered_landmarks = 0
    with open(GLDV2_CSV_PATH, 'r', newline='') as csv_in:
        reader = csv.reader(csv_in)
        header = next(reader)
        for row in reader:
            image_filename = row[0]
            landmark_id = row[1]
            landmark_to_photoset[landmark_id].add(image_filename)
    return landmark_to_photoset

def combine_and_write_out(landmark_to_photoset, landmark_to_country, remaining_countries):
    # IN: landmark_to_example, dict: str -> set(str)
    #     > maps landmarks by landmark_id to a set of all images of that landmark
    #
    # IN: landmark_to_country, dict: str -> str
    #     > maps landmark_id to its country
    #
    # IN: remaining_countries, list[str]
    #     > list of country names with at least MIN_LANDMARKS landmarks
    #
    # READS FROM:
    #
    # OUT: country_to_landmark_map, dict: str -> set(str)
    #     > maps a country name to a set of its landmarks by landmark id
    #
    # WRITES TO: METADATA_PATH - master data table
    with open(METADATA_PATH, 'w', newline='') as csv_out:
        writer = csv.writer(csv_out)
        country_to_number_of_landmarks = defaultdict(int)
        landmark_to_number_of_photos = defaultdict(int)
        writer.writerow(["image_id", "landmark_id", "country_name"])
        for landmark_id in landmark_to_photoset.keys():

            if landmark_id not in landmark_to_country:
                continue

            for landmark_photo_id in landmark_to_photoset[landmark_id]:
                country_name = landmark_to_country[landmark_id]
                if country_name in remaining_countries:
                    country_to_number_of_landmarks[country_name] += 1
                    landmark_to_number_of_photos[landmark_id] += 1
                    writer.writerow([landmark_photo_id, landmark_id, country_name])
        return country_to_number_of_landmarks, landmark_to_number_of_photos


################ READ-FROM-METADATA FUNCTIONS ################

def partition_landmarks():
    # IN: 
    #
    # READS FROM:
    #
    # OUT: train_landmarks, val_landmarks, test_landmarks, set(str)s
    #      > partitions the set of landmark_ids into three disjoint sets
    landmark_to_country = load_landmark_to_country()
    country_to_landmarks = load_country_to_landmarks(landmark_to_country)

    train_landmarks = set()
    val_landmarks = set()
    test_landmarks = set()

    for _, landmarks in country_to_landmarks.items():
        landmarks = list(landmarks)
        random.shuffle(landmarks)

        n = len(landmarks)
        n_train = int(TRAIN_FRAC * n)
        n_val = int(VAL_FRAC * n)

        train_landmarks.update(landmarks[:n_train])
        val_landmarks.update(landmarks[n_train:n_train + n_val])
        test_landmarks.update(landmarks[n_train + n_val:])
    
    return train_landmarks, val_landmarks, test_landmarks

def write_splits(train_set, val_set, test_set):
    # IN: train_landmarks, val_landmarks, test_landmarks, set(str)s
    #      > partitions the set of landmark_ids into three disjoint sets
    #
    # READS FROM: METADATA_PATH
    #
    # WRITES-TO: SPLIT_DIR / {test, train, val}_{images, countries, landmarks}.txt
    #      > writes the image-name, country_name, and landmark_ name to txt files
    #        to be used by pytorch
    with (open(METADATA_PATH, 'r', newline='') as csv_in):

        metadata_reader = csv.reader(csv_in)
        header = next(csv_in)

        file_outs = {
            (tvt, id_type): open(f"{SPLIT_DIR}/{tvt}_{id_type}.txt", "w")
            for tvt in TVTs
            for id_type in IDs
        }

        for row in metadata_reader:
            tvt_idx = 0
            image_id = IMAGE_DIR / row[0]
            landmark_id = row[1]
            country_id = row[2]

            if landmark_id in train_set:
                print(landmark_id)
                tvt_idx = 0

            elif landmark_id in val_set:
                tvt_idx = 1

            elif landmark_id in test_set:
                tvt_idx = 2

            file_outs[(TVTs[tvt_idx],IDs[0])].write(str(image_id) + "\n")
            file_outs[(TVTs[tvt_idx],IDs[1])].write(landmark_id + "\n")
            file_outs[(TVTs[tvt_idx],IDs[2])].write(country_id + "\n")

        for f in file_outs.values():
            f.close()

if __name__ == "__main__":
    print("hello.")