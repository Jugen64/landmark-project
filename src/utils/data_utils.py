import csv
from pathlib import Path
from collections import defaultdict

DATASET_DIR = Path("~/Documents/Code/projects/datasets").expanduser()
IMAGE_DIR = DATASET_DIR / "gldv2_micro/images"
METADATA_PATH = Path("~/Documents/Code/projects/landmark_project/data/processed/metadata.csv").expanduser()

# --- vvv loads LANDMARK -> COUNTRY map as python dict 'landmark_to_country' vvv ---
def load_landmark_to_country(path):
    landmark_to_country = {}
    with open(path, newline='') as lm_t_c:
        landmark_to_country_reader = csv.reader(lm_t_c)
        header = next(landmark_to_country_reader)
        for row in landmark_to_country_reader:
            country = row[1]
            landmark_id = row[0]
            landmark_to_country[landmark_id] = country
    return landmark_to_country

# --- vvv builds COUNTRY -> IMAGE map as python dict 'coutnry_to_landmark' vvv ---
def load_country_to_landmarks(data_path, image_dir_path, landmark_to_country):
    country_to_landmarks = defaultdict(set)
    with open(data_path, newline='') as csv_in:
        reader = csv.reader(csv_in)
        header = next(reader)
        
        for row in reader:
            image_filename = row[0]
            landmark_id = row[1]
            if landmark_id not in landmark_to_country:
                continue

            image_path = image_dir_path / f"{image_filename}"
            if not image_path.exists():
                missing_images += 1
                continue
            
            country_name = landmark_to_country[landmark_id]
            country_to_landmarks[country_name].add(landmark_id)
    return country_to_landmarks

# --- vvv filters COUNTRIES by number of unique LANDMARKS vvv --- #
def filter_countries(country_to_landmarks, threshold):
    remaining_countries = {x for x in country_to_landmarks.keys() if len(country_to_landmarks[x]) > threshold }
    return remaining_countries

# --- vvv associates one EXAMPLE PHOTO to each LANDMARK --- #
def landmark_to_single_photo(path):
    landmark_to_example = {}
    filtered_landmarks = 0
    with open(path, 'r', newline='') as csv_in:
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

# --- vvv associates a SET OF PHOTOS to each LANDMARK --- #
def landmark_to_photoset_fxn(path):
    landmark_to_photoset = defaultdict(set)
    filtered_landmarks = 0
    with open(path, 'r', newline='') as csv_in:
        reader = csv.reader(csv_in)
        header = next(reader)
        for row in reader:
            image_filename = row[0]
            landmark_id = row[1]
            landmark_to_photoset[landmark_id].add(image_filename)
    return landmark_to_photoset

def combine_and_write_out(path, landmark_to_photoset, landmark_to_country, remaining_countries):
    with open(path, 'w', newline='') as csv_out:
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


def metadata_landmark_to_country(path):
    landmark_to_country = {}
    with open(path, newline='') as lm_t_c:
        landmark_to_country_reader = csv.reader(lm_t_c)
        header = next(landmark_to_country_reader)
        for row in landmark_to_country_reader:
            country = row[2]
            landmark_id = row[1]
            landmark_to_country[landmark_id] = country
    return landmark_to_country

def metadata_country_to_landmarks(path):
    country_to_landmarks = defaultdict(set)
    with open(path, newline='') as lm_t_c:
        landmark_to_country_reader = csv.reader(lm_t_c)
        header = next(landmark_to_country_reader)
        for row in landmark_to_country_reader:
            country = row[2]
            landmark_id = row[1]
            country_to_landmarks[country].add(landmark_id)
    return country_to_landmarks

def metadata_landmark_to_image_path(metadata_path):
    landmark_to_image_paths = defaultdict(set)
    with open(metadata_path, newline='') as meta:
        metadata_reader = csv.reader(meta)
        header = next(metadata_reader)
        for row in metadata_reader:
            image_name = row[0]
            landmark_id = row[1]
            image_path = IMAGE_DIR / image_name
            landmark_to_image_paths[landmark_id].add(image_path)
    return landmark_to_image_paths

def expand_landmarks(landmark_ids, landmark_to_images, landmark_to_label):
    image_paths = []
    labels = []

    for landmark_id in landmark_ids:
        for img_path in landmark_to_images[landmark_id]:
            image_paths.append(img_path)
            labels.append(landmark_to_label[landmark_id])

    return image_paths, labels

def split_to_list(path):
    path_list = []
    with open(path, 'r', newline='') as file_in:
        split_reader = csv.reader(file_in)
        for row in split_reader:
            path_list.append(row[0])
    return list(path_list)

def metadata_countries():
    country_list = []
    with open(METADATA_PATH, 'r', newline='') as file_in:
        metadata_reader = csv.reader(file_in)
        for row in metadata_reader:
            country = row[2]
            country_list.append(country)
    return list(country_list)

if __name__ == "__main__":
    print("hello.")