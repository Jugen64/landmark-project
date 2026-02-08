import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path("~/Documents/Code/projects/datasets").expanduser()

METADATA_OUT_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"

GLDV2_CSV_PATH = DATASET_DIR  / "gldv2_micro/gldv2_micro.csv"
IMAGE_DIR = DATASET_DIR / "gldv2_micro/images"

LANDMARK_TO_COUNTRY_CSV_PATH = DATASET_DIR / "landmark_to_country/landmark_to_country.csv"

MIN_LANDMARKS = 20


# --- vvv loads LANDMARK -> COUNTRY map as python dict 'landmark_to_country' vvv ---
def landmark_to_country_fxn(path):
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
def country_to_landmarks_fxn(path, landmark_to_country):
    country_to_landmarks = defaultdict(set)
    with open(path, newline='') as csv_in:
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


# --- vvv filters COUNTRIES by number of unique LANDMARKS vvv --- #
def filter_countries(country_to_landmarks, threshold):
    remaining_countries = {x for x in country_to_landmarks.keys() if len(country_to_landmarks[x]) > threshold }
    print(f"remaining_countries = {remaining_countries}")
    print(f"country pool size: {len(remaining_countries)}")
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


def combine_and_write_out(path, landmark_to_example, landmark_to_country):
    with open(path, 'w', newline='') as csv_out:
        writer = csv.writer(csv_out)
        writer.writerow(["image_id", "landmark_id", "country_name"])
        for landmark_id in landmark_to_example.keys():

            if landmark_id not in landmark_to_country:
                continue
            example_image_id = landmark_to_example[landmark_id]
            country_name = landmark_to_country[landmark_id]
            writer.writerow([example_image_id, landmark_id, country_name])

    
def main():
    missing_images = 0

    # --- vvv loads LANDMARK -> COUNTRY map as python dict 'landmark_to_country' vvv --- #
    landmark_to_country = landmark_to_country_fxn(LANDMARK_TO_COUNTRY_CSV_PATH)

    # --- vvv builds COUNTRY -> IMAGE map as python dict 'coutnry_to_landmark' vvv --- #
    country_to_landmarks = country_to_landmarks_fxn(GLDV2_CSV_PATH, landmark_to_country)

    # --- vvv filters COUNTRIES by number of unique LANDMARKS vvv --- #
    remaining_countries = filter_countries(country_to_landmarks, MIN_LANDMARKS)
    print(f"remaining countries: {remaining_countries}")
    
    # --- vvv associates one EXAMPLE PHOTO to each LANDMARK --- #
    landmark_to_example, filtered_landmarks = landmark_to_single_photo(GLDV2_CSV_PATH)
    print(f"Number of filtered landmarks: {filtered_landmarks}")

    # --- vvv writes [photo_id, landmark_id, country_name] (1 per. landmark) to METADAT_OUT_PATH--- #
    combine_and_write_out(METADATA_OUT_PATH, landmark_to_example, landmark_to_country)
    
    print(f"Process completed.")


if __name__ == "__main__":
    main()