import csv
from pathlib import Path
from collections import defaultdict

from src.utils import data_utils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path("~/Documents/Code/projects/datasets").expanduser()

METADATA_OUT_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"

GLDV2_CSV_PATH = DATASET_DIR  / "gldv2_micro/gldv2_micro.csv"
IMAGE_DIR = DATASET_DIR / "gldv2_micro/images"

LANDMARK_TO_COUNTRY_CSV_PATH = DATASET_DIR / "landmark_to_country/landmark_to_country.csv"

MIN_LANDMARKS = 20

    
def main():
    missing_images = 0

    # --- vvv loads LANDMARK -> COUNTRY map as python dict 'landmark_to_country' vvv --- #
    landmark_to_country = data_utils.load_landmark_to_country(LANDMARK_TO_COUNTRY_CSV_PATH)

    # --- vvv builds COUNTRY -> IMAGE map as python dict 'coutnry_to_landmark' vvv --- #
    country_to_landmarks = data_utils.load_country_to_landmarks(GLDV2_CSV_PATH, IMAGE_DIR, landmark_to_country)

    # --- vvv filters COUNTRIES by number of unique LANDMARKS vvv --- #
    remaining_countries = list(data_utils.filter_countries(country_to_landmarks, MIN_LANDMARKS))
    remaining_countries.sort()
    # print(f"Remaining countries: {remaining_countries}")
    print(f"Country pool size: {len(remaining_countries)}")
    
    # --- vvv associates one EXAMPLE PHOTO to each LANDMARK --- #
    # landmark_to_example, filtered_landmarks = landmark_to_single_photo(GLDV2_CSV_PATH)
    # print(f"Number of filtered landmarks: {filtered_landmarks}")

    landmark_to_photoset_map = data_utils.landmark_to_photoset_fxn(GLDV2_CSV_PATH)
    

    # --- vvv writes [photo_id, landmark_id, country_name] (1 per. landmark) to METADAT_OUT_PATH--- #
    country_to_number_of_landmarks, landmark_to_number_of_photos = data_utils.combine_and_write_out(METADATA_OUT_PATH, landmark_to_photoset_map, landmark_to_country, remaining_countries)
    
    for country_name in sorted(country_to_number_of_landmarks.keys()):
        print(f"{country_name} has {country_to_number_of_landmarks[country_name]} landmarks.")
    print(f"Process completed.")


if __name__ == "__main__":
    main()