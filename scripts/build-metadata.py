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



def main():
    missing_images = 0
    landmark_to_country = {}
    country_to_landmarks = defaultdict(set)

    # --- vvv loads LANDMARK -> COUNTRY map as python dict 'landmark_to_country' vvv ---

    with open(LANDMARK_TO_COUNTRY_CSV_PATH, newline='') as lm_t_c:
        landmark_to_country_reader = csv.reader(lm_t_c)
        header = next(landmark_to_country_reader)
        for row in landmark_to_country_reader:
            country = row[1]
            landmark_id = row[0]
            landmark_to_country[landmark_id] = country
        print(landmark_to_country['185116'])

    # --- ^^^ loads LANDMARK -> COUNTRY map as python dict 'landmark_to_country' ^^^ ---


    # --- vvv builds COUNTRY -> IMAGE map as python dict 'coutnry_to_landmark' vvv ---

    with open(GLDV2_CSV_PATH, newline='') as csv_in:
        GLDV2_reader = csv.reader(csv_in)
        header = next(GLDV2_reader)
        
        for row in GLDV2_reader:
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

        print(f"Test: portugal = {country_to_landmarks["Portugal"]}")

    # --- ^^^ builds COUNTRY -> LANDMARKS map as python dict 'coutnry_to_landmark' ^^^ ---

    # --- vvv filters COUNTRIES by number of LANDMARKS vvv ---

    filtered_countries = {x for x in country_to_landmarks.keys() if len(country_to_landmarks[x]) > MIN_LANDMARKS }
    print(f"filtered_countries = {filtered_countries}")
    
    # --- ^^^ filters COUNTRIES by number of LANDMARKS ^^^ ---


    landmark_to_example = {}

    with open(GLDV2_CSV_PATH, 'r', newline='') as csv_in:
        GLDV2_reader = csv.reader(csv_in)
        header = next(GLDV2_reader)
        for row in GLDV2_reader:
            image_filename = row[0]
            landmark_id = row[1]
            print(f"landmark_id={landmark_id}")
            if landmark_id not in landmark_to_example:
                print(f"landmark {landmark_id} associated with image {image_filename}")
                landmark_to_example[landmark_id] = image_filename

    with open(METADATA_OUT_PATH, 'w', newline='') as csv_out:
        writer = csv.writer(csv_out)
        writer.writerow(["image_id", "landmark_id", "country_name"])
        for landmark_id in landmark_to_example.keys():

            print(f"landmark_id={landmark_id}")
            if landmark_id not in landmark_to_country:
                continue
            example_image_id = landmark_to_example[landmark_id]
            country_name = landmark_to_country[landmark_id]
            writer.writerow([example_image_id, landmark_id, country_name])

    print(f"Process completed.")


if __name__ == "__main__":
    main()