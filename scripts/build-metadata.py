import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path("~/Documents/Code/projects/datasets").expanduser()

CSV_OUT_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"

CSV_IN_PATH = DATASET_DIR  / "gldv2_micro/gldv2_micro.csv"
IMAGE_DIR = DATASET_DIR / "gldv2_micro/images"

LANDMARK_TO_COUNTRY_CSV_PATH = DATASET_DIR / "landmark_to_country/landmark_to_country.csv"






def main():
    missing_images = 0
    landmark_to_country = {}
    country_to_landmark = defaultdict(set)

    # --- vvv loads LANDMARK -> COUNTRY map as python dict 'landmark_to_country' vvv ---

    with open(LANDMARK_TO_COUNTRY_CSV_PATH, newline='') as lm_t_c:
        landmark_to_country_reader = csv.reader(lm_t_c)
        for row in landmark_to_country_reader:
            country = row[1]
            id = row[0]
            landmark_to_country[id] = country
        print(landmark_to_country['185116'])

    # --- ^^^ loads LANDMARK -> COUNTRY map as python dict 'landmark_to_country' ^^^ ---


    # --- vvv builds COUNTRY -> IMAGE map as python dict 'coutnry_to_landmark' vvv ---

    with open(CSV_IN_PATH, newline='') as csv_in, open(CSV_OUT_PATH, "w", newline="") as csv_out:
        reader = csv.reader(csv_in)
        writer = csv.writer(csv_out)
        print(f"CSV_IN_PATH={CSV_IN_PATH}")
        writer.writerow(["image_id", "landmark_id", "country_id"])

        for row in reader:
            image_filename = row[0]
            landmark_id = row[1]
            if landmark_id not in landmark_to_country:
                continue

            image_path = IMAGE_DIR / f"{image_filename}"
            if not image_path.exists():
                missing_images += 1
                continue

            if landmark_id not in landmark_to_country:
                continue
            
            country = landmark_to_country[landmark_id]
            writer.writerow([image_filename, landmark_id, country])

        print(f"Wrote to {CSV_OUT_PATH}, skipped {missing_images} missing images.")

if __name__ == "__main__":
    main()