import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_OUT_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
RAW_ROOT = Path("/Users/jblee/Documents/Code/projects/datasets/gldv2_micro").expanduser()
CSV_IN_PATH = RAW_ROOT / "gldv2_micro.csv"
IMAGE_DIR = RAW_ROOT / "images"






def main():
    missing_images = 0
    with open(CSV_IN_PATH, newline='') as csv_in, open(CSV_OUT_PATH, "w", newline="") as csv_out:
        reader = csv.reader(csv_in)
        writer = csv.writer(csv_out)
        print(f"CSV_IN_PATH={CSV_IN_PATH}")
        writer.writerow(["image_id", "landmark_id", "country_id"])

        for row in reader:
            image_filename = row[0]
            landmark_id = row[1]
            print(f"filename={image_filename}, landmark_id={landmark_id}")

            image_path = IMAGE_DIR / f"{image_filename}"
            if not image_path.exists():
                missing_images += 1
                continue

            country_id = 0
            writer.writerow([image_filename, landmark_id, country_id])
        print(f"Wrote to {CSV_OUT_PATH}, skipped {missing_images} missing images.")

if __name__ == "__main__":
    main()