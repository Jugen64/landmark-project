import csv
from pathlib import Path
from collections import defaultdict

from src.utils import metadata_utils

def main():
    landmark_to_country = metadata_utils.load_landmark_to_country()
    country_to_landmarks = metadata_utils.load_country_to_landmarks(landmark_to_country)

    remaining_countries = list(metadata_utils.filter_countries(country_to_landmarks))
    remaining_countries.sort()
    
    landmark_to_photoset_map = metadata_utils.landmark_to_photoset()
    
    country_to_number_of_landmarks, landmark_to_number_of_photos = metadata_utils.combine_and_write_out(landmark_to_photoset_map, landmark_to_country, remaining_countries)
    
    for country_name in sorted(country_to_number_of_landmarks.keys()):
        print(f"{country_name} has {country_to_number_of_landmarks[country_name]} landmarks.")
    print(f"Process completed.")


if __name__ == "__main__":
    main()