from scrape.scrape_sf import scrape_sf_data
from scrape.scrape_meteo import scrape_meteo_data

from process.process_sf import process_sf_data
from process.match_data import match_data


def main():
    raw_sf_df = scrape_sf_data()
    sf_df = process_sf_data(raw_sf_df)
    print("San Francisco Fire Department Data:")
    print(sf_df.head())

    meteo_df = scrape_meteo_data("2026-09-28", "2026-09-28")
    print("\nMeteo Data:")
    print(meteo_df.head())

    merged_df = match_data(sf_df, meteo_df)
    print("\nMerged Data:")
    print(merged_df.head())


if __name__ == "__main__":
    main()
