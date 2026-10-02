from scrape_sf import scrape_sf_data
from scrape_meteo import scrape_meteo_data


def main():
    sf_df = scrape_sf_data()

    meteo_df = scrape_meteo_data()

    print("SF Data:")
    print(sf_df.head())
    print("\nMeteo Data:")
    print(meteo_df.head())


if __name__ == "__main__":
    main()