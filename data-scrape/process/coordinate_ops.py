import requests
import time
import os
import json
import pandas as pd

station_id = {
    "01" : ["37.77940116861372", "-122.40426940334474"],
    "02" : ["37.79710979421665", "-122.40995266064301"],
    "03" : ["37.786700545822086", "-122.41937276064341"],
    "04" : ["37.77335748206085", "-122.38931572662781"],
    "05" : ["37.78049425009573", "-122.43060698577733"],
    "06" : ["37.76721190287737", "-122.43096092016697"],
    "07" : ["37.7602837999999", "-122.41515536064426"],
    "08" : ["37.777252551783434", "-122.3967304299567"],
    "09" : ["37.745227604548205", "-122.40118685879375"],
    "10" : ["37.785713904680726", "-122.44669181831517"],
    "11" : ["37.74868402216353", "-122.42649657413715"],
    "12" : ["37.76349885362452", "-122.4525316011213"],
    "13" : ["37.79552643811893", "-122.4015395490017"],
    "14" : ["37.77907311077708", "-122.48588432995669"],
    "15" : ["37.72338513732683", "-122.45287167413788"],
    "16" : ["37.798781313441275", "-122.43671661646364"],
    "17" : ["37.72755421501454", "-122.38500040297377"],
    "18" : ["37.751051519558416", "-122.49043065879357"],
    "19" : ["37.72801984330921", "-122.47901844715307"],
    "20" : ["37.75134003269091", "-122.45625618948046"],
    "21" : ["37.77569273861892", "-122.44041709133069"],
    "22" : ["37.76395634568694", "-122.47375189133108"],
    "23" : ["37.76156601802456", "-122.50481601831599"],
    "24" : ["37.75338234207149", "-122.44099807228581"],
    "25" : ["37.746769543171226", "-122.38693052995767"],
    "26" : ["37.74028755629141", "-122.43349894900366"],
    "28" : ["37.80274166905134", "-122.4094794778375"],
    "29" : ["37.7663562982512", "-122.40452357783856"],
    "31" : ["37.779973528954855", "-122.47101867598721"],
    "32" : ["37.73651974296607", "-122.421208189481"],
    "33" : ["37.71101127447205", "-122.45860714530247"],
    "34" : ["37.77959920826336", "-122.50264048762854"],
    "35" : ["37.79034367305018", "-122.38843028051014"],
    "36" : ["37.77529265804824", "-122.4210666146134"],
    "37" : ["37.75761894759794", "-122.3991960587933"],
    "38" : ["37.78953570695838", "-122.42990304715087"],
    "39" : ["37.740260248320766", "-122.45874333366032"],
    "40" : ["37.74777200706637", "-122.47509276858378"],
    "41" : ["37.793431269856676", "-122.41633803209163"],
    "42" : ["37.73169318068309", "-122.4054173281072"],
    "43" : ["37.71633987200489", "-122.43166171646641"],
    "44" : ["37.71667094600305", "-122.40050073180994"],
    "48" : ["37.826684920738884", "-122.3693111029703"],
    "49" : ["37.74523010454871", "-122.40116091461442"],
    "51" : ["37.80162617198334", "-122.45547441831457"]
}

def match_station(df, station_area):
    # there is no station house 27, 30, 45, 46, 47, 50

    coords = station_area.astype(str).map(station_id)
    df["station_long"] = coords.str[1]
    df["station_lat"] = coords.str[0]
    return df

def distance_between_coords(df):

    prev_distances = {}
    if os.path.exists("location_cache.json"):
        with open("location_cache.json", "r") as f:
            prev_distances = json.load(f)
    else:
        prev_distances = {}

    df['trip_name'] = (df['station_lat'].round(4).astype(str) + "," + df['station_long'].round(4).astype(str) + "->" + df['latitude'].round(4).astype(str) + "," + df['longitude'].round(4).astype(str))

    df['distance_km'] = df['trip_name'].map(prev_distances)

    missing_rows = df[df['distance_km'].isna()].copy()
    missing_rows = missing_rows.dropna(subset=['latitude', 'longitude', 'station_lat', 'station_long'])
    
    for station_id, group in missing_rows.groupby('station_area'):
        st_lon = group['station_long'].iloc[0]
        st_lat = group['station_lat'].iloc[0]

        for i in range(0, len(group), 50):

            batch = group[i : i + 50]

            coords_list = [f"{st_lon},{st_lat}"]

            for _, row in batch.iterrows():
                coords_list.append(f"{row['longitude']},{row['latitude']}")


            coords = ";".join(coords_list)

            dest_indexes = ";".join([str(x) for x in range(1, len(batch) + 1)])

            url = f"http://router.project-osrm.org/table/v1/driving/{coords}?sources=0&destinations={dest_indexes}&annotations=distance"
            print(st_lat, st_lon)
            try:
                response = requests.get(url)
                data = response.json()

                if data.get("code") == "Ok":
                    distance_metres = data['distances'][0]

                    distances_km = []
                    for d in distance_metres:
                        if d is not None:
                            distances_km.append(d / 1000)
                        else:
                            distances_km.append(None)

                    df.loc[batch.index, 'distance_km'] = distances_km

                    for trip_name, dist in zip(batch['trip_name'], distances_km):
                        prev_distances[trip_name] = dist

                    with open('location_cache.json', "w") as f:
                            json.dump(prev_distances, f)
                    
                time.sleep(1)
            except Exception as e:
                print(f"API Error: {e}")
   

    df = df.drop(columns=['trip_name'])

    return df