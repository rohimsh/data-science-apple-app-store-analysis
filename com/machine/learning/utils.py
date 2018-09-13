import pandas as pd

app_genre_map = {
'Book': 10000001,
'Business': 10000002,
'Catalogs': 10000003,
'Education': 10000004,
'Entertainment': 10000005,
'Finance': 10000006,
'Food & Drink': 10000007,
'Games': 10000008,
'Health & Fitness': 10000009,
'Lifestyle': 10000010,
'Medical': 10000011,
'Music': 10000012,
'Navigation': 10000013,
'News': 10000014,
'Photo & Video': 10000015,
'Productivity': 10000016,
'Reference': 10000017,
'Shopping':10000018,
'Social Networking': 10000019,
'Sports': 10000020,
'Travel': 10000021,
'Utilities': 10000022,
'Weather': 10000023
}

def main():
    # loading our data as a panda
    df = pd.read_csv('apple_app_store_data.csv', delimiter=",", dtype=str)
    grouped_data = df.groupby('app_genre', group_keys=True)
    training_file = open('training_data.csv', 'a')
    test_file = open('test_data.csv', 'a')
    training_file.write('app_name,app_size_bytes,app_price,app_ratings_count,app_rating,app_genre\n')
    test_file.write('app_name,app_size_bytes,app_price,app_ratings_count,app_rating,app_genre\n')
    count = 0;
    for genre, group in grouped_data:
        # print(genre)
        for row, data in group.iterrows():
            # print(data)
            count = count + 1
            if count % 2 == 0:
                training_file.write(
                    data['app_name'] + ',' + data['app_size_bytes'] + ',' + data['app_price'] + ',' + data[
                        'app_ratings_count'] + ',' + data['app_rating'] + ',' + str(app_genre_map[data['app_genre']]) + ' \n')
            else:
                test_file.write(data['app_name'] + ',' + data['app_size_bytes'] + ',' + data['app_price'] + ',' + data[
                    'app_ratings_count'] + ',' + data['app_rating'] + ',' + str(app_genre_map[data['app_genre']]) + ' \n')
    training_file.close()
    test_file.close()


if __name__ == '__main__':
    main()
