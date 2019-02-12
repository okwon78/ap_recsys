import csv
from collections import defaultdict
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pymongo


def save(filename, list_values):
    if not isinstance(list_values, list):
        print(f"Invalid data type: {filename}, {type(list_values)}")
        return

    np.save(filename, list_values)


def load(filename):
    return np.load(filename)


def draw_hist(values, step=1000):
    start = min(values)
    stop = max(values)
    #stop = min(max(values), 5000)

    boxes = range(start, stop, step)

    fig, ax = plt.subplots()
    ax.hist(x=values, bins=boxes, histtype='bar', rwidth=0.8)

    ax.set_xlabel('count')
    ax.set_ylabel('the number of sku')
    ax.set_title(f'Data distribution (max count: {stop})')

    plt.show()


def add_cart_data_from_csv(recsys_db):
    recsys_db.records.delete_many({})
    recsys_db.items.delete_many({})

    records = dict()
    item_count = defaultdict(int)

    with open('./apmall_cart.csv', 'r') as f:
        header = f.readline()

        for cols in csv.reader(f, delimiter=','):
            try:
                comcsno = int(cols[0].strip())
                tmp = cols[2].strip()
                insert_time = datetime.strptime(tmp, '%Y%m%d%H%M%S').timestamp()
                insert_time = int(insert_time)
                itemId = cols[3].strip()
                buy_flag = True if cols[5].strip() == 'Y' else False

                if comcsno in records:
                    record = records[comcsno]
                else:
                    record = {
                        'comcsno': comcsno,
                        'itemIds': []
                    }

                    records[comcsno] = record

                record['itemIds'].append({
                    'itemId': itemId,
                    'purchased': buy_flag,
                    'timestamp': insert_time
                })

                item_count[itemId] = item_count[itemId] + 1

            except Exception as e:
                print(f'Failed to upload to ftp: {str(e)}')
                return

    user_idx = 0
    for record in records.values():

        if len(record['itemIds']) > 2:
            continue

        record['itemIds'].sort(key=lambda item: int(item['timestamp']))
        record['sorted_items'] = [item['itemId'] for item in record['itemIds']]
        record['user_index'] = user_idx
        user_idx += 1

    items = list()
    for itemId, count in item_count.items():
        items.append({
            'itemId': itemId,
            'count': count
        })
    records = list(records.values())
    item_count = list(item_count.values())

    save('records', records)
    save('items', items)
    save('item_count', item_count)

    recsys_db.records.insert_many(records)
    recsys_db.items.insert_many(items)

    draw_hist(item_count)


def add_cart_data_from_file(recsys_db):
    recsys_db.records.delete_many({})
    recsys_db.items.delete_many({})

    records = load('records.npy')
    items = load('items.npy')
    item_count = load('item_count.npy')

    #recsys_db.records.insert_many(records)
    #recsys_db.items.insert_many(items)

    draw_hist(item_count)


def main():
    client = pymongo.MongoClient(host='13.209.6.203',
                                 port=27017,
                                 username='romi',
                                 password="Amore12345!",
                                 authSource='admin',
                                 authMechanism='SCRAM-SHA-256')

    db = client.recsys_apmall
    add_cart_data_from_file(db)
    # add_cart_data_from_csv(db)
    print("end")


if __name__ == '__main__':
    print('matplotlib: ', matplotlib.__version__)
    main()
