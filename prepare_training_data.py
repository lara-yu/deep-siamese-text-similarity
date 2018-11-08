import psycopg2
import csv

conn = psycopg2.connect("dbname=feedback_nlp user=postgres password=postgres host=data-platform-sh-01")


def prepare_for_training(train, file):
    with open(file, 'w', encoding='UTF-8',
              newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        return writer.writerows(train)


def main():
    cur = conn.cursor()
    cur.execute(
        "SELECT trim(formal_name) AS gn, trim(machine_name) AS mn "
        "FROM game_alias WHERE status = 'confirmed' AND formal_name != machine_name")
    conn.commit()

    _dict = {}
    ls = []

    for record in cur:
        if record[0] != record[1]:
            if record[0] in _dict:
                _dict[record[0]].append(record[1])
            else:
                _dict[record[0]] = [record[1]]

    for idx in _dict:
        # _dict1 = _dict.copy()
        # _dict1.pop(key)

        # for i in _dict1:
        #     ls.append((i, key, 'n'))
        val = _dict[idx]
        if len(val) > 1:
            for j in val:
                for i in range(len(val)):
                    if j != val[i]:
                        ls.append((j, val[i], 'y'))

    return prepare_for_training(ls, 'title_db_names.train')


if __name__ == '__main__':
    main()
    conn.close()
