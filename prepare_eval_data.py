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
        "SELECT trim(machine_name) AS mn FROM game_alias WHERE status = 'confirmed'")
    conn.commit()
    ls = []
    for record in cur:
        if record[0] == 'bf5' or record[0] == 'bfv' or record[0] == 'battlefield_v' or record[0] == 'battlefield_5':
            ls.append((1, record[0], 'bf_v'))
        else:
            ls.append((0, record[0], 'bf_v'))

    return prepare_for_training(ls, 'validation.eval')


if __name__ == '__main__':
    main()
    conn.close()
