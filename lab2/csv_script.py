import csv
import random

with open('mnist_test.csv', 'r') as mnist_test:
    teach_data = []
    test_data = []

    cursor_count = 0
    
    count0 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count8 = 0

    csv_reader = csv.reader(mnist_test)
    for row in csv_reader:
        if count0 == 200 and count2 == 200 and count3 == 200 and count4 == 200 and count8 == 200:
            count0 = count2 = count3 = count4 = count8 = 0
            break

        if row[0] == '0' and count0 != 200:
            count0 += 1
            teach_data.append(row)
        if row[0] == '2' and count2 != 200:
            count2 += 1
            teach_data.append(row)
        if row[0] == '3' and count3 != 200:
            count3 += 1
            teach_data.append(row)
        if row[0] == '4' and count4 != 200:
            count4 += 1
            teach_data.append(row)
        if row[0] == '8' and count8 != 200:
            count8 += 1
            teach_data.append(row)
    print(len(teach_data))

    random.shuffle(teach_data)

    with open('amogus_teach.csv', 'x', newline='') as amogus_teach:
        writer = csv.writer(amogus_teach)
        for row in teach_data:
            writer.writerow(row)

    for row in csv_reader:
        cursor_count += 1

        if count0 == 100 and count2 == 100 and count3 == 100 and count4 == 100 and count8 == 100:
            count0 = count2 = count3 = count4 = count8 = 0
            break

        if cursor_count > 5999:
            if row[0] == '0' and count0 != 100:
                count0 += 1
                test_data.append(row)
            if row[0] == '2' and count2 != 100:
                count2 += 1
                test_data.append(row)
            if row[0] == '3' and count3 != 100:
                count3 += 1
                test_data.append(row)
            if row[0] == '4' and count4 != 100:
                count4 += 1
                test_data.append(row)
            if row[0] == '8' and count8 != 100:
                count8 += 1
                test_data.append(row)
    print(len(test_data))

    random.shuffle(test_data)

    with open('amogus_test.csv', 'x', newline='') as amogus_test:
        writer = csv.writer(amogus_test)
        for row in test_data:
            writer.writerow(row)
    
    
    

