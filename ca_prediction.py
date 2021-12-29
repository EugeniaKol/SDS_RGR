import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from collections import Counter

# завантажуємо датасет та виводимо інформацію про нього
data = pd.read_csv('KDDcup99.csv')
print('\nІнформація по датасету KDDcup99.csv:\n')
data.info()
print()

# обираємо з датасету лише дані з необхідним типом атак та підрахуємо кількість таких рядків
data1 = data[(data["label"] == 'guess_passwd') | (data["label"] == 'ftp_write')]
print('Кількість загрз типу normal: ', len(data[data["label"] == 'normal']))
print('Кількість загроз типу guess_passwd: ', len(data[data["label"] == 'guess_passwd']))
print('Кількість загроз типу ftp_write: ', len(data[data["label"] == 'ftp_write']))
print('Кількість загроз типу guess_passwd та ftp_write: ', len(data1))

# додаємо таку ж саму кількість нормальних даних та об`єднуємо всі рядки в один датасет
data_normal = data.loc[data["label"] == 'normal'].iloc[:61]
data = pd.concat([data_normal, data1])
print(data)
print()

# виводимо інформацію про отриманий об'єднаний датасет і кількість загроз кожного виду у ньому
data.info()
print('Вибірка до кодування: ', data['protocol_type'])
print(Counter(data[0:]))
print()

print('\nСтовпець protocol_type до кодування: ', data['protocol_type'])
print('Стовпець service до кодування: ', data['service'])
print('Стовпець flag до кодування: ', data['flag'])
# Кодування (всі слова що у нас є заміняємо числами бо слова нам не підходять)
# Так як у нас три стовпці (крім label) містять строкові значення, заміняємо їх числами
# Отримуємо унікальні значення строкових стовпців
protocol_type_types = data['protocol_type'].unique()
service_types = data['service'].unique()
flag_types = data['flag'].unique()

# створюємо словники для заміни слів на числа (від 1го і далі)
protocol_type_replace = dict(zip(protocol_type_types, range(0, len(protocol_type_types))))
service_replace = dict(zip(service_types, range(0, len(service_types))))
flag_replace = dict(zip(flag_types, range(0, len(flag_types))))
all_for_replace = {'protocol_type': protocol_type_replace, 'service': service_replace, 'flag': flag_replace}
# робимо заміну значень стовпців на числа
data = data.replace(all_for_replace)
print('\nСтовпець protocol_type після кодування: ', data['protocol_type'])
print('Стовпець service після кодування: ', data['service'])
print('Стовпець flag після кодування: ', data['flag'])

# Робимо розбиття вибірки на вхідні (X) та вихідні (Y) дані
Y = data["label"]
X = data.drop(["label"], axis = 'columns')

# Проводимо нормалізацію по X
for column in X:
    if np.max(X[column]) - np.min(X[column]) not in (0, 0.0):
        X[column] = (X[column] - np.min(X[column]))/(np.max(X[column]) - np.min(X[column]))
print(X)

# розбиття на тренувальну та тестову вибірку
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=33, test_size=0.8, stratify=Y)
print(Counter(Y_train))
print(Counter(Y_test))
print('Довжина X_train: ', len(X_train))
print('Довжина X_test: ', len(X_test))
print('Довжина Y_train: ', len(Y_train))
print('Довжина Y_test: ', len(Y_test))
print('\nX_train\n', X_train)
print('X_test\n', X_test)
print('Y_train\n', Y_train)
print('Y_test\n', Y_test)

# навчання мережі
# виділяємо унікальні класи з вихідних даних
Y_un = np.unique(Y)
number_class = len(Y_un)

# ініціація матриці вагів
# створюємо матриці заповнені нулями потрібного розміру для вагів кожного шару
w1 = np.zeros((X_train.shape[0], X_train.shape[1]))
w2 = np.zeros((number_class, X_train.shape[0]), dtype=bool)
# встановлюємо ваги для шару образів, ваги ідентичні вхідним даним
w1 = np.copy(X_train)
# встановлюємо ваги для шару додавання, порівнюємо очікувані класи та поточні класи для кожного нейрона
for i in range(number_class):
    w2[i] = (Y_train == Y_un[i])

# Будуємо нейрону мережу
def PNN(x_new):
    image_layer = np.zeros(X_train.shape[0]) # створюємо список з нулів, кількість дорівнює кількості навчальних прикладів

    # Шар образів
    for i in range(X_train.shape[0]):
        image_layer[i] = np.exp(np.sum(-(w1[i] - x_new)**2 / 0.3 **2)) # Формула для отримання шару образів
    # print("Шар образів = ", image_layer)

    # Шар додавання
    sum_layer = np.zeros(number_class) # створюємо масив sum_layer розміром по кількості унікальних класів(2), заповнений нулями
    for i in range(number_class):
        sum_layer[i] = np.mean(image_layer[w2[i]])
    # print("Шар додавання = ", sum_layer)
    return Y_un[np.argmax(sum_layer)]

# Етап розпізнавання
for i in range(X_test.shape[0]):
    print('Очікуваний результат: ', Y_test.iloc[i], 'Отриманий результат: ', PNN(X_test.iloc[i]))
