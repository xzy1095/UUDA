import matplotlib.pyplot as plt

def f1():
    num_users_list = [2, 4, 8, 16, 32, 50, 100]
    results21 = [98.8, 95.9, 96.7, 95.6, 95.8, 95.6, 94.7] #21
    results11 = [92.3, 93.1, 93.6, 93.2, 93.2, 93.1, 92.5] #11
    results12 = [92.2, 93.0, 91.3, 91.3, 90.6, 90.0, 90.1] #12

    x_positions = range(len(num_users_list))

    plt.plot(x_positions, results21, marker='o', label='M/S=2:1')
    plt.plot(x_positions, results11, marker='o', label='M/S=1:1')
    plt.plot(x_positions, results12, marker='o', label='M/S=1:2')

    plt.title('Multi-Fashion Results vs Number of Users')
    plt.xlabel('Number of Users')
    plt.ylabel('Results (%)')

    plt.xticks(x_positions, num_users_list)

    plt.legend()

    plt.grid(True)
    plt.ylim(85, 100)

    plt.show()


def f2():
    sample_num_list = [120, 140, 160, 180, 200]
    results21 = [94.5, 94.8, 95.1, 95.2, 95.5] #21
    results11 = [91.5, 92.0, 92.2, 92.6, 92.8] #11
    results12 = [88.4, 88.4, 89.2, 89.7, 90.1] #12

    x_positions = range(len(sample_num_list))

    plt.plot(x_positions, results21, marker='o', label='M/S=2:1')
    plt.plot(x_positions, results11, marker='o', label='M/S=1:1')
    plt.plot(x_positions, results12, marker='o', label='M/S=1:2')

    plt.title('Multi-Fashion Results vs Number of Samples')
    plt.xlabel('Number of Samples')
    plt.ylabel('Results (%)')

    plt.xticks(x_positions, sample_num_list)

    plt.legend()

    plt.grid(True)
    plt.ylim(85, 100)

    plt.show()


f1()
f2()
