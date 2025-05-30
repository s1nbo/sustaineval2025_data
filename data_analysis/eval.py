from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# plot task_a_label
def plot_frequency(task_a_label, task_b_label):

    count_a = Counter(task_a_label)
    count_b = Counter(task_b_label)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(count_a.keys(), count_a.values())
    plt.xticks(list(count_a.keys()))
    plt.title('Task A Label')
    plt.xlabel('Label')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.scatter(count_b.keys(), count_b.values())
    plt.title('Task B Label')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.savefig('Frequency.png')


# Box plot, X axis is task_a_label, Y axis is task_b_label matched to task_a_label
def plot_frequencypertask(task_a_label, task_b_label):
    task_a_validation = defaultdict(list)
    for i in range(len(task_a_label)):
        task_a_validation[task_a_label[i]].append(task_b_label[i])
       
    # sort task_a_validation by key
    task_a_validation = dict(sorted(task_a_validation.items()))
    
    plt.figure(figsize=(10, 5))
    plt.boxplot(task_a_validation.values())
    plt.xticks(list(task_a_validation.keys()))
    plt.title('Task A Label vs Task B Label')
    plt.xlabel('Task A Label')
    plt.ylabel('Task B Label')
    plt.savefig('Task_A_vs_Task_B.png')


# Find correlation between task_a_label and task_b_label and given year in data
def plot_correlation(task_b_label, year):

    year_list = [[] for _ in range(6)]
    for i in range(len(year)):
        year_list[(year[i]-2016)].append(task_b_label[i])

    plt.figure(figsize=(10, 5))
    plt.boxplot(year_list)
    # increase the x axis tick names by 2015
    plt.xticks([1, 2, 3, 4, 5, 6], ['2016', '2017', '2018', '2019', '2020', '2021'])
    plt.title('Task B Label vs Year')
    plt.xlabel('Year')
    plt.ylabel('Task B Label')
    plt.savefig('Task_B_vs_Year.png')

# plot word count for total words

def word_count_analysis(word_count:dict, occurence:int):
        print(f'Length of word_count: {len(word_count)}')
        f''' give length of word_count with more than {occurence} occurences'''
        print(len([key for key in word_count.keys() if word_count[key] > occurence]))
        print('Ten most common words:')
        for key, value in sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f'{key}: {value}')


# save the word count to a file
def save_word_count(word_count:dict, occurence:int, a_label:int):
    with open('word_count.txt', 'a') as f:
        f.write(f'Class {a_label}\n')
        f.write(f'Length of word_count: {len(word_count)}\n')
        f.write(f'Length of word_count with more than {occurence} occurences: {len([key for key in word_count.keys() if word_count[key] > occurence])}\n')
        f.write('Ten most common words:\n')
        for key, value in sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f'{key}: {value}\n')
        f.write('\n')

def length(word_count:dict):
    # sort word_count by key
    word_count = dict(sorted(word_count.items()))
    # box plot X axis is task_a_label, Y axis is length of data
    plt.figure(figsize=(10, 5))
    plt.boxplot(word_count.values())
    plt.xticks(list(word_count.keys()))
    plt.title('Task A Label vs Length of Data')
    plt.xlabel('Task A Label')
    plt.ylabel('Length of Data')
    plt.savefig('Task_A_vs_Length_of_Data.png')
    plt.show()


