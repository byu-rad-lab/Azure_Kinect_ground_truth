import time


def copy_txt_contents(source_file, target_file):
    with open(source_file, 'r') as src:
        contents = src.read()
    with open(target_file, 'w') as tgt: 
        tgt.write(contents)


if __name__ == '__main__':
    while True:
        for i in range(10):
            copy_txt_contents(f"filtered_data{i}.txt", 'filtered_data_any.txt')
            time.sleep(.02)
            print(f'Switched to filtered_data{i}.txt')