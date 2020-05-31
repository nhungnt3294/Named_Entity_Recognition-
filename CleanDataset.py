import os


def remove_xml_tag(data_path, new_data_path):
    if not os.path.exists(new_data_path):
        os.mkdir(new_data_path)
    for file_name in os.listdir(data_path):
        file_path = data_path + '/' + file_name
        new_file_path = new_data_path + '/' + file_name
        with open(file_path, 'r', encoding='utf-8') as old_file:
            with open(new_file_path, 'w', encoding='utf-8') as new_file:
                for line in old_file:
                    line.strip()
                    if ('<title>' in line) or line.startswith('<e') or line.startswith('-D') or line.startswith('<s>'):
                        pass
                    elif line.startswith('</'):
                        new_file.write(line.replace(line, '\n'))
                    else:
                        new_file.write(line)
        old_file.close()
        new_file.close()


old_path = 'G:/Project/python/NER/data/NER2016-training_data-1'
new_path = 'G:/Project/python/NER/data/NER2016-training_data'
remove_xml_tag(old_path, new_path)
