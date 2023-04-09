def preprocess_text_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            words = line.split()
            if len(words) > 2:
                outfile.write(line + '\n')

if __name__ == '__main__':
    path= 'yelp/'
    input_file = path+'train.1'      # Replace with your input file name
    output_file = path+'process_train.1'     # Replace with your desired output file name
    preprocess_text_file(input_file, output_file)
