import numpy as np
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Run conversion to nifti'
    )
    parser.add_argument('-train_cases_txt', type=str)
    parser.add_argument('-test_cases_txt', type=str)
    parser.add_argument('-target', type=str)
    args = parser.parse_args()


    def transform_inputs(input_lines):
        transformed_cases = []

        for line in input_lines:
            subject, label = line.strip().split('_seg-vert_')
            subject = subject.replace('sub-', '')
            label_number = label.split('_')[0]
            transformed_case = f'/sub-{subject}/{line}_aligned-false_label-{label_number}_voxels-128_zoom-1-1-1_msk'
            transformed_cases.append(transformed_case)

        return np.array(transformed_cases)

    # Read input lines from the train text file
    train_file_path = args.train_cases_txt
    with open(train_file_path, 'r') as train_file:
        train_input_lines = train_file.readlines()

    # Read input lines from the test text file
    test_file_path = args.test_cases_txt
    with open(test_file_path, 'r') as test_file:
        test_input_lines = test_file.readlines()

    # Transform the input lines into train and test cases
    train_cases_array = transform_inputs(train_input_lines)
    test_cases_array = transform_inputs(test_input_lines)

    # Create a dictionary to store train and val cases grouped by "sub-verse" number
    train_cases_dict = {}
    val_cases_dict = {}

    # Group cases by "sub-verse" number in the train cases dictionary
    for case in train_cases_array:
        sub_verse_number = case.split('/')[1][4:]
        if sub_verse_number in train_cases_dict:
            train_cases_dict[sub_verse_number].append(case)
        else:
            train_cases_dict[sub_verse_number] = [case]

    # Split train cases into train and val arrays while preserving the "sub-verse" groups
    train_cases_split = []
    val_cases_split = []
    split_index = int(0.8 * len(train_cases_dict.keys()))

    for sub_verse in list(train_cases_dict.keys())[:split_index]:
        train_cases_split.extend(train_cases_dict[sub_verse])
    for sub_verse in list(train_cases_dict.keys())[split_index:]:
        val_cases_split.extend(train_cases_dict[sub_verse])

    # Save the train, val, and test cases as an NPZ file
    np.savez(args.target, train=train_cases_split, val=val_cases_split, test=test_cases_array)
    print(len(train_cases_split))
    print(len(test_cases_array))
    print(len(val_cases_split))