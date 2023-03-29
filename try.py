my_list = [[1, 2, 3], [4, 5], [6, 7, 8], [9, 10], [11, 12, 13], [14, 15, 16]]

# function to group similar sublists together
def group_similar_sublists(lst):
    result = []
    for sublst in lst:
        found_similar = False
        for idx, existing_sublst in enumerate(result):
            if set(sublst) == set(existing_sublst):
                result[idx].extend(sublst)
                found_similar = True
                break
        if not found_similar:
            result.append(sublst)
    return result

# split sublists and group together similar sublists
split_sublists = [sublst for lst in my_list for sublst in lst]
grouped_sublists = group_similar_sublists(split_sublists)

# print results
print("Original list: ", my_list)
print("Split sublists: ", split_sublists)
print("Grouped sublists: ", grouped_sublists)
