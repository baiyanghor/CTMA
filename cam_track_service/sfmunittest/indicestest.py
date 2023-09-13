
def match_indices(image_index, max_index):
    a = image_index - 1
    if a < 0:
        a = 0
    b = image_index
    if b == max_index:
        b = image_index - 1
    return a, b


for i in range(8):
    pre_idx, next_idx = match_indices(i, 7)
    print(f"\nimage idx: {i}\npre: {pre_idx} next: {next_idx}")
