import os
import json
from extract.extract import get_extract_trajectory
from extract.util import convert_from_vector, compare_trajectory

dataset_name = "extended_subset/"
save_dir = "results/" + dataset_name
data_dir = "data/" + dataset_name

file_path="data/latte_subset/latte_24.json"
results = get_extract_trajectory(file_path)

compare_trajectory(results[0], results[1], results[2], results[3])

 
