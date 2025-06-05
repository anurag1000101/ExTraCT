import os 
import json
from extract.extract import get_extract_trajectory
save_dir="results/"
data_dir="data/"

for filename in os.listdir(data_dir):
    print("Getting results for file ",filename)
    file_path = os.path.join(data_dir, filename)
    results=get_extract_trajectory(file_path)

    with open(os.path.join(save_dir,filename), "w") as outfile:
        json.dump({
            'zero_shot_trajectory': results[0].tolist(),
            'final_trajectory': results[1],
        }, outfile, indent=4)

print("Obtained and stored results for all the trajectories")


