import os
import json
from extract.extract import get_extract_trajectory
from extract.util import convert_from_vector

dataset_name = "extended_subset/"
save_dir = "results/" + dataset_name
data_dir = "data/" + dataset_name

os.makedirs(save_dir, exist_ok=True)

for filename in os.listdir(data_dir):
    print("Getting results for file ", filename)
    file_path = os.path.join(data_dir, filename)
    results = get_extract_trajectory(file_path)

    with open(os.path.join(save_dir, filename), "w") as outfile:
        # save in same format as of OVITA if anytime feedback is implemented
        json.dump(
            {
                "zero_shot_trajectory": {
                    "trajectory": convert_from_vector(results[0]),
                    "modified trajectory": convert_from_vector(results[1]),
                    "instruction": results[2],
                    "objects": results[3],
                    "high_level_plan": "",
                    "code": "",
                    "interpretation": "",
                    "code_executability": True,
                },
                "final_trajectory": {
                    "trajectory": [],
                    "instruction": "",
                    "objects": [],
                    "modified trajectory": [],
                    "high_level_plan": "",
                    "code": "",
                    "interpretation": "",
                    "code_executability": True,
                },
            },
            outfile,
            indent=4,
        )

print(f"Obtained and stored results for all the trajectories in {dataset_name}")
