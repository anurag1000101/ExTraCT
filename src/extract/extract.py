import argparse
import json
import numpy as np
from extract.util import (
    detect_objects,
    get_trajectory,
    convert_from_vector,
    convert_to_vector,
    compare_trajectory
)
from extract.extract_utils import Extract
from extract.feature_templates import FEATURE_TEMPLATES
from extract.deformation_functions import DeformationFunctions
import torch

corrections = Extract()
deformation_operator=DeformationFunctions()

def get_extract_trajectory(file_path, instruction=None):

    with open(file_path, "r") as file:
        data = json.loads(file.read())

    trajectory = convert_to_vector(get_trajectory(data))
    objects = detect_objects(data)
    if instruction is None: 
        instruction = data["instruction"]

    all_features = corrections.generate_features(objects, FEATURE_TEMPLATES)

    instruction_embedding = corrections.generate_bert_embeddings([instruction])
    # import ipdb; ipdb.set_trace()
    features_embeddings = corrections.generate_bert_embeddings(all_features["TDT"])

    # expanding to have the same dimensions
    instruction_embedding = instruction_embedding.expand_as(features_embeddings)
    cosine_similarity = corrections.compute_cosine_similarity(
        features_embeddings, instruction_embedding
    )

    best_index = torch.argmax(cosine_similarity)

    FT = all_features["FT"][best_index]
    obj_selected = all_features["objects"][best_index]
    obj_position=None
    if obj_selected!='no_object':
        for obj in objects:
            if obj['name']==obj_selected:
                obj_position=[obj['x'], obj['y'], obj['z']]
                break

    # select the deformation function
    modified_trajectory=np.array(deformation_operator.apply_deformation(FT,trajectory, obj_position)).tolist()
    # import ipdb; ipdb.set_trace()
    return trajectory.tolist(), modified_trajectory, instruction, objects


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Trajectory corrections using ExTraCT")
    parser.add_argument(
        "--trajectory_path", 
        type=str, 
        required=True, 
        help="Path to the trajectory JSON file."
    )
    parser.add_argument(
        "--instruction", 
        type=str, 
        required=False, 
        help="Pass your custom instruction here",
        default=None
    )
    args = parser.parse_args()
    results=get_extract_trajectory(args.trajectory_path,args.instruction)
    compare_trajectory(results[0], results[1],results[2],results[3])


