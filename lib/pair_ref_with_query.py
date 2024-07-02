from pathlib import Path
import os
from utils.read_model import parse_pose_list
def get_pairs_imagepath(pairs, render_path: Path, image_path: Path, engine = 'blender'):
    """
    Generate a dictionary of image paths for query and render images based on pairs.

    Args:
        pairs (dict): Dictionary containing query names and corresponding list of render names.
        render_path (Path): Path to the directory where render images are stored.
        image_path (Path): Path to the directory where query images are stored.

    Returns:
        dict: Dictionary with query image paths as keys and lists of tuples containing
              render image paths and corresponding EXR file paths as values.
              (pairs[query path] :[render_RGB path, render_depth path])
    """
    render_dir = {}
    for query_name, imgr_name_list in pairs.items():
        renders = []
        if 'query' not in query_name:
            query_name = 'query/' + query_name
        imgq_pth = image_path / query_name  # Path to the query image
        
        for imgr_name in imgr_name_list:
            imgr = imgr_name.split('/')[-1].split('.')[0]  # Extract image name from the render name
            
            if engine == 'blender':
                imgr_pth = str(render_path / (imgr + '.jpg'))  # Path to the render image
                exrr_pth = str(render_path / (imgr + '0001.exr'))# Path to the corresponding EXR file
            elif engine == 'osg':
                imgr_pth = str(render_path / (imgr + '.png'))  # Path to the render image
                exrr_pth = str(render_path / (imgr + '.tiff'))# Path to the corresponding EXR file
                
            # Create a tuple of the render image path and the EXR file path
            render = [imgr_pth, exrr_pth]
            
            renders.append(render)  # Add the tuple to the list of renders
        
        # Use the string representation of the query image path as the key
        render_dir[str(imgq_pth)] = renders
    
    return render_dir
def get_image_name(path):
    """
    Parses a pose list file and get render list.

    Args:
        path (str): Path to the pose list file.

    Returns:
        name list.
    """
    names = []
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            tokens = data.split()
            name = tokens[0]
            names.append(name)

    return names
def get_render_candidates(renders, queries):
    """
    Creates a dictionary of pairs where each query image is paired with a corresponding render image.

    Args:
        renders (.txt): A file of render image identifiers.
        queries (.txt): A file of query image identifiers.

    Returns:
        dict: A dictionary with query image identifiers as keys and lists containing
              a single render image identifier as values.
    """
    pairs = {}
    # Parse the file to get names
    render_names = get_image_name(renders)
    query_names = get_image_name(queries)
    
    
    for query_name in query_names:
        render_candidate = []
        query = (query_name.split('/')[-1]).split('.')[0]
        for render_name in render_names:
            if query in render_name:
                render_candidate.append(render_name)
        pairs[query_name] = render_candidate
    return pairs
 
def main(image_dir: Path, render_dir: Path, query_camera: Path, render_camera: Path, render_extrinsics: Path, engine = 'blender'):
    """
    Main function to process camera data and generate image pairs for rendering.

    Args:
        image_dir (Path): Path to the directory containing query images.
        render_dir (Path): Path to the directory with rendered images.
        query_camera (Path): Path to the file containing query camera data.
        render_camera (Path): Path to the file containing render camera data.
        render_extrinsics (Path): Path to the file containing render extrinsics data.
        iter (int): Iteration number or parameter for processing.

    Returns:
        dict: Dictionary to store and return processed data.
    """
    data = dict()
    # Ensure that the specified camera and extrinsics files exist
    assert render_camera.exists(), "Render camera file does not exist."
    assert render_extrinsics.exists(), "Render extrinsics file does not exist."
    assert query_camera.exists(), "Query camera file does not exist."

    # Create pairs of render and query images based on their names
    pairs = get_render_candidates(render_extrinsics, query_camera)

    # Get the full image paths for all pairs of images
    all_pairs_path = get_pairs_imagepath(pairs, render_dir, image_dir, engine=engine)

    # Get the dictionary of poses with names as keys
    render_pose_raw = parse_pose_list(render_extrinsics)
    render_pose = dict()
    for key, item in render_pose_raw.items():
        render_pose[key.split('.')[0]] = item

    data["render_pose"] = render_pose
    data["pairs"] = all_pairs_path

    return data