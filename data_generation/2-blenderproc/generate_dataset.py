import blenderproc as bproc
import argparse
import os
import numpy as np
import bpy

from os import path
from blenderproc.python.material import MaterialLoaderUtility

import debugpy
debugpy.listen(5678)
""" Uncomment this line to use the debugger as described in the BlenderProc documentation """
# debugpy.wait_for_client()

TLESS_OBJ_IDs = [21]
NUM_FRAMES = 200
USE_PATTERN = True

SAVE_OBJ = True

PATH_ROOT = "Path/to/your/files"

patterns = {
    0: None,
    1: (path.join(PATH_ROOT, "pattern.png"), "PATTERN"),
    2: (path.join(PATH_ROOT, "penrose.png"), "PENROSE"),
    3: (path.join(PATH_ROOT, "Geometric.png"), "GEOMETRIC"),
    4: (path.join(PATH_ROOT, "Letters.jpg"), "LETTERS"),
    5: (path.join(PATH_ROOT, "Lines.jpg"), "LINES")
}

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', nargs='?', help="Path to the bop datasets parent directory")
parser.add_argument('bop_dataset_name', nargs='?', help="Main BOP dataset")
parser.add_argument('cc_textures_path', nargs='?', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved ")
parser.add_argument('object_id', nargs='?', help="Id of the object to use ")
parser.add_argument('pattern_id', nargs='?', help="Id of the pattern to use ")
# parser.add_argument('background_id', nargs='?', help="Id of the background to use ")
args = parser.parse_args()


TLESS_OBJ_IDs = [int(args.object_id)]

bproc.init()
# activate depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

# load a random sample of bop objects into the scene
sampled_bop_objs = bproc.loader.load_bop_objs(
                                bop_dataset_path=os.path.join(
                                    args.bop_parent_path,
                                    args.bop_dataset_name),
                                obj_ids=TLESS_OBJ_IDs,
                                model_type="cad",
                                mm2m=True,
                                sample_objects=True,
                                num_of_objs_to_sample=len(TLESS_OBJ_IDs))


# load distractor bop objects
## For the project no distraction objects were used, thus this is set to be an empty list.
distractor_bop_objs = []
# distractor_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'tless'),
#                                      model_type = 'cad',
#                                      mm2m = True,
#                                      sample_objects = True,
#                                      num_of_objs_to_sample = 3)
# distractor_bop_objs += bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'lm'),
#                                       mm2m = True,
#                                       sample_objects = True,
#                                       num_of_objs_to_sample = 3)

# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, args.bop_dataset_name))

# load cc textures
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path, use_all_materials=False)
       
# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
            bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
            bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
            bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
            bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))    
light_plane.replace_materials(light_plane_material)

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(np.random.uniform(75, 250))
light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                        elevation_min = 5, elevation_max = 89, uniform_volume = False)
light_point.set_location(location)

# Not used in productive code, for testing this was used to select certain textures that should be applied
room_textures = [
    bproc.filter.one_by_attr(cc_textures, "name", "Wood050"),
    # bproc.filter.one_by_attr(cc_textures, "name", "Fabric034"),
    # bproc.filter.one_by_attr(cc_textures, "name", "PavingStones117"),
    # bproc.filter.one_by_attr(cc_textures, "name", "Asphalt020L"),
    # bproc.filter.one_by_attr(cc_textures, "name", "Cardboard002"),
]

# Define a function that samples the initial pose of a given object above the ground
def sample_initial_pose(obj: bproc.types.MeshObject):
    obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
                                                min_height=1, max_height=4, face_sample_range=[0.4, 0.6]))
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi * 2, np.pi * 2, np.pi * 2]))


# Sample objects on the given surface
placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=sampled_bop_objs + distractor_bop_objs,
                                        surface=room_planes[0],
                                        sample_pose_func=sample_initial_pose,
                                        min_distance=0.01,
                                        max_distance=0.2)

# BVH tree used for camera obstacle checks
bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(placed_objects)

pattern_id = int(args.pattern_id)

texture = True
if pattern_id == 0:
    texture = False

print(f"Running pattern_id {pattern_id} with texturing {str(texture)}")

# set shading and physics properties and randomize PBR materials
for j, obj in enumerate(sampled_bop_objs + distractor_bop_objs):
    obj.set_shading_mode('auto')
    if texture:
        pattern = MaterialLoaderUtility.create_material_from_texture(
            *patterns[pattern_id])
        obj.replace_materials(pattern)
        obj.add_uv_mapping("smart")
    else:
        mat = obj.get_materials()[0]
        if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
            grey_col = np.random.uniform(0.3, 0.9)   
            mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])        
            mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
            mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))      
        
if SAVE_OBJ:
    p = path.join(PATH_ROOT, f"used_models_cad/obj_{TLESS_OBJ_IDs[0]:6d}")
    # create new image or just find your image in bpy.data
    image_new = bpy.data.images.new(name = 'newimage',width=1024,height=1024,alpha=True)
    image_old = bpy.data.images[0]

    # saving image_new which is created above
    mypath = 'D:/01 Projects/mytexture.png'
    image_new.filepath = mypath
    image_new.save()

    # and you can save loaded images without setting filepath attribute:
    image_old.save_render(filepath='D:/01 Projects/mytexture.png')


random_cc_texture = np.random.choice(cc_textures)
for plane in room_planes:
    plane.replace_materials(random_cc_texture)

poses = 0

while poses < NUM_FRAMES:
    # Sample location
    location = bproc.sampler.shell(center=[0, 0, 0],
                                   radius_min=0.25,
                                   radius_max=0.5,  # 1.24,
                                   elevation_min=5,
                                   elevation_max=15,  # 89,
                                   uniform_volume=False)
    # Determine point of interest in scene as the object closest to the mean of a subset of objects
    poi = bproc.object.compute_poi(np.random.choice(placed_objects, size=10))

    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

    # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
    if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.1}, bop_bvh_tree):
        # Persist camera pose
        bproc.camera.add_camera_pose(cam2world_matrix)
        poses += 1

# render the whole pipeline
data = bproc.renderer.render()

# Write data in bop format
bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                        dataset=args.bop_dataset_name,
                        depths=data["depth"],
                        colors=data["colors"],
                        color_file_format="JPEG",
                        ignore_dist_thres=10,
                        frames_per_chunk=NUM_FRAMES)
