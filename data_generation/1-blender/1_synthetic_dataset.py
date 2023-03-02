"""
Create synthetic dataset
"""

import sys
import os
import numpy as np
import json
import argparse

import bpy
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view
import bmesh

from get_correspondence import get_correspondence


# Blender to CV
T_b2cv = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])


def set_intrinsics(K, width, height, cam):
    # Set intrinsics
    sensor_width_in_mm = K[1, 1] * K[0, 2] / (K[0, 0] * K[1, 2])
    # sensor_height_in_mm = 1  # doesn't matter
    # resolution_x_in_px = K[0, 2] * 2  # principal point assumed at the center
    # resolution_y_in_px = K[1, 2] * 2  # principal point assumed at the center
    resolution_x_in_px = width  # 1242
    resolution_y_in_px = height  # 375

    s_u = resolution_x_in_px / sensor_width_in_mm
    # TODO include pixel aspect ratio
    # s_v = resolution_y_in_px / sensor_height_in_mm
    f_in_mm = K[0, 0] / s_u

    # recover original resolution
    pixel_scale = 1.0
    bpy.context.scene.render.resolution_x = int(resolution_x_in_px / pixel_scale)
    bpy.context.scene.render.resolution_y = int(resolution_y_in_px / pixel_scale)
    bpy.context.scene.render.resolution_percentage = int(pixel_scale * 100)
    # Lens
    # cam.type = 'PERSP'
    cam.data.lens = f_in_mm
    cam.data.lens_unit = 'MILLIMETERS'
    cam.data.sensor_width = sensor_width_in_mm

    # move principal point of the blender camera
    # r = float(bpy.context.scene.render.resolution_x) / float(bpy.context.scene.render.resolution_y)
    maxDim = max(bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y)
    cam.data.shift_x = (float(bpy.context.scene.render.resolution_x) / 2.0 - K[0, 2]) / maxDim
    cam.data.shift_y = -(float(bpy.context.scene.render.resolution_y) / 2.0 - K[1, 2]) / maxDim


def transform(obj, position, rotation):
    obj.location = Vector(tuple(position))
    # obj.rotation_euler = (Matrix.Rotation(rotation[0] / 180.0 * np.pi, 3, 'X') @\
    #                       Matrix.Rotation(rotation[1] / 180.0 * np.pi, 3, 'Y') @\
    #                       Matrix.Rotation(rotation[2] / 180.0 * np.pi, 3, 'Z')).to_euler()
    obj.rotation_euler = (np.array(rotation) / 180.0 * np.pi).tolist()
    obj.keyframe_insert(data_path='location')
    obj.keyframe_insert(data_path='rotation_euler')


def setup_scene():
    # Clean scene
    bpy.data.objects.remove(bpy.data.objects["Cube"])

    # Setup background color
    bpy.data.worlds["World"].use_nodes = True
    tree = bpy.data.worlds["World"].node_tree
    tree.nodes.new(type='ShaderNodeRGB')
    tree.nodes["RGB"].outputs[0].default_value = (0.01, 0.01, 0.01, 1)
    tree.links.new(tree.nodes["RGB"].outputs["Color"], tree.nodes["Background"].inputs["Color"])

    tree.nodes.new(type="ShaderNodeValue")
    tree.nodes["Value"].outputs[0].default_value = 1
    tree.links.new(tree.nodes["Value"].outputs["Value"], tree.nodes["Background"].inputs["Strength"])


def load_model(model_path, scale=1.0):

    obj = None
    if model_path.lower().endswith(".obj"):
        # ShapeNet
        bpy.ops.import_scene.obj(filepath=os.path.abspath(model_path))
        obs = bpy.context.selected_editable_objects[:]  # editable = not linked from library
        for ob in obs:
            obj = ob
        # Index for instance mask
        obj.pass_index = 1
    elif model_path.lower().endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=os.path.abspath(model_path))
        obs = bpy.context.selected_editable_objects[:]  # editable = not linked from library
        max_vol = 0.0
        for ob in obs:
            vol = np.prod(ob.dimensions)
            if vol > max_vol:
                max_vol = vol
                obj = ob
        # Index for instance mask
        obj.pass_index = 1
    elif model_path.lower().endswith(".ply"):
        bpy.ops.import_mesh.ply(filepath=os.path.abspath(model_path))
        obs = bpy.context.selected_editable_objects[:]  # editable = not linked from library
        max_vol = 0.0
        for ob in obs:
            vol = np.prod(ob.dimensions)
            if vol > max_vol:
                max_vol = vol
                obj = ob
        # Index for instance mask
        obj.pass_index = 1
    elif model_path.lower().endswith(".blend"):
        # Blender
        # TODO: Add rigid body simulation to make wheels move https://www.youtube.com/watch?v=nKc7b3Kuums
        # append object from .blend file
        with bpy.data.libraries.load(os.path.abspath(model_path)) as (data_from, data_to):  # , link=True, relative=True
            # data_to.objects = data_from.objects
            # data_to.collections = data_from.collections
            data_to.collections.append("Collection")

        # link object to current scene
        # for obj in data_to.objects:
        #     if obj is not None and not obj.name in bpy.context.scene.objects:
        #         bpy.context.scene.collection.objects.link(obj)
        #         obj.pass_index = 1

        for col in data_to.collections:
            if col is not None and not col.name in bpy.context.scene.collection.children:
                bpy.context.scene.collection.children.link(col)
                # if not col.name == "Collection 1":
                #     col.hide_render = True
                #     col.hide_viewport = True

                # Object Index for segmentation
                for obj in col.objects:
                    if obj is not None:
                        obj.pass_index = 1

        obj = bpy.data.objects["CanModel"]

    if obj:
        obj.scale[0] = scale
        obj.scale[1] = scale
        obj.scale[2] = scale
    return obj


def setup_renderer(output_path):
    if engine == "cycles":
        bpy.context.scene.render.engine = 'CYCLES'
        #AutoNode()
        bpy.context.scene.cycles.samples = 50
        bpy.context.scene.cycles.caustics_reflective = False
        bpy.context.scene.cycles.caustics_refractive = False
        bpy.context.scene.cycles.min_bounces = 1
        bpy.context.scene.cycles.max_bounces = 10
        bpy.context.scene.cycles.feature_set = 'EXPERIMENTAL'
        bpy.context.scene.cycles.device = 'GPU'
        #bpy.context.scene.view_layers['ViewLayer'].cycles.use_denoising = True

    # Render output
    # switch on nodes and get reference
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # Clear tree
    tree.nodes.clear()
    tree.links.clear()

    # Setup passes
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True

    # Create Render Layers node
    render_node = tree.nodes.new(type='CompositorNodeRLayers')
    render_node.name = "Render Layers"
    render_node.location = 100, 250

    # denoise
    denoise_node = tree.nodes.new(type='CompositorNodeDenoise')
    denoise_node.name = "Denoise"
    denoise_node.location = 400, 450
    tree.links.new(tree.nodes["Render Layers"].outputs["Image"], tree.nodes["Denoise"].inputs["Image"])

    # create rgb output node
    output_node = tree.nodes.new(type='CompositorNodeOutputFile')
    output_node.name = "Output File"
    output_node.base_path = output_path
    output_node.location = 700, 250

    # rgb
    output_node.file_slots["Image"].use_node_format = False
    output_node.file_slots["Image"].format.file_format = 'PNG'
    output_node.file_slots["Image"].path = "RGB/frame_"
    tree.links.new(tree.nodes["Denoise"].outputs["Image"], tree.nodes["Output File"].inputs["Image"])

    # depth
    output_node.file_slots.new("Depth")
    output_node.file_slots["Depth"].use_node_format = False
    output_node.file_slots["Depth"].format.file_format = 'OPEN_EXR'
    output_node.file_slots["Depth"].format.use_zbuffer = True
    output_node.file_slots["Depth"].path = "Depth/frame_"
    tree.links.new(tree.nodes["Render Layers"].outputs["Depth"], tree.nodes["Output File"].inputs["Depth"])

    # Normal
    output_node.file_slots.new("Normal")
    output_node.file_slots["Normal"].use_node_format = False
    output_node.file_slots["Normal"].format.file_format = 'OPEN_EXR'
    # output_node.file_slots["Normal"].format.file_format = 'PNG'
    output_node.file_slots["Normal"].format.use_zbuffer = True
    output_node.file_slots["Normal"].path = "Normal/frame_"
    tree.links.new(tree.nodes["Render Layers"].outputs["Normal"], tree.nodes["Output File"].inputs["Normal"])

    # Instance
    output_node.file_slots.new("Instance")
    output_node.file_slots["Instance"].use_node_format = False
    output_node.file_slots["Instance"].format.file_format = 'PNG'
    output_node.file_slots["Instance"].path = "Instance/frame_"
    tree.links.new(tree.nodes["Render Layers"].outputs["IndexOB"], tree.nodes["Output File"].inputs["Instance"])

    # Viewer
    output_node = tree.nodes.new(type='CompositorNodeViewer')
    output_node.name = "Viewer"
    output_node.location = 700, 50
    #tree.links.new(tree.nodes["Denoise"].outputs["Image"], tree.nodes["Viewer"].inputs["Image"])
    tree.links.new(tree.nodes["Render Layers"].outputs["Depth"], tree.nodes["Viewer"].inputs["Image"])


def setup_camera(K, width, height):
    cam = bpy.data.objects["Camera"]
    set_intrinsics(K, width, height, cam)
    bpy.context.scene.camera = cam
    # cam.location = Vector((0.0, 0.0, 0.0))
    # cam.rotation_euler = (np.array([90.0, 0.0, 180.0]) / 180.0 * np.pi).tolist()
    return cam


def setup_light():
    # Lights
    light = bpy.data.objects["Light"]
    light.data.type = 'SUN'
    light.data.energy = 2
    transform(light, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    light_data = bpy.data.lights.new(name="Light.001", type='SUN')
    light_object = bpy.data.objects.new(name="Light.001", object_data=light_data)
    bpy.context.scene.collection.objects.link(light_object)
    light_object.data.energy = 2
    transform(light_object, [0.0, 0.0, 0.0], [180.0, 0.0, 0.0])

    # Setup background color/light
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0.397221, 0.397221, 0.397221, 1)
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 1
    return light


def render(output_path):
    # redirect output to log file
    logfile = os.path.join(output_path, 'render.log')
    open(logfile, 'a').close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)

    # render
    bpy.ops.render.render()

    # disable output redirection
    os.close(1)
    os.dup(old)
    os.close(old)


def object_pose(obj):
    matrix_world = np.array(obj.matrix_world)
    pose = matrix_world @ np.linalg.inv(T_b2cv)
    return pose


def get_intrinsics(cam):
    # Get intrinsics
    cam.data.lens_unit = 'MILLIMETERS'
    width = int(float(bpy.context.scene.render.resolution_x) * bpy.context.scene.render.resolution_percentage / 100.0)
    height = int(float(bpy.context.scene.render.resolution_y) * bpy.context.scene.render.resolution_percentage / 100.0)
    s_u = float(width) / cam.data.sensor_width
    f_in_mm = cam.data.lens
    f = f_in_mm * s_u

    maxDim = max(bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y)
    cx = float(bpy.context.scene.render.resolution_x) / 2.0 - cam.data.shift_x * maxDim
    cy = float(bpy.context.scene.render.resolution_y) / 2.0 + cam.data.shift_y * maxDim

    K = np.eye(3, 3)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = cx
    K[1, 2] = cy

    return K, width, height


def get_velocity(obj):
    # TODO: Compute 6D velocity and acceleration vectors
    # TODO: Replace path animation with physics simulation
    velocity = 0.0

    # Not correct
    if obj.name + ".Trajectory" in bpy.data.objects:
        length = bpy.data.objects[obj.name + ".Trajectory"].data.splines.active.calc_length()
        duration = bpy.data.objects[obj.name + ".Trajectory"].data.path_duration
        eval_time = bpy.data.objects[obj.name + ".Trajectory"].data.eval_time
        distance = (eval_time / duration) * length
        velocity = distance / (eval_time + 1.0e-6)

    # a = bpy.data.objects["Camera"].location
    # bpy.data.objects[obj.name + ".Trajectory"].data.eval_time -= 1.0
    # bpy.context.evaluated_depsgraph_get().update()
    # obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    # b = obj.location
    # bpy.data.objects[obj.name + ".Trajectory"].data.eval_time += 1.0
    return np.array([velocity])


def camera_trajectory(cam, num_points, sphere_radius=6.0):
    # Example trajectory
    # velocity = 3.0 * bpy.context.scene.render.fps_base
    # camera_trajectory = np.zeros((num_steps, 3))
    # camera_trajectory[0] = [0.0, 0.0, 1.5]
    # for i in range(1, camera_trajectory.shape[0]):
    #     camera_trajectory[i] = camera_trajectory[i - 1] + np.array([0.5 * np.sin(0.01 * i * velocity), -velocity, 0.0])
    # create_trajectory(cam, 0, num_steps - 1, camera_trajectory)

    # Spherical sampling based on fibonacci sphere
    # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    # TODO: Automatically adjust distance based on object dimensions
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(num_points):
        z = (1.0 - (i / float(num_points - 1)) * 2.0) * sphere_radius  # z goes from R to -R
        # https://en.wikipedia.org/wiki/Circle_of_a_sphere
        radius = np.sqrt(sphere_radius*sphere_radius - z*z) # radius at z
        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        y = np.sin(theta) * radius

        # New animation frame
        bpy.context.scene.frame_set(i)

        # Move on circle around object
        position = [x, y, z]

        norm = np.linalg.norm(position)

        # Look at object center
        # https://en.wikipedia.org/wiki/Spherical_coordinate_system
        longitude = np.arctan2(y, x)
        latitude = np.arccos(z / sphere_radius)
        rotation = [-latitude / np.pi * 180.0, 0.0, longitude / np.pi * 180.0 - 90.0]

        # Debug
        # cam = bpy.data.cameras.new("Camera")
        # cam = bpy.data.objects.new("Camera", cam)
        # bpy.context.scene.collection.objects.link(cam)

        transform(cam, position, rotation)

def get_coordinates(obj, cam):
    scene = bpy.context.scene
    bpy.context.view_layer.update()
    
    # get_correspondence(obj,0)
    
    # needed to rescale 2d coordinates
    render = scene.render
    res_x = render.resolution_x
    res_y = render.resolution_y

    # use generator expressions () or list comprehensions []
    # verts = (vert.co for vert in obj.data.vertices)
    # coords_2d = [world_to_camera_view(scene, cam, coord) for coord in verts]
    
    # Threshold to test if ray cast corresponds to the original vertex
    limit = 0.0001
    coords_2d = []
    uv_coords = []
    
    
    mWorld = obj.matrix_world
    
    vertices = [mWorld @ v.co for v in obj.data.vertices]
    
    # verts = [[v, i] for f in obj.data.polygons for v,i in zip(f.vertices, f.loop_indices)]
    # vertices = [(mWorld @  v[0].co, v[1]) for v in verts]
    
    for i, vert in enumerate(vertices):
        cord2d = world_to_camera_view(scene, cam, vert)
        if 0.0 <= cord2d.x <= 1.0 and 0.0 <= cord2d.y <= 1.0: 
            # Try a ray cast, in order to test the vertex visibility from the camera
            ray = obj.ray_cast( cam.location, (vert - cam.location).normalized() )
            # If the ray hits something and if this hit is close to the vertex, we assume this is the vertex
            if ray[0] and (vert - ray[1]).length < limit:
                uv_coords.append(obj.data.uv_layers.active.data[i].uv)
                coords_2d.append(cord2d)
    

    # 2d data printout:
    rnd = lambda i: round(i)
    
    # find min max distance, between eye and coordinate.
    rnd3 = lambda i: round(i, 3)

    limit_finder = lambda f: f(coords_2d, key=lambda i: i[2])[2]
    limits = limit_finder(min), limit_finder(max)
    limits = [rnd3(d) for d in limits]

    out = []
    
    bpy.ops
    
    
    uv_res = 512
    uv_out = []
    for u,v in uv_coords:
        vo = rnd(v*uv_res)
        vo = np.abs(vo - uv_res)
        uv_out.append((u*uv_res, vo))
    
    
    for x, y, z in coords_2d:
        yo = rnd(res_y*y)
        yo = np.abs(yo - res_y)
        out.append(((res_x*x), yo, rnd3(z)))
    
    
    # for x, y, distance_to_lens in coords_2d:
    #     out.append((rnd(res_x*x), rnd(res_y*y)))
        
    return out, uv_out



if __name__ == "__main__":
    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Model path')
    parser.add_argument('output_path', help='Output path')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Config parameters
    num_steps = 25
    fps_base = 0.1
    engine = "cycles"
    width = 1280
    height = 720
    K = np.array([[720.0, 0., width / 2.0],
                  [0., 720.0, height / 2.0],
                  [0., 0., 1.]])

    # Reset
    bpy.ops.wm.read_homefile(use_empty=False)

    # Setup sequence
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = num_steps
    bpy.context.scene.render.fps = 1
    bpy.context.scene.render.fps_base = fps_base

    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.cycles.preview_samples = 1
    bpy.context.scene.cycles.max_bounces = 0
    bpy.context.scene.cycles.diffuse_bounces = 0
    bpy.context.scene.cycles.glossy_bounces = 0
    bpy.context.scene.cycles.transparent_max_bounces = 0
    bpy.context.scene.cycles.transmission_bounces = 0
    bpy.context.scene.cycles.volume_bounces = 0

    # Setup Scene
    setup_scene()

    # Load model
    obj = load_model(args.model_path, scale=1)

    # Render engine
    setup_renderer(args.output_path)

    # Material override for depth rendering
    mat = bpy.data.materials.new("Diffuse")
    mat.diffuse_color = [1.0, 1.0, 1.0, 1.0]
    # obj.data.materials.append(mat)
    # bpy.context.scene.view_layers["ViewLayer"].material_override = mat
    bpy.context.scene.view_layers["ViewLayer"].samples = 1

    # Setup camera
    cam = setup_camera(K, width, height)

    # Debug
    #K, width, height = get_intrinsics(cam)

    # Setup light
    light = setup_light()

    # Camera trajectory
    camera_trajectory(cam, num_steps)

    # Save setup
    bpy.ops.export_mesh.ply(filepath=os.path.join(args.output_path, "model.ply"))
    bpy.ops.wm.save_mainfile(filepath=os.path.abspath(os.path.join(args.output_path, "model.blend")))

    # Create output directories
    os.makedirs(os.path.join(args.output_path, "RGB"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "Depth"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "Normal"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "Instance"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "Object_Pose"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "Camera_Pose"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "Image_Coordinates"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "UV_Coordinates"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "Camera_Intrinsics"), exist_ok=True)

    # Simulate
    for step in range(num_steps):
        # Next step
        bpy.context.scene.frame_set(step)

        # Save object pose
        np.savetxt(os.path.join(args.output_path, "Object_Pose", "object_pose_{:04d}.txt".format(step)),
                    object_pose(obj))

        # Save camera pose
        np.savetxt(os.path.join(args.output_path, "Camera_Pose", "camera_pose_{:04d}.txt".format(step)),
                    object_pose(cam))

        # Save camera intrinsics
        np.savetxt(os.path.join(args.output_path, "Camera_Intrinsics", "camera_intrinsics_{:04d}.txt".format(step)),
                    np.array(K))

        # Get object image coordinates
        coords ,uv = get_coordinates(obj, cam)
        np.savetxt(os.path.join(args.output_path, "Image_Coordinates", "image_coordinates_{:04d}.txt".format(step)),
                   np.asarray(coords, dtype=int), fmt="%i", delimiter=',')
        np.savetxt(os.path.join(args.output_path, "UV_Coordinates", "uv_coordinates_{:04d}.txt".format(step)),
                   np.asarray(uv, dtype=int), fmt="%i", delimiter=',')
        # Render
        render(args.output_path)

    sys.exit(0)
