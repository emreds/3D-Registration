import bpy
from mathutils import Matrix, Vector
from mathutils.geometry import barycentric_transform, intersect_point_tri_2d
from bpy_extras.object_utils import world_to_camera_view
import bmesh

def get_correspondence(obj, i ):
    # my_props = bpy.context.scene.test_pg
    uv_coords = []
    for face in obj.data.polygons:
        for vert_idx, loop_idx in zip(face.vertices, face.loop_indices):
            uv_coords.append(obj.data.uv_layers.active.data[loop_idx].uv)
            # if vert_idx == my_props.sel_vert_idx:
            #     my_uv = uv_coords