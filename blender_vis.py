"""  # noqa 
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/       # noqa 
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/       # noqa 
Author : Benjamin Blundell - benjamin.blundell@kcl.ac.uk

blender_vis.py - Visualisations that make use of Blender.

Running this script inside blender (version >=2.8) will
launch a file dialog box. Select the animation.json from
a neural network run you wish to visualise.

"""

# https://docs.blender.org/api/current/index.html
# https://github.com/njanakiev/blender-scripting


import bpy

SCALE_FACTOR = 10.0


class ScanFileOperator(bpy.types.Operator):
    bl_idname = "error.scan_file"
    bl_label = "Load JSON Animation File"
    filepath = bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        parseJSON(self.filepath)
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


def parseJSON(filepath):
    import json

    with open(filepath, "rU") as f:
        raw_json = f.read()

        points = []
        points_group = bpy.data.collections.new("NN Points")
        bpy.context.scene.collection.children.link(points_group)
        scene = bpy.context.scene

        try:
            data = json.loads(raw_json)
            frame = data["frames"][0]
            num_frames = len(data["frames"])

            for vertex in frame["vertices"]:
                x = vertex["x"] * SCALE_FACTOR
                y = vertex["y"] * SCALE_FACTOR
                z = vertex["z"] * SCALE_FACTOR

                point = bpy.ops.mesh.primitive_ico_sphere_add(location=(x, y, z))
                C = bpy.context
                points.append(C.object)
                points_group.objects.link(C.object)

                modifier = C.object.modifiers.new("Subsurf", "SUBSURF")
                modifier.levels = 2
                modifier.render_levels = 2

                mesh = C.object.data
                for p in mesh.polygons:
                    p.use_smooth = True

            # Now setup some animation
            for i in range(num_frames):
                scene.frame_set(i)

                for j, point in enumerate(points):
                    frame = data["frames"][i]
                    vertex = frame["vertices"][j]
                    x = vertex["x"] * SCALE_FACTOR
                    y = vertex["y"] * SCALE_FACTOR
                    z = vertex["z"] * SCALE_FACTOR
                    point.location = (x, y, z)
                    point.keyframe_insert(data_path="location", index=-1)

        except Exception:
            bpy.ops.error.message(
                "INVOKE_DEFAULT", type="Error", message="Error loading" + filepath
            )

    return points


if __name__ == "__main__":
    bpy.utils.register_class(ScanFileOperator)
    bpy.ops.error.scan_file("INVOKE_DEFAULT")
