import bpy
import os
from pathlib import Path
import shutil
import math
import argparse
import numpy as np
import cv2 as cv

def calc_grid_size(num_frames):
    sqrt_val = int(math.sqrt(num_frames))
    # eg: 64 frames = 8x8; sqrt(64) = 8
    for rows in range(sqrt_val, 0, -1):
        if num_frames % rows == 0:
            cols = num_frames // rows
            return (rows, cols)
    return (1, num_frames)

def render_timeline(self, context):

    frames_to_skip = context.scene.frames_to_skip
    height = context.scene.output_frame_y
    width = context.scene.output_frame_x

    should_render_motion_vector = context.scene.should_render_motion_vector

    temp_folder = context.scene.render.filepath + 'temp_flipbookframes'
    for i in range(context.scene.frame_start, context.scene.frame_end, frames_to_skip):
        context.scene.frame_current = i
        bpy.ops.render.render(True)
        render_result = next(image for image in bpy.data.images if image.type == 'RENDER_RESULT')
        render_result.save_render(temp_folder + f'/FRAME_{i:03}.png')
    
    prev_frame_dir = context.scene.frames_to_pack_dir
    context.scene.frames_to_pack_dir = temp_folder
    pack_flipbook(self, context)
    context.scene.frames_to_pack_dir = prev_frame_dir
    if os.path.exists(temp_folder): shutil.rmtree(temp_folder)

def pack_flipbook(self, context):
    image_folder = context.scene.frames_to_pack_dir
    
    if not os.listdir(image_folder):
        self.report({'WARNING'}, 'Select a folder containing png files')
        return

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    images = [cv.imread(os.path.join(image_folder, f), cv.IMREAD_UNCHANGED) for f in image_files]
    
    if not images:
        self.report({'WARNING'}, 'No images found in the specified directory.')
        return

    rows, cols = calc_grid_size(len(images))
    height, width, channels = images[0].shape
    # self.report({'INFO'}, f'the channels: {_channels}')
    flipbook = np.zeros((height * rows, width * cols, channels), dtype=np.uint8)

    for index, img in enumerate(images):
        row = index // cols
        col = index % cols
        y_offset = row * height
        x_offset = col * width

        # Place the image in the flipbook
        flipbook[y_offset:y_offset + height, x_offset:x_offset + width] = img

    cv.imwrite(context.scene.render.filepath + f'flipbook_{cols}x{rows}.png', flipbook)
    self.report({'INFO'}, f'Saved to {context.scene.render.filepath}')

def bake_motion_vector(self, context):
    
    flipbook_size_x = context.scene.flipbook_size_x  # Number of frames in the x direction
    flipbook_size_y = context.scene.flipbook_size_y  # Number of frames in the y direction
    flipbook = cv.imread(context.scene.flipbook_file, cv.IMREAD_UNCHANGED)
    
    if flipbook is None:
        self.report({'WARNING'}, 'Select a Flipbook texture')
        return
    if flipbook_size_y == 0 or flipbook_size_x == 0:
        self.report({'WARNING'}, 'Describe the dimensions of the flipbook')
        return

    # Get the total height and width of the flipbook
    total_height, total_width, _ = flipbook.shape
    
    # Calculate the height and width of each frame
    frame_height = total_height // flipbook_size_y
    frame_width = total_width // flipbook_size_x
    
    # Initialize the motion vector texture
    motion_vector_texture = np.zeros((total_height, total_width, 3), dtype=np.float32)  # Use float32 for motion vectors

    # Loop through the frames
    for y in range(flipbook_size_y - 1):  # Loop through the rows of frames
        for x in range(flipbook_size_x):  # Loop through the columns of frames
            # Extract the current and next frame
            frame1 = flipbook[y * frame_height:(y + 1) * frame_height, x * frame_width:(x + 1) * frame_width, :]  # Current frame
            if y < flipbook_size_y - 1:
                frame2 = flipbook[(y + 1) * frame_height:(y + 2) * frame_height, x * frame_width:(x + 1) * frame_width, :]  # Next frame
            else:
                continue  # Skip the last row since there's no next frame

            # Convert frames to grayscale
            prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
            next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

            # Calculate optical flow
            flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Store the flow vectors in the motion vector texture
            #why is this done Pixel by Pixel and not just write the flow in the corrosponding coordinates in the motion vector texture???
            for fy in range(flow.shape[0]):
                for fx in range(flow.shape[1]):
                    # Normalize the flow vector
                    norm = np.linalg.norm(flow[fy, fx])
                    if norm > 0:
                        # Normalize to [0, 1]
                        motion_vector_texture[y * frame_height + fy, x * frame_width + fx, 0] = flow[fy, fx, 0] / 10.0  # Scale down for visualization
                        motion_vector_texture[y * frame_height + fy, x * frame_width + fx, 1] = flow[fy, fx, 1] / 10.0  # Scale down for visualization
                    else:
                        motion_vector_texture[y * frame_height + fy, x * frame_width + fx, 0] = 0
                        motion_vector_texture[y * frame_height + fy, x * frame_width + fy, 1] = 0

    # Convert motion vectors to color representation
    # Normalize to [0, 255] for visualization
    motion_vector_texture = (motion_vector_texture - motion_vector_texture.min()) / (motion_vector_texture.max() - motion_vector_texture.min()) * 255
    motion_vector_texture = motion_vector_texture.astype(np.uint8)

    # Save the motion vector texture
    flipbook_path = Path(context.scene.flipbook_file)
    cv.imwrite(f'{flipbook_path.stem}_MotionVector.png', motion_vector_texture)
    self.report({'INFO'}, f'Saved {flipbook_path.stem}_MotionVector')

class VFXToolkitPackPanel(bpy.types.Panel):
    bl_label = "Load Frames To Pack Flipbook"
    bl_idname = "PT_VFXToolkit_PackFrames"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Flipbooks'

    def draw(self, context):
        layout = self.layout
        layout.label(text="↓ Folder Containing PNGs")
        layout.prop(context.scene, "frames_to_pack_dir")
        layout.prop(context.scene, "max_column")
        layout.operator("vfx_toolkit.pack_flipbook_operator")

class VFXToolkitBakePanel(bpy.types.Panel):
    bl_label = "Bake Motion Vector For A Flipbook"
    bl_idname = "PT_VFXToolkit_BakeMotionVector"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Flipbooks'

    def draw(self, context):
        layout = self.layout
        layout.label(text="↓ Flipbook To Bake Motion Vectors For")
        layout.prop(context.scene, "flipbook_file")
        layout.label(text="↓ Describe The Flipbook Dimensions")
        layout.prop(context.scene, "flipbook_size_x")
        layout.prop(context.scene, "flipbook_size_y")
        layout.operator("vfx_toolkit.bake_motion_vector_operator")

class VFXToolkitRenderPanel(bpy.types.Panel):
    bl_label = "Render Timeline To Flipbook"
    bl_idname = "PT_VFXToolkit_RenderTimeline"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Flipbooks'

    def draw(self, context):
        layout = self.layout
        layout.label(text= "Set Frame's Resolution in 𝑂𝑢𝑡𝑝𝑢𝑡 tab")
        layout.label(text= "Check 𝑓𝑖𝑙𝑚 > 𝑡𝑟𝑎𝑛𝑠𝑝𝑎𝑟𝑒𝑛𝑡 in 𝑅𝑒𝑛𝑑𝑒𝑟 tab")
        layout.prop(context.scene, "frames_to_skip")
        # layout.prop(context.scene, "output_frame_x")
        # layout.prop(context.scene, "output_frame_y")
        layout.prop(context.scene, "should_render_motion_vector")
        layout.operator("vfx_toolkit.render_timeline_operator")

class PackFlipbookOperator(bpy.types.Operator):
    bl_idname = "vfx_toolkit.pack_flipbook_operator"
    bl_label = "Pack Flipbook"
    bl_description = "Pack all the frames in the selected folder to a flipbook"

    def execute(self, context):
        pack_flipbook(self, context)
        return {'FINISHED'}

class BakeMotionVectorOperator(bpy.types.Operator):
    bl_idname = "vfx_toolkit.bake_motion_vector_operator"
    bl_label = "Bake Motion Vector"
    bl_description = "Create a motion vector texture for the loaded flipbook (you should specify the size of it)"

    def execute(self, context):
        bake_motion_vector(self, context)
        return {'FINISHED'}

class RenderTimelineOperator(bpy.types.Operator):
    bl_idname = "vfx_toolkit.render_timeline_operator"
    bl_label = "Render Timeline"
    bl_description = "Renders the timeline into a flipbook with the active camera"

    def execute(self, context):
        render_timeline(self, context)
        return {'FINISHED'}

def register_properties():
    bpy.types.Scene.max_column = bpy.props.IntProperty(name="Max Column", default=0,
        description = "max number of columns for the outputted flipbook, 0=automatic"
        )
    bpy.types.Scene.output_frame_x = bpy.props.IntProperty(name="Frame Size X", default=1080)
    bpy.types.Scene.output_frame_y = bpy.props.IntProperty(name="Frame Size Y", default=1080)
    bpy.types.Scene.frames_to_pack_dir = bpy.props.StringProperty(name="Frames Folder", subtype='DIR_PATH',
        description = "folder containing a sequence of png files to be packed into a flipbook"
        )
    bpy.types.Scene.flipbook_file = bpy.props.StringProperty(name="Flipbook File", subtype='FILE_PATH',
        description = "flipbook texture you want to bake motion vector texture for"
        )
    bpy.types.Scene.flipbook_size_x = bpy.props.IntProperty(name="Flipbook Size X", default=0)
    bpy.types.Scene.flipbook_size_y = bpy.props.IntProperty(name="Flipbook Size Y", default=0)
    bpy.types.Scene.frames_to_skip = bpy.props.IntProperty(name="Frames to Skip", default=10,
        description = "how many frames to skip between each rendered frame in timeline"
        )
    bpy.types.Scene.should_render_motion_vector = bpy.props.BoolProperty(name="Motion Vector", default=True)

def unregister_properties():
    del bpy.types.Scene.max_column
    del bpy.types.Scene.output_frame_x
    del bpy.types.Scene.output_frame_y
    del bpy.types.Scene.frames_to_pack_dir
    del bpy.types.Scene.flipbook_file
    del bpy.types.Scene.flipbook_size_x
    del bpy.types.Scene.flipbook_size_y
    del bpy.types.Scene.frames_to_skip
    del bpy.types.Scene.should_render_motion_vector

def register():
    register_properties()
    bpy.utils.register_class(VFXToolkitPackPanel)
    # bpy.utils.register_class(VFXToolkitBakePanel)
    bpy.utils.register_class(VFXToolkitRenderPanel)
    bpy.utils.register_class(PackFlipbookOperator)
    bpy.utils.register_class(BakeMotionVectorOperator)
    bpy.utils.register_class(RenderTimelineOperator)

def unregister():
    unregister_properties()
    bpy.utils.unregister_class(VFXToolkitPackPanel)
    # bpy.utils.unregister_class(VFXToolkitBakePanel)
    bpy.utils.unregister_class(VFXToolkitRenderPanel)
    bpy.utils.unregister_class(PackFlipbookOperator)
    bpy.utils.unregister_class(BakeMotionVectorOperator)
    bpy.utils.unregister_class(RenderTimelineOperator)
