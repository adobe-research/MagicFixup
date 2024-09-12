# copyrights reserved to Michael Gharbi (michael.yanis.gharbi@gmail.com)
import numpy as np
import imageio
from pathlib import Path

import pyglet
from pyglet.gl import *
from ctypes import byref, POINTER
from pyglet import shapes
import glob
import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor

# TODO:
# - shortcuts to send to back / from

class CollageApp(pyglet.window.Window):
    def __init__(
        self,
        image_path: str = "data/lizards.jpg",
        output_folder="output",
        show_help_on_launch: bool = True,
        additional_images: list[str] = [],
    ):
        self.rgb_sprites_batch = pyglet.graphics.Batch()
        self.xy_sprites_batch = pyglet.graphics.Batch()
        
        self.point_markers_batch = pyglet.graphics.Batch()
        self.point_markers_group = pyglet.graphics.Group()
        
        self.output_batch = pyglet.graphics.Batch()
        
        self.multi_click = False


        # TODO auto-download checkpoint
        print("Loading SAM...")
        self.sam = sam_model_registry["default"](checkpoint="data/sam_vit_h_4b8939.pth")
        self.sam_predictor = SamPredictor(self.sam)
        print("done.")

        self.filename = Path(image_path)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True, parents=True)
        
        ref_image = self.prepare_image(image_path)

        self.save_counter = 1
        edit_paths = glob.glob(str(self.output_folder / (self.filename.with_suffix("").name + '__edit__*png')))
        for edit_path in edit_paths:
            if 'output.png' in edit_path:
                continue
            else:
                self.save_counter += 1


        
        if len(additional_images) > 0:
            all_images = [ref_image]
            ref_height, ref_width, _ = ref_image.shape
            for extra_image_path in additional_images:
                extra_image = self.prepare_image(extra_image_path)
                extra_height, extra_width, _ = extra_image.shape
                new_width = int(extra_width * ref_height/ extra_height)
                extra_image_resized = cv2.resize(extra_image, (ref_height, new_width))
                # breakpoint()
                all_images.append(extra_image_resized)
            image = np.concatenate(all_images, axis=1)
                
        else:
            image = ref_image
        
        height, width, _ = image.shape

        super().__init__(width=width, height=height, caption="MagicFixup")

        self.width = width
        self.height = height

        # Store original coordinates
        x, y = np.meshgrid(
            np.linspace(0, 1, width), np.linspace(0, 1, height), indexing="xy"
        )
        xy = np.stack([x, y, np.zeros_like(x), np.ones_like(x)], -1).astype(np.float32)

        self.bg_sprite = Sprite(
            image,
            xy,
            x=0,
            y=0,
            batch=self.rgb_sprites_batch,
            xy_batch=self.xy_sprites_batch,
            center=False,
        )

        self.output_sprite = Sprite(
            image,
            xy,
            x=0,
            y=0,
            batch=self.output_batch,
            xy_batch=self.xy_sprites_batch,
            center=False,
        )

        self._init_framebuffer()

        self.sprites = []
        self.selected_sprite = None
        self.hovered_sprite = None

        self._make_help()
        self.help_group.visible = show_help_on_launch
        self.show_correspondences = False
        self.show_output = False
        
        self.point = None
        self.pos_points = []
        self.neg_points = []
        self.mode = 'positive'
        self.circles = []
        

        print(self.point_markers_group.visible)
    

        # Disable antialiasing (for cleaner sprite edges without artifacts)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        
        self._save(init=True)
    
    def prepare_image(self, image_path):
        image = imageio.imread(image_path)
        image = np.flipud(image)
        assert image.dtype == np.uint8, "expected uint8 image inputs"

        if image.shape[-1] == 4:
            image = image[...,:-1]

        assert len(image.shape) == 3 and image.shape[-1] == 3, "expected RGB input"
        
        # Add an alpha channel
        height, width, _ = image.shape
        alpha = np.ones((height, width, 1), dtype=np.uint8) * 255
        image = np.concatenate([image, alpha], -1)
        
        return image

    def on_draw(self):
        # Enable sprite alpha blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Draw to the offscreen fp32 framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        glViewport(0, 0, self.width, self.height)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        
        if self.show_output:
            self.output_batch.draw()
        else:
            if self.show_correspondences:
                self.xy_sprites_batch.draw()
                self.point_markers_batch.draw()
            else:
                self.rgb_sprites_batch.draw()
                self.point_markers_batch.draw()
        self.help_batch.draw()

        # Copy hidden buffer
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.fbo_id)

        # Account for Retina displays
        pixel_ratio = self.get_pixel_ratio()
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glBlitFramebuffer(
            0,
            0,
            self.width,
            self.height,
            0,
            0,
            int(self.width * pixel_ratio),
            int(self.height * pixel_ratio),
            GL_COLOR_BUFFER_BIT,
            GL_NEAREST,
        )

        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _save(self, init=False):
        self.clear_clicks()
        imname = self.filename.with_suffix("").name + '*'
        rgb_path = (self.output_folder / imname).with_suffix(f".png")
        if init:
            suffix=f'_og'
        else:
            suffix=f'__edit__{self.save_counter:03d}'
        imname = self.filename.with_suffix("").name + suffix
        rgb_path = (self.output_folder / imname).with_suffix(f".png")
        xy_path = (self.output_folder / (imname + f"_correspondences")).with_suffix(
            ".tif"
        )

        print(f"Saving image: {rgb_path}, and correspondences: {xy_path}")
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        glViewport(0, 0, self.width, self.height)

        # Disable sprite highlights for final output
        for s in self.sprites:
            s.hover(False)
        # TODO(mgharbi): restore state after rendering

        # Draw RGB buffer
        glClearColor(0.0, 0.0, 0.0, 0.0)
        self.rgb_sprites_batch.draw()
        rgb = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        glReadPixels(
            0,
            0,
            self.width,
            self.height,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            rgb.ctypes.data_as(POINTER(GLuint)),
        )
        rgb = np.flipud(rgb)
        imageio.imsave(rgb_path, rgb)

        glClearColor(0.0, 0.0, 0.0, 0.0)
        self.xy_sprites_batch.draw()
        xy = np.zeros((self.height, self.width, 4), dtype=np.float32)
        glReadPixels(
            0,
            0,
            self.width,
            self.height,
            GL_RGBA,
            GL_FLOAT,
            xy.ctypes.data_as(POINTER(GLuint)),
        )
        xy = np.flipud(xy)
        imageio.imsave(xy_path, float2uint16(xy))

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        return rgb_path
    
    
    def _update_output(self, output_path):
        image = imageio.imread(output_path)
        image = np.flipud(image)
        assert image.dtype == np.uint8, "expected uint8 image inputs"

        if image.shape[-1] == 4:
            image = image[...,:-1]

        assert len(image.shape) == 3 and image.shape[-1] == 3, "expected RGB input"

        # resize image
        
        image = cv2.resize(image, (self.height, self.width))

        height, width, _ = image.shape

        
        # Add an alpha channel
        alpha = np.ones((height, width, 1), dtype=np.uint8) * 255
        image = np.concatenate([image, alpha], -1)
        
        self.output_image = image

        # update region
        self.output_sprite.update_region(0, 0, self.width, self.height, new_rgb=self.output_image, new_xy=self.output_sprite.xy_map.data)
            

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            print("Quitting...")
            self.close()
        elif (
            symbol == pyglet.window.key.DELETE or symbol == pyglet.window.key.BACKSPACE
        ):
            if self.hovered_sprite is not None:
                print("Deleting sprite... ", self.hovered_sprite)
                sprite_to_delete = self.hovered_sprite
                del_idx = None
                for idx, s in enumerate(self.sprites):
                    if s == sprite_to_delete:
                        del_idx = idx
                        break
                if del_idx is not None:
                    self.hovered_sprite = None
                    del self.sprites[del_idx]
                        
                    
        elif symbol == pyglet.window.key.S:
            self._save()
        elif symbol == pyglet.window.key.SPACE:
            print("toggling help")
            self.help_group.visible = not self.help_group.visible
        elif symbol == pyglet.window.key.C:
            self.show_correspondences = not self.show_correspondences
        elif symbol == pyglet.window.key.R:
            if self.hovered_sprite is not None:
                if modifiers == pyglet.window.key.MOD_SHIFT:
                    self.hovered_sprite.flipy()
                else:
                    self.hovered_sprite.flipx()
        elif symbol == pyglet.window.key.A:
            if len(self.pos_points) > 0:
                x, y = self.pos_points[0]
                self._extract_segment(self.pos_points, self.neg_points)

                self._update_hover(x, y)
                self.pos_points = []
                self.neg_points = []
                self.circles = []
        elif symbol == pyglet.window.key.M:
            self.multi_click = not self.multi_click
        elif symbol == pyglet.window.key.P:
            self.mode = 'positive'
        elif symbol == pyglet.window.key.N:
            self.mode = 'negative'
        elif symbol == pyglet.window.key.U:
            self._upload()
        elif symbol == pyglet.window.key.H:
            self.show_output = not self.show_output
            
        elif symbol == pyglet.window.key.D:
            # duplicate
            if self.hovered_sprite is not None:
                duplicate = self.hovered_sprite.copy()
                self.hovered_sprite = duplicate
                self.sprites.append(duplicate)
        elif symbol == pyglet.window.key._0:
            # clear clicks
            self.clear_clicks()
            

    def clear_clicks(self):
        self.pos_points = []
        self.neg_points = []
        self.circles = []

    def on_key_release(self, symbol, modifiers):
        pass

    def on_mouse_press(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.RIGHT and self.hovered_sprite is not None:
            self.hovered_sprite.reset()
            return

        if button != pyglet.window.mouse.LEFT:
            return

        # Start from back to deal with overlapping sprites: select frontmost
        for sprite in reversed(self.sprites):
            if sprite.intersect(x, y):
                self.selected_sprite = sprite
                return

        if self.hovered_sprite:
            return

        # Segment image if no sprite selected or hovered
        # if modifiers == pyglet.window.key.MOD_SHIFT:
        
        if self.multi_click:
        
            if self.mode == 'positive':
                self.pos_points.append([x,y])
                circle = pyglet.shapes.Circle(
                x=x,#self.width // 2,
                y=y, #self.height // 2,
                radius=3,
                # height=help_height,
                group=self.point_markers_group,
                batch=self.point_markers_batch,
                color=(0, 200, 0, 180),
                )
                circle.anchor_position = 0,0#help_width // 2, help_height
                self.circles.append(circle)
            elif self.mode == 'negative':
                self.neg_points.append([x,y])
                circle = pyglet.shapes.Circle(
                x=x,#self.width // 2,
                y=y, #self.height // 2,
                radius=3,
                # height=help_height,
                group=self.point_markers_group,
                batch=self.point_markers_batch,
                color=(200, 0, 0, 180),
                )
                circle.anchor_position = 0,0#help_width // 2, help_height
                self.circles.append(circle)
            else:
                raise ValueError
        else:
            self.pos_points.append([x,y])
            self._extract_segment(self.pos_points, self.neg_points)

            self._update_hover(x, y)
            self.pos_points = []
            self.neg_points = []
            self.circles = []
        
        

    def on_mouse_release(self, x, y, button, modifiers):
        self.selected_sprite = None

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.selected_sprite is None:
            return
        if modifiers == pyglet.window.key.MOD_COMMAND:
            self.selected_sprite.rotate(dy * 0.5)
        elif modifiers == pyglet.window.key.MOD_SHIFT:
            scale = 1.0 + dy * 0.01
            scale = max(scale, 0.1)
            self.selected_sprite.scale(scale)
        elif modifiers == (pyglet.window.key.MOD_SHIFT + pyglet.window.key.MOD_COMMAND):
            scale = 1.0 + dy * 0.01
            scale = max(scale, 0.1)
            self.selected_sprite.scale_x(scale)
        else:
            self.selected_sprite.move(dx, dy)

    def on_mouse_motion(self, x, y, dx, dy):
        self._update_hover(x, y)

    def _update_hover(self, x, y):
        top_sprite_highlighted = False
        self.hovered_sprite = None
        for sprite in reversed(self.sprites):
            if sprite.intersect(x, y):
                self.set_mouse_cursor(self.get_system_mouse_cursor(self.CURSOR_HAND))
                sprite.hover(not top_sprite_highlighted)
                if not top_sprite_highlighted:
                    self.hovered_sprite = sprite
                top_sprite_highlighted = True
            else:
                sprite.hover(False)

        if not top_sprite_highlighted:
            self.set_mouse_cursor(self.get_system_mouse_cursor(self.CURSOR_DEFAULT))

    def _segment(self, pos_points, neg_points):
        print('pos points', pos_points)
        print('neg points', neg_points)
        input_point = np.array(pos_points + neg_points)
        labels = [1] * len(pos_points) + [0] * len(neg_points)
        print('input points', input_point)
        print('labels', labels)
        input_label = np.array(labels)

        dummy = False

        if dummy:
            mask = np.copy(self.bg_sprite.rgb.data[..., -1]).astype(bool)
            mask[:, : x - 32] = 0
            mask[:, x + 32 :] = 0
            mask[: y - 32] = 0
            mask[y + 32 :] = 0
        else:
            print("Setting SAM input image...")
            self.sam_predictor.set_image(
                self.bg_sprite.rgb.data[..., :3], image_format="RGB"
            )
            print("done")

            print("Running SAM...")
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )
            print('masks shape', masks.shape)
            print("done")

            # Select best mask
            mask = masks[np.argmax(scores), :, :]

            # TODO(mgharbi): apply some post processing, e.g. mask dilation

            # mask = masks[0]

        bbox = BBox.from_mask(mask)

        return np.expand_dims(mask, -1), bbox

    def _extract_segment(self, pos_points, neg_points):
        """Lift segment at (x, y) to a new sprite."""

        mask, bbox = self._segment(pos_points, neg_points)

        rgb_data = self.bg_sprite.rgb.data
        xy_data = self.bg_sprite.xy_map.data

        rgb_crop = uint2float(
            rgb_data[bbox.min_y : bbox.max_y, bbox.min_x : bbox.max_x]
        )
        xy_crop = xy_data[bbox.min_y : bbox.max_y, bbox.min_x : bbox.max_x]
        mask = mask[bbox.min_y : bbox.max_y, bbox.min_x : bbox.max_x].astype(np.float32)

        pos_x, pos_y = bbox.center
        sprite = Sprite(
            np.copy(float2uint8(rgb_crop * mask)),
            np.copy(xy_crop * mask),
            pos_x,
            pos_y,
            batch=self.rgb_sprites_batch,
            xy_batch=self.xy_sprites_batch,
        )
        self.sprites.append(sprite)

        new_rgb = np.copy(rgb_crop)
        new_rgb[..., -1] *= 1 - mask[..., -1]
        # float2uint8(rgb_crop * (1 - mask))

        new_xy = np.copy(xy_crop)
        new_xy[..., -1] *= 1 - mask[..., -1]

        self.bg_sprite.update_region(
            bbox.min_x,
            bbox.min_y,
            bbox.width,
            bbox.height,
            float2uint8(new_rgb),
            new_xy,
        )

    def _init_framebuffer(self):
        self.fbo_id = GLuint(0)
        glGenFramebuffers(1, byref(self.fbo_id))
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)

        # Create the texture (internal pixel data for the framebuffer).
        self.framebuffer_texture = gl.GLuint(0)
        glGenTextures(1, byref(self.framebuffer_texture))
        glBindTexture(GL_TEXTURE_2D, self.framebuffer_texture)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            self.width,
            self.height,
            0,
            GL_RGBA,
            GL_FLOAT,
            # buffer,
            None,
        )

        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            self.framebuffer_texture,
            0,
        )
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _make_help(self):
        help_width = 500
        help_height = 500
        margin = 10

        self.help_batch = pyglet.graphics.Batch()
        self.help_group = pyglet.graphics.Group()

        self.help = pyglet.text.decode_text(
            "Help\n"
            "----\n"
            "SPACEBAR: toggle this help message\n"
            "DELETE or BACKSPACE: delete highlighted sprite\n"
            "S: take a screenshot\n"
            "R: mirror a sprite horizontally\n"
            "D: duplicate a sprite \n"
            "C: show correspondences\n"
            "H: hide/toggle the output\n"
            "SHIFT+R: mirror a sprite vertically\n"
            "\n"
            "Left-Click: add a new sprite segment\n"
            "Right-Click: reset highlited sprite segment\n"
            "Click+Drag: move a sprite\n"
            "Shift+Click+Drag: scale a sprite\n"
            "Cmd+Click+Drag: rotate a sprite\n"
            "Cmd+Shift+Click+Drag: resize a sprite along its x axis.\n"
            "M: enable multi-click segmenting. Details down \n"
            "    (P): enable positive clicks\n"
            "    (N): enable negative clicks\n"
            "    (0): clear the clicks\n"
            "    (A): segment\n"
        )
        self.help.set_style(0, -1, dict(color=[255, 255, 255, 255]))

        self.help_rectangle = pyglet.shapes.Rectangle(
            x=self.width // 2,
            y=self.height - margin,
            width=help_width,
            height=help_height,
            group=self.help_group,
            batch=self.help_batch,
            color=(80, 80, 100, 180),
        )
        self.help_rectangle.anchor_position = help_width // 2, help_height

        self.help_message = pyglet.text.layout.TextLayout(
            self.help,
            multiline=True,
            wrap_lines=False,
            height=help_height,
            width=help_width,
            group=self.help_group,
            batch=self.help_batch,
        )
        self.help_message.x = self.width // 2
        self.help_message.y = self.height - margin * 2
        self.help_message.anchor_x = "center"
        self.help_message.anchor_y = "top"
        
        # circle = shapes.Circle(700, 150, 100, color=(50, 225, 30), batch=self.help_batch, group=self.help_group)
        
        # rectangle = shapes.Rectangle(250, 300, 400, 200, color=(255, 22, 20), batch=self.help_batch)
        # rectangle.opacity = 128
        # rectangle.rotation = 33
        # line = shapes.Line(100, 100, 100, 200, width=19, batch=self.help_batch)
        # line2 = shapes.Line(150, 150, 444, 111, width=4, color=(200, 20, 20), batch=self.help_batch)
        # star = shapes.Star(800, 400, 60, 40, num_spikes=20, color=(255, 255, 0), batch=self.batch)
        # circle = pyglet.shapes.Circle(x=100, y=150, radius=200, color=(244, 0, 0, 180), batch=self.point_markers_batch, group=self.point_markers_group)
        # rect = pyglet.shapes.Rectangle(
        #         x=50,
        #         y=50,
        #         width=200,
        #         height=100,
        #         group=self.point_markers_group,
        #         batch=self.point_markers_batch,
        #         color=(200, 80, 100, 180),
        #     )


def uint2float(im):
    return im.astype(np.float32) / 255


def float2uint8(im):
    return (np.clip(im, 0, 1) * 255).astype(np.uint8)


def float2uint16(im):
    return (np.clip(im, 0, 1) * (2**16 - 1)).astype(np.uint16)


class Sprite:
    def __init__(self, rgb_data, xy_data, x, y, batch=None, xy_batch=None, center=True):
        self.rgb = CustomTexture.create(rgb_data)
        self.xy_map = CustomTexture.create(xy_data)

        if center:
            # Move the anchor to the sprite center
            self.rgb.anchor_x = self.rgb.width // 2
            self.rgb.anchor_y = self.rgb.height // 2
            self.xy_map.anchor_x = self.xy_map.width // 2
            self.xy_map.anchor_y = self.xy_map.height // 2

        # Stores the image appearance and alpha mask
        self.sprite = pyglet.sprite.Sprite(self.rgb, x=x, y=y, batch=batch)

        # Stores the original image coordinates
        self.xy_sprite = pyglet.sprite.Sprite(self.xy_map, x=x, y=y, batch=xy_batch)

        self._original_x = x
        self._original_y = y

    def update_region(self, start_x, start_y, width, height, new_rgb, new_xy):
        glBindTexture(GL_TEXTURE_2D, self.rgb.id)
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            start_x,
            start_y,
            width,
            height,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            new_rgb.ctypes.data_as(POINTER(GLuint)),
        )
        self.rgb.data[start_y : start_y + height, start_x : start_x + width] = new_rgb

        glBindTexture(GL_TEXTURE_2D, self.xy_map.id)
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            start_x,
            start_y,
            width,
            height,
            GL_RGBA,
            GL_FLOAT,
            new_xy.ctypes.data_as(POINTER(GLuint)),
        )
        self.xy_map.data[start_y : start_y + height, start_x : start_x + width] = new_xy

    def reset(self):
        for s in (self.sprite, self.xy_sprite):
            s.rotation = 0
            s.scale = 1
            s.scale_x = 1
            s.scale_y = 1
            s.x = self._original_x
            s.y = self._original_y

    def rotate(self, angle):
        self.sprite.rotation += angle
        self.xy_sprite.rotation += angle

    def scale(self, s):
        self.sprite.scale *= s
        self.xy_sprite.scale *= s

    def scale_x(self, s):
        self.sprite.scale_x *= s
        self.xy_sprite.scale_x *= s

    def move(self, dx, dy):
        for s in (self.sprite, self.xy_sprite):
            s.x += dx
            s.y += dy

    def flipx(self):
        self.sprite.scale_x *= -1
        self.xy_sprite.scale_x *= -1

    def flipy(self):
        self.sprite.scale_y *= -1
        self.xy_sprite.scale_y *= -1

    def intersect(self, x, y):
        canvas_xy = pyglet.math.Vec3(x, y, 1)
        sprite_xy = pyglet.math.Vec3(self.sprite.x, self.sprite.y, 1)
        diff = canvas_xy - sprite_xy

        print(
            f"canvas: {canvas_xy}, sprite: {sprite_xy}, scale: {self.sprite.scale}, {self.sprite.scale_x}, {self.sprite.scale_y}"
        )

        # Transform to local coordinates scale is accounted for in the sprite dimension already
        m = pyglet.math.Mat3()
        m = m.scale(self.sprite.scale_x, self.sprite.scale_y)
        m = m.rotate(self.sprite.rotation)

        xy = pyglet.math.Vec3(diff.x, diff.y, 1.0)
        xformed_xy = m @ xy

        nxy = pyglet.math.Vec2(
            xformed_xy.x / self.sprite.width, xformed_xy.y / self.sprite.height
        )

        if abs(nxy.x) < 0.5 and abs(nxy.y) < 0.5:
            # Within bounding box, now check alpha
            h, w = self.rgb.data.shape[:2]
            x = int((nxy.x + 0.5) * w)
            y = int((nxy.y + 0.5) * h)
            pixel = self.rgb.data[y, x]
            alpha = pixel[-1]
            if alpha > 128:
                return True
        return False

    def hover(self, status: bool):
        if status:
            self.sprite.color = (180, 100, 100)
            self.xy_sprite.color = (180, 100, 100)
        else:
            self.sprite.color = (255, 255, 255)
            self.xy_sprite.color = (255, 255, 255)
    
    def copy(self):
        
        copied_sprite = Sprite(rgb_data=self.rgb.data, xy_data=self.xy_map.data, batch=self.sprite.batch, xy_batch=self.xy_sprite.batch, x=self._original_x, y=self._original_y)
        copied_sprite.sprite.scale = self.sprite.scale
        copied_sprite.sprite.position = self.sprite.position
        copied_sprite.sprite.rotation = self.sprite.rotation
        return copied_sprite


class CustomTexture(pyglet.image.Texture):
    def __init__(self, width, height, target, tex_id, type=GL_UNSIGNED_BYTE):
        super().__init__(width, height, target, tex_id)
        self.data = None

    @classmethod
    def create(
        cls,
        data,
        target=GL_TEXTURE_2D,
        internalformat=GL_RGBA,
        fmt=GL_RGBA,
    ):
        # TODO(mgharbi): not sure we want the interpolation in the end
        min_filter = GL_LINEAR
        mag_filter = GL_LINEAR
        # min_filter = GL_NEAREST
        # mag_filter = GL_NEAREST

        assert (
            len(data.shape) >= 2 and len(data.shape) <= 3
        ), "expected data tensor to be 2D or 3D"
        height, width = data.shape[:2]

        tex_id = GLuint()
        glGenTextures(1, byref(tex_id))
        glBindTexture(target, tex_id.value)
        glTexParameteri(target, GL_TEXTURE_MIN_FILTER, min_filter)
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, mag_filter)

        tex_type = None
        tex_data = None
        if data.dtype == np.uint8:
            assert len(data.shape) == 3 and data.shape[2] <= 4
            tex_type = GL_UNSIGNED_BYTE
            tex_data = (GLubyte * data.size).from_buffer(data)
        elif data.dtype == np.float32:
            assert len(data.shape) == 3 and data.shape[2] <= 4
            tex_type = GL_FLOAT
            tex_data = (GLfloat * data.size).from_buffer(data)
        else:
            raise NotImplementedError(f"Texture type {type} not implemented")

        if internalformat is not None:
            glTexImage2D(
                target,
                0,
                internalformat,
                width,
                height,
                0,
                fmt,
                tex_type,
                tex_data,
            )
            glFlush()

        texture = cls(width, height, target, tex_id.value)
        texture.min_filter = min_filter
        texture.mag_filter = mag_filter
        texture.tex_coords = cls.tex_coords
        texture.data = data

        return texture

    def get_image_data(self, z=0):
        raise NotImplementedError()


class BBox:
    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    @classmethod
    def from_mask(cls, m):
        x = m.max(axis=0)
        y = m.max(axis=1)
        left = np.argmax(np.maximum.accumulate(x))
        right = x.size - np.argmax(np.maximum.accumulate(np.flip(x)))
        top = np.argmax(np.maximum.accumulate(y))
        bottom = y.size - np.argmax(np.maximum.accumulate(np.flip(y)))

        return BBox(left, top, right, bottom)

    @property
    def center(self):
        x = (self.min_x + self.max_x) * 0.5
        y = (self.min_y + self.max_y) * 0.5
        return x, y

    @property
    def width(self):
        return self.max_x - self.min_x

    @property
    def height(self):
        return self.max_y - self.min_y


if __name__ == "__main__":
    print("Launching app. Press `spacebar` for help.")
    # TODO add file browser
    print(len(sys.argv))
    if len(sys.argv) != 2:
        raise ValueError("usage python app.py <image-path>")
    image_path = sys.argv[1].strip() 

    app = CollageApp(image_path=image_path, additional_images=sys.argv[2:])
    pyglet.app.run()
