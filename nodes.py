import os
import torch
import numpy as np
from PIL import Image
import PIL.ImageOps as ImageOps
import folder_paths
import comfy.utils

class ImageConcatenateBatchWithTxt:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_folder1": ("STRING", {"default": ""}),
                "image_folder2": ("STRING", {"default": ""}),
                "output_folder": ("STRING", {"default": ""}),
                "prompt_prefix": ("STRING", {"default": ""}),
                "prompt_subfix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process_images"
    CATEGORY = "image"
 
    def concanate(self, image1, image2, direction, match_image_size, first_image_shape=None):
        # Check if the batch sizes are different
        batch_size1 = image1.shape[0]
        batch_size2 = image2.shape[0]

        if batch_size1 != batch_size2:
            # Calculate the number of repetitions needed
            max_batch_size = max(batch_size1, batch_size2)
            repeats1 = max_batch_size // batch_size1
            repeats2 = max_batch_size // batch_size2
            
            # Repeat the images to match the largest batch size
            image1 = image1.repeat(repeats1, 1, 1, 1)
            image2 = image2.repeat(repeats2, 1, 1, 1)

        if match_image_size:
            # Use first_image_shape if provided; otherwise, default to image1's shape
            target_shape = first_image_shape if first_image_shape is not None else image1.shape

            original_height = image2.shape[1]
            original_width = image2.shape[2]
            original_aspect_ratio = original_width / original_height

            if direction in ['left', 'right']:
                # Match the height and adjust the width to preserve aspect ratio
                target_height = target_shape[1]  # B, H, W, C format
                target_width = int(target_height * original_aspect_ratio)
            elif direction in ['up', 'down']:
                # Match the width and adjust the height to preserve aspect ratio
                target_width = target_shape[2]  # B, H, W, C format
                target_height = int(target_width / original_aspect_ratio)
            
            # Adjust image2 to the expected format for common_upscale
            image2_for_upscale = image2.movedim(-1, 1)  # Move C to the second position (B, C, H, W)
            
            # Resize image2 to match the target size while preserving aspect ratio
            image2_resized = comfy.utils.common_upscale(image2_for_upscale, target_width, target_height, "lanczos", "disabled")
            
            # Adjust image2 back to the original format (B, H, W, C) after resizing
            image2_resized = image2_resized.movedim(1, -1)
        else:
            image2_resized = image2

        # Concatenate based on the specified direction
        if direction == 'right':
            concatenated_image = torch.cat((image1, image2_resized), dim=2)  # Concatenate along width
        elif direction == 'down':
            concatenated_image = torch.cat((image1, image2_resized), dim=1)  # Concatenate along height
        elif direction == 'left':
            concatenated_image = torch.cat((image2_resized, image1), dim=2)  # Concatenate along width
        elif direction == 'up':
            concatenated_image = torch.cat((image2_resized, image1), dim=1)  # Concatenate along height
        return concatenated_image,

    def load_image(self, image_path):
        # image_path = os.path.join(image_folder1,image_file_name)
        
        img = Image.open(image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        i= img
        i = ImageOps.exif_transpose(i)

        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")

        if len(output_images) == 0:
            w = image.size[0]
            h = image.size[1]
        
        if image.size[0] != w or image.size[1] != h:
            return None
        
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
        return output_image
        #return (output_image, output_mask)

    def process_images(self, image_folder1, image_folder2, output_folder,prompt_prefix, prompt_subfix):
        
        if not os.path.isabs(output_folder):
            output_folder = os.path.join(folder_paths.base_path, os.path.normpath(output_folder))
        if not os.path.isabs(image_folder1):
            image_folder1 = os.path.join(folder_paths.base_path, os.path.normpath(image_folder1))
        if not os.path.isabs(image_folder2):
            image_folder2 = os.path.join(folder_paths.base_path, os.path.normpath(image_folder2))

        if not os.path.exists(image_folder1) or not os.path.exists(image_folder2):
            print("Error: One or both image folders do not exist.")
            print(f"Image folder 1: {image_folder1}")
            print(f"Image folder 2: {image_folder2}")
            print(f"Output folder: {output_folder}")
            return image_folder1, image_folder2, output_folder

        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        # 获取两个文件夹中的图片文件
        image_extensions = ('.png', '.jpg', '.jpeg')
        images1 = [f for f in os.listdir(image_folder1) if f.lower().endswith(image_extensions)]
        images2 = [f for f in os.listdir(image_folder2) if f.lower().endswith(image_extensions)]


        for img1_name in images1:
            img1_path = os.path.join(image_folder1, img1_name)
            txt1_name = os.path.splitext(img1_name)[0] + '.txt'
            txt1_path = os.path.join(image_folder1, txt1_name)

            # 加载第一张图片
            img1 = self.load_image(img1_path)

            for img2_name in images2:
                img2_path = os.path.join(image_folder2, img2_name)
                txt2_name = os.path.splitext(img2_name)[0] + '.txt'
                txt2_path = os.path.join(image_folder2, txt2_name)

                # 加载第二张图片
                img2 = self.load_image(img2_path)

                # 调用 concanate 函数
                concatenated_image, = self.concanate(img1, img2, 'right', True)

                # 保存拼接后的图片
                output_img_name = os.path.splitext(img1_name)[0] + '_' + os.path.splitext(img2_name)[0] + '.png'
                output_img_path = os.path.join(output_folder, output_img_name)
                output_image = concatenated_image[0].cpu().numpy()#concatenated_image[0].movedim(0, -1).cpu().numpy()
                output_image = np.clip(255. * output_image, 0, 255).astype(np.uint8)
                Image.fromarray(output_image).save(output_img_path)

                # 读取并合并 txt 文件
                try:
                    with open(txt1_path, 'r', encoding='utf-8') as f1, open(txt2_path, 'r', encoding='utf-8') as f2:
                        txt1_content = f1.read()
                        txt2_content = f2.read()
                        combined_content = prompt_prefix+ ',' + txt1_content + ',' + txt2_content+ ',' + prompt_subfix

                    # 保存合并后的 txt 文件
                    output_txt_name = os.path.splitext(txt1_name)[0] + '_' + os.path.splitext(txt2_name)[0] + '.txt'
                    output_txt_path = os.path.join(output_folder, output_txt_name)
                    with open(output_txt_path, 'w', encoding='utf-8') as f_out:
                        f_out.write(combined_content)
                except FileNotFoundError:
                    continue
        
        return output_folder,


NODE_CLASS_MAPPINGS = {
    "ImageConcatenateBatchWithTxt": ImageConcatenateBatchWithTxt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageConcatenateBatchWithTxt": "Image Concatenate Batch With Txt",
}

