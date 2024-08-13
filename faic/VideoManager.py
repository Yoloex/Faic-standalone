from math import ceil, floor
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import onnxruntime
import torch
import torchvision
from skimage import transform as trans
from torchvision import transforms
from torchvision.transforms import v2

from faic.Dicts import PARAM_VARS

torchvision.disable_beta_transforms_warning()
torch.set_grad_enabled(False)
onnxruntime.set_default_logger_severity(4)

device = "cuda"


class VideoManager:
    def __init__(self, models):
        self.models = models

        self.arcface_dst = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )

        self.FFHQ_kps = np.array(
            [
                [192.98138, 239.94708],
                [318.90277, 240.1936],
                [256.63416, 314.01935],
                [201.26117, 371.41043],
                [313.08905, 371.15118],
            ]
        )

        self.capture = None  # cv2 video

        self.action_q = []  # queue for sending to coordinator
        self.frame_q = []  # result frames queue ready for coordinator
        self.video_q = Queue(maxsize=1)

        # Threads
        self.read_thread = None
        self.swap_thread = None

        self.found_faces = []
        self.parameters = []
        self.control = []
        self.latent = None

    def load_models(self):
        swap = torch.randn([1, 3, 128, 128], dtype=torch.float32, device=device)
        img = torch.randn([1, 3, 256, 256], dtype=torch.float32, device=device)
        latent = torch.randn([1, 512], dtype=torch.float32, device=device)
        det_img = torch.randn([3, 480, 640], dtype=torch.float32, device=device)
        img_512 = torch.randn([1, 3, 512, 512], dtype=torch.float16, device=device)

        kps = np.random.randn(5, 2)

        self.models.run_swapper(swap, latent, swap)
        self.models.run_GPEN_256(img, img)
        self.models.run_GPEN_512(img_512, img_512)
        self.models.run_recognize(det_img, kps)
        self.models.run_restoreplus(img_512, img_512)

    def assign_found_faces(self, found_faces):
        self.found_faces = found_faces

    def load_webcam(self):
        if self.capture:
            self.capture.release()

        self.capture = cv2.VideoCapture(self.parameters["CameraSourceSel"])
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def add_action(self, action, param):
        temp = [action, param]
        self.action_q.append(temp)

    def get_action_length(self):
        return len(self.action_q)

    def get_action(self):
        action = self.action_q[0]
        self.action_q.pop(0)
        return action

    def get_frame(self):
        frame = self.frame_q[0]
        self.frame_q.pop(0)
        return frame

    def get_frame_length(self):
        return len(self.frame_q)

    def process(self):
        if len(self.found_faces):
            source = self.found_faces[0]["AssignedEmbedding"]
            self.latent = (
                torch.from_numpy(self.models.calc_swapper_latent(source))
                .float()
                .to("cuda")
            )
        # Add threads to Queue

        if len(self.control) > 0:
            if self.control["SwapFacesButton"]:
                if self.read_thread is None:
                    self.read_thread = Thread(target=self.read_frame)
                    self.read_thread.start()
                if self.swap_thread is None:
                    self.swap_thread = Thread(target=self.swap_frame)
                    self.swap_thread.start()
            else:
                if self.capture:
                    self.capture.release()
        else:
            if self.capture:
                self.capture.release()

    def swap_frame(self):
        while True:
            if self.video_q.qsize() > 0:
                target_image = self.video_q.get()
                res = self.swap_video(target_image)
                self.frame_q.append(res)

    def swap_video(self, target_image):
        # Grab a local copy of the parameters to prevent threading issues
        parameters = self.parameters.copy()
        control = self.control.copy()

        # Load frame into VRAM
        img = torch.from_numpy(target_image).to("cuda")  # HxWxc
        img = img.permute(2, 0, 1)  # cxHxW

        # Scale up frame if it is smaller than 512
        img_x = img.size()[2]
        img_y = img.size()[1]

        if img_x < 512 and img_y < 512:
            # if x is smaller, set x to 512
            if img_x <= img_y:
                tscale = v2.Resize((int(512 * img_y / img_x), 512), antialias=True)
            else:
                tscale = v2.Resize((512, int(512 * img_x / img_y)), antialias=True)

            img = tscale(img)

        elif img_x < 512:
            tscale = v2.Resize((int(512 * img_y / img_x), 512), antialias=True)
            img = tscale(img)

        elif img_y < 512:
            tscale = v2.Resize((512, int(512 * img_x / img_y)), antialias=True)
            img = tscale(img)

        # Find all faces in frame and return a list of 5-pt kpss
        kpss = self.models.run_detect(img, max_num=1, score=PARAM_VARS["DetectScore"])

        if kpss is None or len(kpss) == 0:
            return

        img = self.swap_core(img, kpss[0], parameters, control)
        img = img.permute(1, 2, 0).cpu().numpy()
        return img.astype(np.uint8)

    def swap_core(self, img, kps, parameters, control):
        # 512 transforms
        dst = self.arcface_dst * 4.0
        dst[:, 0] += 32.0

        tform = trans.SimilarityTransform()
        tform.estimate(kps, dst)

        # Scaling Transforms
        t512 = v2.Resize((512, 512), antialias=False)
        # t512a = v2.Resize((512, 512), antialias=True)
        t256 = v2.Resize((256, 256), antialias=False)
        t128 = v2.Resize((128, 128), antialias=False)

        # Grab 512 face from image and create 256 and 128 copys
        original_face_512 = v2.functional.affine(
            img,
            tform.rotation * 57.2958,
            (tform.translation[0], tform.translation[1]),
            tform.scale,
            0,
            center=(0, 0),
            interpolation=v2.InterpolationMode.NEAREST,
        )

        original_face_512 = v2.functional.crop(
            original_face_512, 0, 0, 512, 512
        )  # 3, 512, 512
        original_face_256 = t256(original_face_512)
        original_face_128 = t128(original_face_256)

        # Prepare for swapper formats
        swap = torch.reshape(original_face_128, (1, 3, 128, 128)).contiguous()
        swap = torch.div(swap, 255)

        # Swap Face and blend according to Strength
        itex = 1

        # Additional swaps based on strength
        for i in range(itex):
            prev_swap = swap.clone()
            self.models.run_swapper(prev_swap, self.latent, swap)

        # Format to 3x128x128 [0..255] uint8
        swap = torch.squeeze(swap)
        swap = torch.mul(swap, 255)  # should I carry [0..1] through the pipe insteadf?
        swap = torch.clamp(swap, 0, 255)
        swap = swap.type(torch.uint8)
        swap = t512(swap)

        # Create border mask
        border_mask = torch.ones((128, 128), dtype=torch.float32, device=device)
        border_mask = torch.unsqueeze(border_mask, 0)

        # if parameters['BorderState']:
        top = PARAM_VARS["BorderTopSlider"]
        left = PARAM_VARS["BorderSidesSlider"]
        right = 128 - PARAM_VARS["BorderSidesSlider"]
        bottom = 128 - PARAM_VARS["BorderBottomSlider"]

        border_mask[:, :top, :] = 0
        border_mask[:, bottom:, :] = 0
        border_mask[:, :, :left] = 0
        border_mask[:, :, right:] = 0

        gauss = transforms.GaussianBlur(
            PARAM_VARS["BorderBlurSlider"] * 2 + 1,
            (PARAM_VARS["BorderBlurSlider"] + 1) * 0.2,
        )
        border_mask = gauss(border_mask)

        # Create image mask
        swap_mask = torch.ones((128, 128), dtype=torch.float32, device=device)
        swap_mask = torch.unsqueeze(swap_mask, 0)

        # Restorer
        if parameters["RestorerSwitch"]:
            swap = self.apply_restorer(swap, parameters)

        # Add blur to swap_mask results
        gauss = transforms.GaussianBlur(
            PARAM_VARS["BlendAmout"] * 2 + 1,
            (PARAM_VARS["BlendAmout"] + 1) * 0.2,
        )
        swap_mask = gauss(swap_mask)

        # Combine border and swap mask, scale, and apply to swap
        swap_mask = torch.mul(swap_mask, border_mask)
        swap_mask = t512(swap_mask)
        swap = torch.mul(swap, swap_mask)

        # Cslculate the area to be mergerd back to the original frame
        IM512 = tform.inverse.params[0:2, :]
        corners = np.array([[0, 0], [0, 511], [511, 0], [511, 511]])

        x = IM512[0][0] * corners[:, 0] + IM512[0][1] * corners[:, 1] + IM512[0][2]
        y = IM512[1][0] * corners[:, 0] + IM512[1][1] * corners[:, 1] + IM512[1][2]

        left = floor(np.min(x))
        if left < 0:
            left = 0
        top = floor(np.min(y))
        if top < 0:
            top = 0
        right = ceil(np.max(x))
        if right > img.shape[2]:
            right = img.shape[2]
        bottom = ceil(np.max(y))
        if bottom > img.shape[1]:
            bottom = img.shape[1]

        # Untransform the swap
        swap = v2.functional.pad(swap, (0, 0, img.shape[2] - 512, img.shape[1] - 512))
        swap = v2.functional.affine(
            swap,
            tform.inverse.rotation * 57.2958,
            (tform.inverse.translation[0], tform.inverse.translation[1]),
            tform.inverse.scale,
            0,
            interpolation=v2.InterpolationMode.NEAREST,
            center=(0, 0),
        )
        swap = swap[0:3, top:bottom, left:right]
        swap = swap.permute(1, 2, 0)

        # Untransform the swap mask
        swap_mask = v2.functional.pad(
            swap_mask, (0, 0, img.shape[2] - 512, img.shape[1] - 512)
        )
        swap_mask = v2.functional.affine(
            swap_mask,
            tform.inverse.rotation * 57.2958,
            (tform.inverse.translation[0], tform.inverse.translation[1]),
            tform.inverse.scale,
            0,
            interpolation=v2.InterpolationMode.NEAREST,
            center=(0, 0),
        )
        swap_mask = swap_mask[0:1, top:bottom, left:right]
        swap_mask = swap_mask.permute(1, 2, 0)
        swap_mask = torch.sub(1, swap_mask)

        # Apply the mask to the original image areas
        img_crop = img[0:3, top:bottom, left:right]
        img_crop = img_crop.permute(1, 2, 0)
        img_crop = torch.mul(swap_mask, img_crop)

        # Add the cropped areas and place them back into the original image
        swap = torch.add(swap, img_crop)
        swap = swap.type(torch.uint8)
        swap = swap.permute(2, 0, 1)
        img[0:3, top:bottom, left:right] = swap

        return img

    def apply_restorer(self, swapped_face_upscaled, parameters):
        temp = swapped_face_upscaled
        t512 = v2.Resize((512, 512), antialias=False)
        t256 = v2.Resize((256, 256), antialias=False)

        # If using a separate detection mode
        if (
            parameters["RestorerDetTypeTextSel"] == "Blend"
            or parameters["RestorerDetTypeTextSel"] == "Reference"
        ):
            if parameters["RestorerDetTypeTextSel"] == "Blend":
                # Set up Transformation
                dst = self.arcface_dst * 4.0
                dst[:, 0] += 32.0

            tform = trans.SimilarityTransform()
            tform.estimate(dst, self.FFHQ_kps)

            # Transform, scale, and normalize
            temp = v2.functional.affine(
                swapped_face_upscaled,
                tform.rotation * 57.2958,
                (tform.translation[0], tform.translation[1]),
                tform.scale,
                0,
                center=(0, 0),
            )
            temp = v2.functional.crop(temp, 0, 0, 512, 512)

        temp = torch.div(temp, 255)
        temp = v2.functional.normalize(
            temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False
        )

        if parameters["RestorerTypeTextSel"] == "GPEN256":
            temp = t256(temp)
            temp = torch.unsqueeze(temp, 0).contiguous().type(torch.float16)
            outpred = torch.empty(
                (1, 3, 256, 256), dtype=torch.float16, device=device
            ).contiguous()
            self.models.run_GPEN_256(temp, outpred)

        if parameters["RestorerTypeTextSel"] == "GPEN512":
            temp = torch.unsqueeze(temp, 0).contiguous().type(torch.float16)
            outpred = torch.empty(
                (1, 3, 512, 512), dtype=torch.float16, device=device
            ).contiguous()
            self.models.run_GPEN_512(temp, outpred)

        if parameters["RestorerTypeTextSel"] == "RestorePlus":
            temp = torch.unsqueeze(temp, 0).contiguous().type(torch.float16)
            outpred = torch.empty(
                (1, 3, 512, 512), dtype=torch.float16, device=device
            ).contiguous()
            self.models.run_restoreplus(temp, outpred)

        outpred = torch.clamp(outpred, -1, 1)
        outpred = torch.add(outpred, 1)
        outpred = torch.div(outpred, 2)
        outpred = torch.squeeze(outpred)
        outpred = torch.mul(outpred, 255)

        if parameters["RestorerTypeTextSel"] == "GPEN256":
            outpred = t512(outpred)

        # Invert Transform
        if (
            parameters["RestorerDetTypeTextSel"] == "Blend"
            or parameters["RestorerDetTypeTextSel"] == "Reference"
        ):
            outpred = v2.functional.affine(
                outpred,
                tform.inverse.rotation * 57.2958,
                (tform.inverse.translation[0], tform.inverse.translation[1]),
                tform.inverse.scale,
                0,
                interpolation=v2.InterpolationMode.BILINEAR,
                center=(0, 0),
            )

        # Blend
        alpha = float(parameters["RestorerSlider"]) / 100.0
        outpred = torch.add(
            torch.mul(outpred, alpha),
            torch.mul(swapped_face_upscaled, 1 - alpha),
        )

        return outpred

    def read_frame(self):
        while True:
            success, img = self.capture.read()

            if success:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.video_q.put(img)

    def clear_mem(self):
        del self.models
