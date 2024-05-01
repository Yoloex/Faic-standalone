import cv2
import threading
import time
import numpy as np
from skimage import transform as trans
from math import floor, ceil
import onnxruntime
import torchvision
import torch
from torchvision import transforms
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from faic.Dicts import PARAM_VARS
torch.set_grad_enabled(False)
onnxruntime.set_default_logger_severity(4)

device = 'cuda'

lock=threading.Lock()

class VideoManager():  
    def __init__(self, models ):
        self.models = models
        # Model related
        self.swapper_model = []             # insightface swapper model

        self.arcface_dst = np.array( [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)     
        self.FFHQ_kps = np.array([[ 192.98138, 239.94708 ], [ 318.90277, 240.1936 ], [ 256.63416, 314.01935 ], [ 201.26117, 371.41043 ], [ 313.08905, 371.15118 ] ])
        
        #Video related
        self.capture = []                   # cv2 video
        self.current_frame = 0              # the input frame index
        self.output_frame = 0               # the output frame index
        
        # Play related
        self.frame_timer = 0.0      # used to set the framerate during playing

        # Queues
        self.action_q = []                  # queue for sending to the coordinator
        self.frame_q = []                   # queue for converted result frames that are ready for coordinator

        # swapping related
        self.found_faces = []   # array that maps the found faces to source faces    
        self.parameters = []

        self.fps = 1.0
        self.fps_average = []
        
        self.perf_test = False
        self.control = []
        
        self.latent = None

        self.process_q =    {
                            "Thread":                   [],
                            "FrameNumber":              [],
                            "ProcessedFrame":           [],
                            "Status":                   'clear',
                            "ThreadTime":               []
                            }   
        self.process_qs = []

    def load_models(self):
        swap = torch.randn([1, 3, 128, 128], dtype=torch.float32).to('cuda')
        img = torch.randn([1, 3, 256, 256], dtype=torch.float32).to('cuda')
        mask = torch.randn([256, 256], dtype=torch.float32).to('cuda')
        latent = torch.randn([1, 512], dtype=torch.float32).to('cuda')
        det_img = torch.randn([3, 480, 640], dtype=torch.float32).to('cuda')
        kps = np.random.randn(5, 2)

        self.models.run_swapper(swap, latent, swap)
        self.models.run_GPEN_256(img, img)
        self.models.run_recognize(det_img, kps)

    def assign_found_faces(self, found_faces):
        self.found_faces = found_faces
    
    def load_webcam(self):
        if self.capture:
            self.capture.release()

        device = 0 if self.parameters['CameraSourceSel'] == 'HD Webcam' else 1

        for i in range(PARAM_VARS['ThreadsNum']):
            new_process_q = self.process_q.copy()
            self.process_qs.append(new_process_q)

        self.capture = cv2.VideoCapture(device)
        self.current_frame = 0
        self.output_frame = 0
        
    ## Action queue
    def add_action(self, action, param):
        temp = [action, param]
        self.action_q.append(temp)    
    
    def get_action_length(self):
        return len(self.action_q)

    def get_action(self):
        action = self.action_q[0]
        self.action_q.pop(0)
        return action
     
    ## Queues for the Coordinator
    def get_frame(self):
        frame = self.frame_q[0]
        self.frame_q.pop(0)
        return frame
    
    def get_frame_length(self):
        return len(self.frame_q)
        
    def find_lowest_frame(self, queues):
        min_frame=999999999
        index=-1
        
        for idx, thread in enumerate(queues):
            frame = thread['FrameNumber']
            if frame != []:
                if frame < min_frame:
                    min_frame = frame
                    index=idx
        return index, min_frame
 
    # @profile
    def process(self):
        
        if len(self.found_faces):
            source = self.found_faces[0]['AssignedEmbedding']
            self.latent = torch.from_numpy(self.models.calc_swapper_latent(source)).float().to('cuda')
        # Add threads to Queue

        if len(self.control) > 0:
            if self.control['SwapFacesButton']:
                for item in self.process_qs:
                    if item['Status'] == 'clear':
                        item['Thread'] = threading.Thread(target=self.thread_video_read).start()
                        item['FrameNumber'] = self.current_frame
                        item['Status'] = 'started'
                        item['ThreadTime'] = time.time()

                        self.current_frame += 1
                        break
            else:
                if self.capture:
                    self.capture.release()
        else:
            if self.capture:
                self.capture.release()
                
        # Always be emptying the queues
        time_diff = time.time() - self.frame_timer

        if time_diff >= 1.0/float(self.fps):

            index, min_frame = self.find_lowest_frame(self.process_qs)

            if index != -1:
                if self.process_qs[index]['Status'] == 'finished':
                    temp = [self.process_qs[index]['ProcessedFrame'], self.process_qs[index]['FrameNumber']]
                    self.frame_q.append(temp)

                    # Report fps, other data
                    self.fps_average.append(1.0/time_diff)
                    if len(self.fps_average) >= floor(self.fps):
                        fps = round(np.average(self.fps_average), 2)
                        msg = "%s fps, %s process time" % (fps, round(self.process_qs[index]['ThreadTime'], 4))
                        self.fps_average = []

                    self.process_qs[index]['Status'] = 'clear'
                    self.process_qs[index]['Thread'] = []
                    self.process_qs[index]['FrameNumber'] = []
                    self.process_qs[index]['ThreadTime'] = []
                    self.frame_timer += 1.0/self.fps
                    
    # @profile
    def thread_video_read(self):  
        with lock:
            success, target_image = self.capture.read()

        if success:
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            temp = self.swap_video(target_image)
            
            for item in self.process_qs:
                if item['FrameNumber'] == self.output_frame:
                    item['ProcessedFrame'] = temp
                    item['Status'] = 'finished'
                    item['ThreadTime'] = time.time() - item['ThreadTime']
                    
                    self.output_frame += 1
                    break

    # @profile
    def swap_video(self, target_image):   
        # Grab a local copy of the parameters to prevent threading issues
        parameters = self.parameters.copy()
        control = self.control.copy()
        
        # Load frame into VRAM
        img = torch.from_numpy(target_image).to('cuda') #HxWxc
        img = img.permute(2,0,1)#cxHxW
        
        #Scale up frame if it is smaller than 512
        img_x = img.size()[2]
        img_y = img.size()[1]
        
        if img_x<512 and img_y<512:
            # if x is smaller, set x to 512
            if img_x <= img_y:
                tscale = v2.Resize((int(512*img_y/img_x), 512), antialias=True)
            else:
                tscale = v2.Resize((512, int(512*img_x/img_y)), antialias=True)

            img = tscale(img)
            
        elif img_x<512:
            tscale = v2.Resize((int(512*img_y/img_x), 512), antialias=True)
            img = tscale(img)
        
        elif img_y<512:
            tscale = v2.Resize((512, int(512*img_x/img_y)), antialias=True)
            img = tscale(img)

        # Find all faces in frame and return a list of 5-pt kpss
        kpss = self.func_w_test("detect", self.models.run_detect, img, max_num=1, score=PARAM_VARS['DetectScore'])
        if kpss is None or len(kpss) == 0:
            return
        img = self.func_w_test("swap_video", self.swap_core, img, kpss[0], parameters, control)
        img = img.permute(1,2,0)
        
        if self.perf_test:
            print('------------------------')  
        
        # Unscale small videos
        if img_x <512 or img_y < 512:
            tscale = v2.Resize((img_y, img_x), antialias=True)
            img = img.permute(2,0,1)
            img = tscale(img)
            img = img.permute(1,2,0)

        img = img.cpu().numpy()  

        return img.astype(np.uint8)

    def findCosineDistance(self, vector1, vector2):
        vec1 = vector1.flatten()
        vec2 = vector2.flatten()

        a = np.dot(vec1.T, vec2)
        b = np.dot(vec1.T, vec1)
        c = np.dot(vec2.T, vec2)
        return 1 - (a/(np.sqrt(b)*np.sqrt(c)))

    def func_w_test(self, name, func, *args, **argsv):
        timing = time.time()
        result = func(*args, **argsv)
        if self.perf_test:
            print(name, round(time.time()-timing, 5), 's')
        return result

    # @profile    
    def swap_core(self, img, kps, parameters, control): # img = RGBa
        # 512 transforms
        dst = self.arcface_dst * 4.0
        dst[:,0] += 32.0
        
        tform = trans.SimilarityTransform()
        tform.estimate(kps, dst) 

        # Scaling Transforms
        t512 = v2.Resize((512, 512), antialias=False)
        #t512a = v2.Resize((512, 512), antialias=True)
        t256 = v2.Resize((256, 256), antialias=False)
        t128 = v2.Resize((128, 128), antialias=False)

        # Grab 512 face from image and create 256 and 128 copys
        original_face_512 = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0), interpolation=v2.InterpolationMode.NEAREST ) 

        original_face_512 = v2.functional.crop(original_face_512, 0,0, 512, 512)# 3, 512, 512
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
        swap = torch.mul(swap, 255) # should I carry [0..1] through the pipe insteadf?
        swap = torch.clamp(swap, 0, 255)
        swap = swap.type(torch.uint8)

        # test = swap.permute(1, 2, 0)
        # test = test.cpu().numpy()
        # cv2.imwrite('2.jpg', test)

        swap_128 = swap
        swap = t512(swap)
        
        # Create border mask
        border_mask = torch.ones((128, 128), dtype=torch.float32, device=device)
        border_mask = torch.unsqueeze(border_mask,0)
        
        # if parameters['BorderState']:
        top = PARAM_VARS['BorderTopSlider']
        left = PARAM_VARS['BorderSidesSlider']
        right = 128 - PARAM_VARS['BorderSidesSlider']
        bottom = 128 - PARAM_VARS['BorderBottomSlider']

        border_mask[:, :top, :] = 0
        border_mask[:, bottom:, :] = 0
        border_mask[:, :, :left] = 0
        border_mask[:, :, right:] = 0

        gauss = transforms.GaussianBlur(PARAM_VARS['BorderBlurSlider']*2+1, (PARAM_VARS['BorderBlurSlider']+1)*0.2)
        border_mask = gauss(border_mask)        

        # Create image mask
        swap_mask = torch.ones((128, 128), dtype=torch.float32, device=device)
        swap_mask = torch.unsqueeze(swap_mask,0)    

        # Restorer
        if parameters["RestorerSwitch"]: 
            swap = self.func_w_test('Restorer', self.apply_restorer, swap, parameters)  
        
        # Add blur to swap_mask results
        gauss = transforms.GaussianBlur(PARAM_VARS['BlendAmout']*2+1, (PARAM_VARS['BlendAmout']+1)*0.2)
        swap_mask = gauss(swap_mask)  
        

        # Combine border and swap mask, scale, and apply to swap
        swap_mask = torch.mul(swap_mask, border_mask)
        swap_mask = t512(swap_mask)
        swap = torch.mul(swap, swap_mask)

        # Cslculate the area to be mergerd back to the original frame
        IM512 = tform.inverse.params[0:2, :]
        corners = np.array([[0,0], [0,511], [511, 0], [511, 511]])

        x = (IM512[0][0]*corners[:,0] + IM512[0][1]*corners[:,1] + IM512[0][2])
        y = (IM512[1][0]*corners[:,0] + IM512[1][1]*corners[:,1] + IM512[1][2])
        
        left = floor(np.min(x))
        if left<0:
            left=0
        top = floor(np.min(y))
        if top<0: 
            top=0
        right = ceil(np.max(x))
        if right>img.shape[2]:
            right=img.shape[2]            
        bottom = ceil(np.max(y))
        if bottom>img.shape[1]:
            bottom=img.shape[1]   

        # Untransform the swap
        swap = v2.functional.pad(swap, (0,0,img.shape[2]-512, img.shape[1]-512))
        swap = v2.functional.affine(swap, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0,interpolation=v2.InterpolationMode.NEAREST, center = (0,0) )  
        swap = swap[0:3, top:bottom, left:right]
        swap = swap.permute(1, 2, 0)
        
        # Untransform the swap mask
        swap_mask = v2.functional.pad(swap_mask, (0,0,img.shape[2]-512, img.shape[1]-512))
        swap_mask = v2.functional.affine(swap_mask, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.NEAREST, center = (0,0) ) 
        swap_mask = swap_mask[0:1, top:bottom, left:right]                        
        swap_mask = swap_mask.permute(1, 2, 0)
        swap_mask = torch.sub(1, swap_mask) 

        # Apply the mask to the original image areas
        img_crop = img[0:3, top:bottom, left:right]
        img_crop = img_crop.permute(1,2,0)            
        img_crop = torch.mul(swap_mask,img_crop)
        
        #Add the cropped areas and place them back into the original image
        swap = torch.add(swap, img_crop)
        swap = swap.type(torch.uint8)
        swap = swap.permute(2,0,1)
        img[0:3, top:bottom, left:right] = swap  


        return img
        
    # @profile    
    def apply_occlusion(self, img, amount):        
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0)
        outpred = torch.ones((256,256), dtype=torch.float32, device=device).contiguous()
        
        self.models.run_occluder(img, outpred)        
                
        outpred = torch.squeeze(outpred)
        outpred = (outpred > 0)
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)
        
        if amount >0:                   
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=device)

            for i in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))       
                outpred = torch.clamp(outpred, 0, 1)
            
            outpred = torch.squeeze(outpred)
            
        if amount <0:      
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=device)

            for i in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))       
                outpred = torch.clamp(outpred, 0, 1)
            
            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            
        outpred = torch.reshape(outpred, (1, 256, 256)) 
        return outpred         
        
    def apply_restorer(self, swapped_face_upscaled, parameters):     
        temp = swapped_face_upscaled
        t512 = v2.Resize((512, 512), antialias=False)
        t256 = v2.Resize((256, 256), antialias=False)  
        
        # If using a separate detection mode
        if parameters['RestorerDetTypeTextSel'] == 'Blend' or parameters['RestorerDetTypeTextSel'] == 'Reference':
            if parameters['RestorerDetTypeTextSel'] == 'Blend':
                # Set up Transformation
                dst = self.arcface_dst * 4.0
                dst[:,0] += 32.0        

            elif parameters['RestorerDetTypeTextSel'] == 'Reference':
                try:
                    dst = self.models.resnet50(swapped_face_upscaled, score=PARAM_VARS['DetectScore']) 
                except:
                    return swapped_face_upscaled       
            
            tform = trans.SimilarityTransform()
            tform.estimate(dst, self.FFHQ_kps)

            # Transform, scale, and normalize
            temp = v2.functional.affine(swapped_face_upscaled, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            temp = v2.functional.crop(temp, 0,0, 512, 512)        
        
        temp = torch.div(temp, 255)
        temp = v2.functional.normalize(temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
        if parameters['RestorerTypeTextSel'] == 'GPEN256':
            temp = t256(temp)
        temp = torch.unsqueeze(temp, 0).contiguous().type(torch.float16)

        # Bindings
        outpred = torch.empty((1,3,256,256), dtype=torch.float16, device=device).contiguous()
        self.models.run_GPEN_256(temp, outpred) 
        
        # Format back to cxHxW @ 255
        outpred = torch.squeeze(outpred)      
        outpred = torch.clamp(outpred, -1, 1)
        outpred = torch.add(outpred, 1)
        outpred = torch.div(outpred, 2)
        outpred = torch.mul(outpred, 255)
        if parameters['RestorerTypeTextSel'] == 'GPEN256':
            outpred = t512(outpred)
            
        # Invert Transform
        if parameters['RestorerDetTypeTextSel'] == 'Blend' or parameters['RestorerDetTypeTextSel'] == 'Reference':
            outpred = v2.functional.affine(outpred, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )

        # Blend
        alpha = float(parameters["RestorerSlider"])/100.0  
        outpred = torch.add(torch.mul(outpred, alpha), torch.mul(swapped_face_upscaled, 1-alpha))

        return outpred        
        
    def apply_fake_diff(self, swapped_face, original_face, DiffAmount):
        swapped_face = swapped_face.permute(1,2,0)
        original_face = original_face.permute(1,2,0)

        diff = swapped_face - original_face
        diff = torch.abs(diff)
        
        # Find the diffrence between the swap and original, per channel
        fthresh = DiffAmount*2.55
        
        # Bimodal
        diff[diff<fthresh] = 0
        diff[diff>=fthresh] = 1 
        
        # If any of the channels exceeded the threshhold, them add them to the mask
        diff = torch.sum(diff, dim=2)
        diff = torch.unsqueeze(diff, 2)
        diff[diff>0] = 1
        
        diff = diff.permute(2,0,1)

        return diff    
    
    def clear_mem(self):
        del self.swapper_model
        del self.GFPGAN_model
        del self.occluder_model
        del self.face_parsing_model
        del self.codeformer_model
        del self.GPEN_256_model
        del self.GPEN_512_model
        del self.resnet_model
        del self.detection_model
        del self.recognition_model
        
        self.swapper_model = []  
        self.GFPGAN_model = []
        self.occluder_model = []
        self.face_parsing_model = []
        self.codeformer_model = []
        self.GPEN_256_model = []
        self.GPEN_512_model = []
        self.resnet_model = []
        self.detection_model = []
        self.recognition_model = []
                
