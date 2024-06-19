import json
import mimetypes
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import pyvirtualcam
import torch
import torchvision
from PIL import Image, ImageTk

import faic.GUIElements as GE
import faic.Styles as style

torchvision.disable_beta_transforms_warning()


class GUI(tk.Tk):
    def __init__(self, models):
        super().__init__()

        self.models = models
        self.title("Faic")
        self.resizable(width=False, height=False)

        self.action_q = []
        self.camera = None
        self.window_last_change = []
        self.blank = tk.PhotoImage()
        self.input_faces_text = []
        self.source_faces_canvas = []
        self.found_faces_canvas = []
        self.merged_faces_canvas = []
        self.parameters = {}
        self.control = {}
        self.widget = {}
        self.static_widget = {}
        self.layer = {}

        self.json_dict = {
            "source videos": None,
            "source faces": None,
            "saved videos": None,
            "dock_win_geom": [741, 666, 337, 49],
        }

        self.target_face = {
            "TKButton": [],
            "ButtonState": "off",
            "Image": [],
            "Embedding": [],
            "SourceFaceAssignments": [],
            "EmbeddingNumber": 0,  # used for adding additional found faces
            "AssignedEmbedding": [],  # the currently assigned source embedding, including averaged ones
        }
        self.target_faces = []

        self.source_face = {
            "TKButton": [],
            "ButtonState": "off",
            "Image": [],
            "Embedding": [],
        }
        self.source_faces = []
        self.vcam = None

    #####
    def create_gui(self):
        # 1 x 3 Top level grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)

        self.configure(style.frame_style_bg)

        # Top Frame
        top_frame = tk.Frame(self, style.canvas_frame_label_1)
        top_frame.grid(row=0, column=0, sticky="NEWS", padx=1, pady=1)
        top_frame.grid_columnconfigure(0, weight=0)

        # Middle Frame
        middle_frame = tk.Frame(self, style.frame_style_bg)
        middle_frame.grid(row=1, column=0, sticky="NEWS", padx=0, pady=0)
        middle_frame.grid_rowconfigure(0, weight=1)
        # Face Lists
        middle_frame.grid_columnconfigure(0, weight=0)
        # Previews and Parameters
        middle_frame.grid_columnconfigure(1, weight=0)

        # Bottom Frame
        bottom_frame = tk.Frame(self, style.canvas_frame_label_1)
        bottom_frame.grid(row=2, column=0, sticky="NEWS", padx=1, pady=1)
        bottom_frame.grid_columnconfigure(0, minsize=100)
        bottom_frame.grid_columnconfigure(1, weight=0)
        bottom_frame.grid_columnconfigure(2, minsize=100)

        ####### Top Frame
        # Center
        self.layer["topright"] = tk.Frame(
            top_frame, style.canvas_frame_label_1, height=42, width=413
        )
        self.layer["topright"].grid(row=0, column=0, sticky="NEWS", pady=0)
        self.control["ClearVramButton"] = GE.Button(
            self.layer["topright"],
            "ClearVramButton",
            1,
            self.clear_mem,
            None,
            "control",
            x=5,
            y=9,
            width=85,
            height=20,
        )
        self.static_widget["vram_indicator"] = GE.VRAM_Indicator(
            self.layer["topright"], 1, 300, 20, 100, 11
        )

        ####### Middle Frame

        ### Videos and Faces
        v_f_frame = tk.Frame(middle_frame, style.canvas_frame_label_3)
        v_f_frame.grid(row=0, column=0, sticky="NEWS", padx=1, pady=0)
        # Buttons
        v_f_frame.grid_rowconfigure(0, weight=0)
        # Input Media Canvas
        v_f_frame.grid_rowconfigure(1, weight=1)

        # Input Faces Canvas
        v_f_frame.grid_columnconfigure(0, weight=0)
        # Scrollbar
        v_f_frame.grid_columnconfigure(1, weight=0)

        # Input Faces
        # Button Frame
        frame = tk.Frame(v_f_frame, style.canvas_frame_label_2, height=42)
        frame.grid(row=0, column=0, columnspan=2, sticky="NEWS", padx=0, pady=0)

        # Buttons
        self.widget["FacesFolderButton"] = GE.Button(
            frame,
            "LoadSFaces",
            2,
            self.select_faces_path,
            None,
            "control",
            10,
            1,
            width=195,
        )
        self.input_faces_text = GE.Text(frame, "", 2, 10, 20, 190, 20)

        # Scroll Canvas
        self.source_faces_canvas = tk.Canvas(
            v_f_frame, style.canvas_frame_label_3, height=100, width=195
        )
        self.source_faces_canvas.grid(row=1, column=0, sticky="NEWS", padx=10, pady=10)
        self.source_faces_canvas.bind("<MouseWheel>", self.source_faces_mouse_wheel)
        self.source_faces_canvas.create_text(
            8,
            20,
            anchor="w",
            fill="grey25",
            font=("Arial italic", 20),
            text=" Input Faces",
        )

        scroll_canvas = tk.Canvas(
            v_f_frame,
            style.canvas_frame_label_3,
            bd=0,
        )
        scroll_canvas.grid(row=1, column=1, sticky="NEWS", padx=0, pady=0)
        scroll_canvas.grid_rowconfigure(0, weight=1)
        scroll_canvas.grid_columnconfigure(0, weight=0)

        self.static_widget["input_faces_scrollbar"] = GE.Scrollbar_y(
            scroll_canvas, self.source_faces_canvas
        )
        # GE.Separator_y(scroll_canvas, 14, 0)
        GE.Separator_y(v_f_frame, 229, 0)
        GE.Separator_x(v_f_frame, 0, 41)

        ### Parameters
        width = 398

        r_frame = tk.Frame(middle_frame, style.canvas_frame_label_3, bd=0, width=width)
        r_frame.grid(row=0, column=1, sticky="NEWS", pady=0, padx=1)

        r_frame.grid_rowconfigure(0, weight=0)
        r_frame.grid_rowconfigure(1, weight=1)
        r_frame.grid_rowconfigure(2, weight=0)
        r_frame.grid_columnconfigure(0, weight=1)
        r_frame.grid_columnconfigure(1, weight=0)

        ### Preview
        self.layer["preview_column"] = tk.Frame(r_frame, style.canvas_bg, width=width)
        self.layer["preview_column"].grid(row=0, column=0, sticky="NEWS", pady=0)
        self.layer["preview_column"].grid_columnconfigure(0, weight=0)

        # Controls
        self.layer["preview_column"].grid_rowconfigure(0, weight=0)
        # Found Faces
        self.layer["preview_column"].grid_rowconfigure(1, weight=0)

        # Videos
        # Found Faces
        ff_frame = tk.Frame(self.layer["preview_column"], style.canvas_frame_label_1)
        ff_frame.grid(row=1, column=0, sticky="NEWS", pady=1)
        ff_frame.grid_columnconfigure(0, weight=0)
        ff_frame.grid_columnconfigure(1, weight=0)
        ff_frame.grid_rowconfigure(0, weight=0)

        # Buttons
        button_frame = tk.Frame(ff_frame, style.canvas_frame_label_2, height=100, width=112)
        button_frame.grid(row=0, column=0)

        self.widget["FindFacesButton"] = GE.Button(
            button_frame,
            "FindFaces",
            2,
            self.find_face,
            None,
            "control",
            x=0,
            y=0,
            width=112,
            height=33,
        )
        self.widget["SwapFacesButton"] = GE.Button(
            button_frame,
            "SwapFaces",
            4,
            self.toggle_swap,
            None,
            "control",
            x=0,
            y=33,
            width=112,
            height=33,
        )

        # Scroll Canvas
        self.found_faces_canvas = tk.Canvas(ff_frame, style.canvas_frame_label_3, height=100)
        self.found_faces_canvas.grid(row=0, column=1, sticky="NEWS")
        self.found_faces_canvas.bind("<MouseWheel>", self.target_faces_mouse_wheel)
        self.found_faces_canvas.create_text(
            8,
            45,
            anchor="w",
            fill="grey25",
            font=("Arial italic", 20),
            text=" Found Faces",
        )

        self.static_widget["20"] = GE.Separator_y(ff_frame, 111, 0)

        ### Parameter Window Canvas ###

        canvas = tk.Canvas(r_frame, style.canvas_frame_label_3, bd=0)
        canvas.grid(row=1, column=0, sticky="NEWS", pady=0, padx=0)

        parameters_canvas = tk.Frame(
            canvas, style.canvas_frame_label_3, bd=0, width=width, height=600
        )
        parameters_canvas.grid(row=0, column=0, sticky="NEWS", pady=0, padx=0)

        canvas.create_window(0, 0, window=parameters_canvas, anchor="nw")

        ### Parameter Panel Scroll-Y ###
        scroll_canvas = tk.Canvas(
            r_frame,
            style.canvas_frame_label_3,
            bd=0,
        )
        scroll_canvas.grid(row=1, column=1, sticky="NEWS", pady=0)
        scroll_canvas.grid_rowconfigure(0, weight=1)
        scroll_canvas.grid_columnconfigure(0, weight=1)

        ### Layout ###
        top_border_delta = 25
        bottom_border_delta = 5
        switch_delta = 25
        row_delta = 20
        row = 1

        # Camera sources
        row += bottom_border_delta
        self.widget["CameraSourceSel"] = GE.TextSelection(
            parameters_canvas,
            "CameraSourceSel",
            "Camera Source",
            3,
            self.update_data,
            "parameter",
            "parameter",
            398,
            20,
            1,
            row,
            0.62,
        )
        row += top_border_delta
        self.static_widget["9"] = GE.Separator_x(parameters_canvas, 0, row)
        row += bottom_border_delta

        # Restore
        row += bottom_border_delta
        self.widget["RestorerSwitch"] = GE.Switch2(
            parameters_canvas,
            "RestorerSwitch",
            "Restorer",
            3,
            self.update_data,
            "parameter",
            398,
            20,
            1,
            row,
        )
        row += switch_delta
        self.widget["RestorerTypeTextSel"] = GE.TextSelection(
            parameters_canvas,
            "RestorerTypeTextSel",
            "Restorer Type",
            3,
            self.update_data,
            "parameter",
            "parameter",
            398,
            20,
            1,
            row,
            0.62,
        )
        row += row_delta
        self.widget["RestorerDetTypeTextSel"] = GE.TextSelection(
            parameters_canvas,
            "RestorerDetTypeTextSel",
            "Detection Alignment",
            3,
            self.update_data,
            "parameter",
            "parameter",
            398,
            20,
            1,
            row,
            0.62,
        )
        row += row_delta
        self.widget["RestorerSlider"] = GE.Slider2(
            parameters_canvas,
            "RestorerSlider",
            "Blend",
            3,
            self.update_data,
            "parameter",
            398,
            20,
            1,
            row,
            0.62,
        )
        row += bottom_border_delta

        ### Other
        self.layer["tooltip_frame"] = tk.Frame(r_frame, style.canvas_frame_label_3, height=80)
        self.layer["tooltip_frame"].grid(
            row=2, column=0, columnspan=2, sticky="NEWS", padx=0, pady=0
        )
        self.layer["tooltip_label"] = tk.Label(
            self.layer["tooltip_frame"],
            style.info_label,
            wraplength=width - 10,
            image=self.blank,
            compound="left",
            height=80,
            width=width - 10,
        )
        self.layer["tooltip_label"].place(x=5, y=5)
        self.static_widget["13"] = GE.Separator_x(self.layer["tooltip_frame"], 0, 0)

    def update_data(self, mode, name):
        if mode == "parameter":
            self.parameters[name] = self.widget[name].get()
            self.add_action("parameters", self.parameters)

        elif mode == "control":
            self.control[name] = self.widget[name].get()
            self.add_action("control", self.control)

    def set_camera_device(self):
        device = 0 if self.parameters["CameraSourceSel"] == "HD Webcam" else 1
        self.camera = cv2.VideoCapture(device)

    def target_faces_mouse_wheel(self, event):
        self.found_faces_canvas.xview_scroll(1 * int(event.delta / 120.0), "units")

    def source_faces_mouse_wheel(self, event):
        self.source_faces_canvas.yview_scroll(-int(event.delta / 120.0), "units")

        # Center of visible canvas as a percentage of the entire canvas
        center = (self.source_faces_canvas.yview()[1] - self.source_faces_canvas.yview()[0]) / 2
        center = center + self.source_faces_canvas.yview()[0]
        self.static_widget["input_faces_scrollbar"].set(center)

    # refactor - make sure files are closed
    def initialize_gui(self):
        json_object = {}
        # check if data.json exists, if not then create it, else load it
        try:
            data_json_file = open("data.json", "r")
        except Exception as e:
            print("Failed to open data.json with {}".format(e))

            with open("data.json", "w") as outfile:
                json.dump(self.json_dict, outfile)
        else:
            json_object = json.load(data_json_file)
            data_json_file.close()

        # Window position and size
        try:
            self.json_dict["dock_win_geom"] = json_object["dock_win_geom"]
        except Exception as e:
            print("Failed to init with {}".format(e))
            self.json_dict["dock_win_geom"] = self.json_dict["dock_win_geom"]

        # Initialize the window sizes and positions
        self.geometry(
            "%dx%d+%d+%d"
            % (
                self.json_dict["dock_win_geom"][0],
                self.json_dict["dock_win_geom"][1],
                self.json_dict["dock_win_geom"][2],
                self.json_dict["dock_win_geom"][3],
            )
        )
        self.window_last_change = self.winfo_geometry()

        self.resizable(width=False, height=False)

        # Build UI, update ui with default data
        self.create_gui()

        # Create parameters and controls and and selctively fill with UI data
        for key, value in self.widget.items():
            self.widget[key].add_info_frame(self.layer["tooltip_label"])
            if self.widget[key].get_data_type() == "parameter":
                self.parameters[key] = self.widget[key].get()

            elif self.widget[key].get_data_type() == "control":
                self.control[key] = self.widget[key].get()

        try:
            self.json_dict["source faces"] = json_object["source faces"]
        except KeyError:
            self.widget["FacesFolderButton"].error_button()
        else:
            if self.json_dict["source faces"] is None:
                self.widget["FacesFolderButton"].error_button()
            else:
                path = self.create_path_string(self.json_dict["source faces"], 28)
                self.input_faces_text.configure(text=path)

        # Check for a user parameters file and load if present
        try:
            parameters_json_file = open("saved_parameters.json", "r")
        except Exception as e:
            print("Failed to open parameter file with {}".format(e))
            pass
        else:
            temp = json.load(parameters_json_file)
            parameters_json_file.close()
            for key, value in self.parameters.items():
                try:
                    self.parameters[key] = temp[key]
                except KeyError:
                    pass

            # Update the UI
            for key, value in self.parameters.items():
                self.widget[key].set(value, request_frame=False)

        self.add_action("parameters", self.parameters)
        self.add_action("control", self.control)

    def create_path_string(self, path, text_len):
        if len(path) > text_len:
            last_folder = os.path.basename(os.path.normpath(path))
            last_folder_len = len(last_folder)
            if last_folder_len > text_len:
                path = path[:3] + "..." + path[-last_folder_len + 6 :]
            else:
                path = path[: text_len - last_folder_len] + ".../" + path[-last_folder_len:]

        return path

    def select_faces_path(self):
        temp = self.json_dict["source faces"]
        self.json_dict["source faces"] = filedialog.askdirectory(
            title="Select Source Faces Folder", initialdir=temp
        )

        path = self.create_path_string(self.json_dict["source faces"], 28)
        self.input_faces_text.configure(text=path)

        with open("data.json", "w") as outfile:
            json.dump(self.json_dict, outfile)
            outfile.close()
        self.widget["FacesFolderButton"].set(False, request_frame=False)
        self.load_source_faces()

    def load_source_faces(self):
        self.source_faces = []
        self.source_faces_canvas.delete("all")

        shift_i_len = len(self.source_faces)

        # Next Load images
        directory = self.json_dict["source faces"]
        filenames = [
            os.path.join(dirpath, f)
            for (dirpath, dirnames, filenames) in os.walk(directory)
            for f in filenames
        ]

        faces = []

        for file in filenames:  # Does not include full path
            # Find all faces and ad to faces[]
            # Guess File type based on extension
            try:
                file_type = mimetypes.guess_type(file)[0][:5]
            except Exception as e:
                print("Failed to guess type with {}".format(e))
                pass
            else:
                # Its an image
                if file_type == "image":
                    img = cv2.imread(file)

                    if img is not None:
                        img = torch.from_numpy(img.astype("uint8")).to("cuda")

                        pad_scale = 0.2
                        padded_width = int(img.size()[1] * (1.0 + pad_scale))
                        padded_height = int(img.size()[0] * (1.0 + pad_scale))

                        padding = torch.zeros(
                            (padded_height, padded_width, 3),
                            dtype=torch.uint8,
                            device="cuda:0",
                        )

                        width_start = int(img.size()[1] * pad_scale / 2)
                        width_end = width_start + int(img.size()[1])
                        height_start = int(img.size()[0] * pad_scale / 2)
                        height_end = height_start + int(img.size()[0])

                        padding[height_start:height_end, width_start:width_end, :] = img
                        img = padding

                        img = img.permute(2, 0, 1)
                        try:
                            kpss = self.models.run_detect(img, max_num=1)[0]  # Just one face here
                        except IndexError:
                            print("Image cropped too close:", file)
                        else:
                            face_emb, cropped_image = self.models.run_recognize(img, kpss)
                            crop = cv2.cvtColor(cropped_image.cpu().numpy(), cv2.COLOR_BGR2RGB)
                            crop = cv2.resize(crop, (85, 85))
                            faces.append([crop, face_emb])

                    else:
                        print("Bad file", file)

        if len(faces) == 0:
            messagebox.showwarning(
                title="Warning",
                message="No face found! \n Please change your face path.",
            )

        else:
            torch.cuda.empty_cache()

            # Add faces[] images to buttons
            delx, dely = 100, 100

            for i in range(len(faces)):
                # Copy the template dict
                new_source_face = self.source_face.copy()
                self.source_faces.append(new_source_face)

                shift_i = i + shift_i_len

                self.source_faces[shift_i]["Image"] = ImageTk.PhotoImage(
                    image=Image.fromarray(faces[i][0])
                )
                self.source_faces[shift_i]["Embedding"] = faces[i][1]
                self.source_faces[shift_i]["TKButton"] = tk.Button(
                    self.source_faces_canvas,
                    style.media_button_off_3,
                    image=self.source_faces[shift_i]["Image"],
                    height=90,
                    width=90,
                )
                self.source_faces[shift_i]["ButtonState"] = False

                self.source_faces[shift_i]["TKButton"].bind(
                    "<ButtonRelease-1>",
                    lambda event, arg=shift_i: self.toggle_source_faces_buttons_state(event, arg),
                )
                self.source_faces[shift_i]["TKButton"].bind(
                    "<Shift-ButtonRelease-1>",
                    lambda event, arg=shift_i: self.toggle_source_faces_buttons_state_shift(
                        event, arg
                    ),
                )
                self.source_faces[shift_i]["TKButton"].bind(
                    "<MouseWheel>", self.source_faces_mouse_wheel
                )

                self.source_faces_canvas.create_window(
                    (i % 2) * delx,
                    (i // 2) * dely,
                    window=self.source_faces[shift_i]["TKButton"],
                    anchor="nw",
                )

                self.static_widget["input_faces_scrollbar"].resize_scrollbar(None)

    def find_face(self):
        if self.camera:
            self.camera.release()

        self.target_faces = []
        self.set_camera_device()
        start = time.time()

        while True:
            success, img = self.camera.read()

            if success:
                try:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = torch.from_numpy(img).to("cuda")
                    img = img.permute(2, 0, 1)
                    kpss = self.models.run_detect(img, max_num=50)

                    ret = []
                    for i in range(kpss.shape[0]):
                        if kpss is not None:
                            face_kps = kpss[i]

                        face_emb, cropped_img = self.models.run_recognize(img, face_kps)
                        ret.append([face_kps, face_emb, cropped_img])

                except Exception:
                    print("No media selected")

                else:
                    # Find the first face and add this to target_faces[]
                    if ret:
                        face = ret[0]

                        # If we dont find any existing simularities, it means that this is a new face and should be added to our found faces
                        crop = cv2.resize(face[2].cpu().numpy(), (82, 82))

                        new_target_face = self.target_face.copy()
                        self.target_faces.append(new_target_face)
                        last_index = len(self.target_faces) - 1

                        self.target_faces[last_index]["TKButton"] = tk.Button(
                            self.found_faces_canvas,
                            style.media_button_on_3,
                            height=86,
                            width=86,
                        )
                        self.target_faces[last_index]["TKButton"].bind(
                            "<MouseWheel>", self.target_faces_mouse_wheel
                        )
                        self.target_faces[last_index]["ButtonState"] = True
                        self.target_faces[last_index]["Image"] = ImageTk.PhotoImage(
                            image=Image.fromarray(crop)
                        )
                        self.target_faces[last_index]["Embedding"] = face[1]
                        self.target_faces[last_index]["EmbeddingNumber"] = 1

                        # Add image to button
                        self.target_faces[-1]["TKButton"].config(
                            pady=10, image=self.target_faces[last_index]["Image"]
                        )

                        # Add button to canvas
                        self.found_faces_canvas.create_window(
                            (last_index) * 92,
                            8,
                            window=self.target_faces[last_index]["TKButton"],
                            anchor="nw",
                        )
                        self.found_faces_canvas.configure(
                            scrollregion=self.found_faces_canvas.bbox("all")
                        )

                        break

            if time.time() - start > 5:
                messagebox.showwarning(
                    title="Warning",
                    message="Can't detect your face. \nPlease change your camera device",
                )

                break

        self.camera.release()

    def toggle_source_faces_buttons_state(self, event, button):
        state = self.source_faces[button]["ButtonState"]

        for face in self.source_faces:
            face["TKButton"].config(style.media_button_off_3)
            face["ButtonState"] = False

        # Toggle the selected Source Face
        self.source_faces[button]["ButtonState"] = not state

        # If the source face is now on
        if self.source_faces[button]["ButtonState"]:
            self.source_faces[button]["TKButton"].config(style.media_button_on_3)
        else:
            self.source_faces[button]["TKButton"].config(style.media_button_off_3)

        # Determine which target face is selected
        # If there are target faces
        if self.target_faces:
            for face in self.target_faces:
                # Find the first target face that is highlighted
                if face["ButtonState"]:
                    face["SourceFaceAssignments"] = []

                    if self.source_faces[button]["ButtonState"] is True:
                        face["SourceFaceAssignments"].append(button)
                        face["AssignedEmbedding"] = self.source_faces[button]["Embedding"]

                    break

            self.add_action("target_faces", self.target_faces)

    def toggle_source_faces_buttons_state_shift(self, event, button=-1):
        # Set all Source Face buttons to False
        for face in self.source_faces:
            face["TKButton"].config(style.media_button_off_3)

        # Toggle the selected Source Face
        if button != -1:
            self.source_faces[button]["ButtonState"] = not self.source_faces[button]["ButtonState"]

        # Highlight all True buttons
        for face in self.source_faces:
            if face["ButtonState"]:
                face["TKButton"].config(style.media_button_on_3)

        # If a target face is selected
        for tface in self.target_faces:
            if tface["ButtonState"]:
                tface["SourceFaceAssignments"] = []

                # Iterate through all Source faces
                temp_holder = []
                for j in range(len(self.source_faces)):
                    if self.source_faces[j]["ButtonState"] is True:
                        tface["SourceFaceAssignments"].append(j)
                        temp_holder.append(self.source_faces[j]["Embedding"])

                break

        self.add_action("target_faces", self.target_faces)

    def set_image(self, image):
        try:
            if len(image) != 0 and self.widget["SwapFacesButton"].get():
                self.vcam.send(image)
        except Exception as e:
            print("Swap has been stopped with {}".format(e))

    def check_for_video_resize(self):
        # Read the geometry from the last time json was updated. json only updates once the window ahs stopped changing
        win_geom = "%dx%d+%d+%d" % (
            self.json_dict["dock_win_geom"][0],
            self.json_dict["dock_win_geom"][1],
            self.json_dict["dock_win_geom"][2],
            self.json_dict["dock_win_geom"][3],
        )

        # # window has started changing
        if self.winfo_geometry() != win_geom:
            # Resize image in video window
            for k, v in self.widget.items():
                v.hide()
            for k, v in self.static_widget.items():
                v.hide()

            # Check if window has stopped changing
            if self.winfo_geometry() != self.window_last_change:
                self.window_last_change = self.winfo_geometry()
            else:
                for k, v in self.widget.items():
                    v.unhide()
                for k, v in self.static_widget.items():
                    v.unhide()
                # Update json
                str1 = self.winfo_geometry().split("x")
                str2 = str1[1].split("+")
                win_geom = [str1[0], str2[0], str2[1], str2[2]]
                win_geom = [int(strings) for strings in win_geom]
                self.json_dict["dock_win_geom"] = win_geom

                with open("data.json", "w") as outfile:
                    json.dump(self.json_dict, outfile)

    def get_action(self):
        action = self.action_q[0]
        self.action_q.pop(0)

        return action

    def get_action_length(self):
        return len(self.action_q)

    def toggle_swap(self):
        if not self.widget["SwapFacesButton"].get():
            if len(self.target_faces) > 0 and any(
                source_button["ButtonState"] for source_button in self.source_faces
            ):
                self.add_action("load_webcam")
                self.widget["SwapFacesButton"].toggle_button()
                self.widget["SwapFacesButton"].button.configure(text=" Stop Faic Cam")
                self.vcam = pyvirtualcam.Camera(width=1280, height=720, fps=25)

            else:
                messagebox.showwarning(
                    title="Warning",
                    message="Target or source face is not selected!\nPlease select all of these before start",
                )

        else:
            self.add_action("stop_swap")
            self.widget["SwapFacesButton"].toggle_button()
            self.widget["SwapFacesButton"].button.configure(text=" Start Faic Cam")
            self.update_data("control", "SwapFacesButton")
            self.vcam.close()

        self.update_data("control", "SwapFacesButton")

    def add_action(self, action, parameter=None):  #
        self.action_q.append([action, parameter])

    def update_vram_indicator(self):
        try:
            used, total = self.models.get_gpu_memory()
        except Exception as e:
            print("Failed to get GPU info with {}".format(e))
            pass
        else:
            self.static_widget["vram_indicator"].set(used, total)

    def clear_mem(self):
        self.models.delete_models()
        torch.cuda.empty_cache()
