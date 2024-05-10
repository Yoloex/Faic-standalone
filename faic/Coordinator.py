# #!/usr/bin/env python3
import faic.GUI as GUI
import faic.Models as Models
import faic.VideoManager as VM

resize_delay = 1
mem_delay = 1


def coordinator():
    global gui, vm, action, frame, resize_delay, mem_delay

    if gui.get_action_length() > 0:
        action.append(gui.get_action())
    if vm.get_action_length() > 0:
        action.append(vm.get_action())
    if vm.get_frame_length() > 0:
        frame.append(vm.get_frame())
    if len(frame) > 0:
        gui.set_image(frame[0])
        frame.pop(0)
    if len(action) > 0:
        if action[0][0] == "target_faces":
            vm.assign_found_faces(action[0][1])
            action.pop(0)
        elif action[0][0] == "vid_qual":
            vm.vid_qual = int(action[0][1])
            action.pop(0)
        elif action[0][0] == "set_stop":
            vm.stop_marker = action[0][1]
            action.pop(0)
        elif action[0][0] == "ui_vars":
            vm.ui_data = action[0][1]
            action.pop(0)
        elif action[0][0] == "load_webcam":
            vm.load_webcam()
            action.pop(0)
        elif action[0][0] == "control":
            vm.control = action[0][1]
            action.pop(0)
        elif action[0][0] == "parameters":
            vm.parameters = action[0][1]
            action.pop(0)
        elif action[0][0] == "function":
            eval(action[0][1])
            action.pop(0)
        elif action[0][0] == "clear_mem":
            vm.clear_mem()
            action.pop(0)
        elif action[0][0] == "stop_swap":
            frame = []
            action.pop(0)
        else:
            print("Action not found: " + action[0][0] + " " + str(action[0][1]))
            action.pop(0)

    if resize_delay > 100:
        gui.check_for_video_resize()
        resize_delay = 0
    else:
        resize_delay += 1

    if mem_delay > 1000:
        gui.update_vram_indicator()
        mem_delay = 0
    else:
        mem_delay += 1

    vm.process()
    gui.after(1, coordinator)


def run():
    global gui, vm, action, frame, resize_delay, mem_delay

    models = Models.Models()
    gui = GUI.GUI(models)
    vm = VM.VideoManager(models)

    action = []
    frame = []

    vm.load_models()

    gui.initialize_gui()

    coordinator()

    gui.mainloop()

    vm.read_thread.join()
    vm.swap_thread.join()
