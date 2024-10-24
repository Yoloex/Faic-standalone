# #!/usr/bin/env python3
import faic.gui as GUI
import faic.models as Models
import faic.video_manager as VM

RESIZE_DELAY = 1
MEM_DELAY = 1


def coordinator():
    global gui, vm, action, frame, RESIZE_DELAY, MEM_DELAY

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

    if RESIZE_DELAY > 100:
        gui.check_for_video_resize()
        RESIZE_DELAY = 0
    else:
        RESIZE_DELAY += 1

    if MEM_DELAY > 1000:
        gui.update_vram_indicator()
        MEM_DELAY = 0
    else:
        MEM_DELAY += 1

    vm.process()
    gui.after(1, coordinator)


def run():
    global gui, vm, action, frame, RESIZE_DELAY, MEM_DELAY

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
