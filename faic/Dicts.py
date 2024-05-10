DEFAULT_DATA = {
# Buttons
'ClearmemState':            False,
'FindFacesDisplay':         'text', 
'FindFacesInfoText':        'FIND FACES:\nFinds all new faces in the current frame.',
'FindFacesState':           False,  
'FindFacesText':            'Find Faces',  
'ImgDockState':             False,
'ImgVidMode':               'Videos', 
'ImgVidState':              False,
'LoadSFacesDisplay':         'both', 
'LoadSFacesIcon':            './faic/media/save.png',
'LoadSFacesIconHover':        './faic/media/save.png',     
'LoadSFacesIconOff':          './faic/media/save.png',
'LoadSFacesIconOn':           './faic/media/save.png',
'LoadSFacesInfoText':       'SELECT SOURCE FACES FOLDER:\nSelects and loads Source Faces from Folder. Make sure the folder only contains <good> images.',
'LoadSFacesState':          False,
'LoadSFacesText':           'Select Faces Folder',  
'PerfTestState':            False,
'RecordDisplay':            'icon',     
'RecordInfoText':           'RECORD:\nArms the PLAY button for recording. Press RECORD, then PLAY to record. Press PLAY again to stop recording.',  
'RecordState':              False,       
'SwapFacesDisplay':         'text', 
'SwapFacesInfoText':                'SWAP:\nSwap assigned Source Faces and Target Faces.',
'SwapFacesState':                   False,          
'SwapFacesText':                    'Start Faic Cam',  

'ClearVramButtonDisplay':                   'text',   
'ClearVramButtonInfoText':                  'CLEAR VRAM:\nClears models from your VRAM.',
'ClearVramButtonState':                     False,
'ClearVramButtonText':             'Clear VRAM',
 
#Switches       
'RestorerSwitchInfoText':           'FACE RESTORER:\nRestore the swapped image by upscaling.',
'RestorerSwitchState':              True,

# Sliders
'BlendSliderAmount':                5,
'BlendSliderInc':                   1,  
'BlendSliderInfoText':              'BLEND:\nCombined masks blending distance. Is not applied to the border masks.',
'BlendSliderMax':                   100,
'BlendSliderMin':                   0,
'RestorerSliderAmount':             100,
'RestorerSliderInc':                5,
'RestorerSliderInfoText':           'RESTORER AMOUNT:\nBlends the Restored results back into the original swap.',
'RestorerSliderMax':                100,
'RestorerSliderMin':                0,

# Text Selection
'RestorerDetTypeTextSelInfoText':   'ALIGNMENT:\nSelect how the face is aligned for the Restorer. Original preserves facial features and expressions, but can show some artifacts. Reference softens features. Blend is closer to Reference but is much faster.',
'RestorerDetTypeTextSelMode':       'Blend',
'RestorerDetTypeTextSelModes':      ['Blend'],  
'RestorerTypeTextSelInfoText':      'RESTORER TYPE:\nSelect the Restorer type.',
'RestorerTypeTextSelMode':          'GPEN256',
'RestorerTypeTextSelModes':         ['GPEN256'],
'CameraSourceSelInfoText':      'Camera Source TYPE:\nSelect the right camera source.',
'CameraSourceSelMode':          'HD Webcam',
'CameraSourceSelModes':         ['HD Webcam', 'Full HD Webcam'],

}

PARAM_VARS =    {
    'SimilarThres':             0.85,
    'BorderTopSlider':          10,
    'BorderSidesSlider':        10,
    'BorderBottomSlider':       10,
    'BorderBlurSlider':         10,
    'DetectScore':              0.5,
    'BlendAmout':               5
}
 
PARAMS =   {

    'ClearmemFunction':         'self.clear_mem()',

    'LoadSFacesIcon':            './faic/media/save.png',
    
    'ClearmemMessage':         'CLEAR VRAM - Clears all models from VRAM [LB: Clear]',      
    'RefDelMessage':       'REFERENCE DELTA - Modify the reference points. Turn on mask preview to see adjustments. [LB: on/off, RB: translate x/y, and scale, MW: amount]' ,
 
}