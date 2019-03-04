## Installation

**ONLY TESTED ON MACOS**

1. `cd ~/pupil_capture_settings/plugins`
1. `git clone https://github.com/papr/pupil-picoflexx.git picoflexx`
1. Download the [PMD Royale SDK](https://pmdtec.com/picofamily/software-download/)
1. Unzip the coorect zip file for your operating system, e.g. `libroyale-3.20.0.62-APPLE-x86-64Bit.zip` on macOS
1. Copy the contents from the `Python` sub folder into `~/pupil_capture_settings/plugins/picoflexx`

To verify that everything works as expected, run the example files:

```bash
cd ~/pupil_capture_settings/plugins/picoflexx
python3 sample_camera_info.py
```

When starting Capture from bundle, you should see the following lines in your log file:

```log
world - [INFO] plugin: Added: <class 'picoflexx.Picoflexx_Manager'>
world - [INFO] plugin: Added: <class 'picoflexx.Picoflexx_Source'>
```

If the requirements where not installed correctly, you should see the following line:
```log
world - [WARNING] plugin: Failed to load 'picoflexx'. Reason: '<reason>'
```

## Usage

1. Connect your Pico Flexx to the computer
1. Start Capture
1. Select the `UVC Manager` menu on the right
1. Select `Pico Flexx` from the selector
1. Click `Activate Pico Flexx`

You should see a color map of the camera's depth output.