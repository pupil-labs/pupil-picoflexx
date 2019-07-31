## Installation

1. `cd ~/pupil_capture_settings/plugins`
1. `git clone https://github.com/papr/pupil-picoflexx.git picoflexx`
1. Download the [PMD Royale SDK](https://pmdtec.com/picofamily/software-download/)
1. Unzip the coorect zip file for your operating system, e.g. `libroyale-3.21.1.70-APPLE-x86-64Bit.zip` on macOS
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

#### Remote RRF / Picoflexx Mobile

1. Ensure `zstd` is available to import by Pupil.
    * e.g. `pip install --target /path/to/plugins zstd`

### Compile roypycy extension

Note for Windows users: VS2017 or the [VS2017 Build Tools](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017) will be required to compile the extension.

1. Copy (or symlink) the Royale SDK `include` folder here, e.g. `ln -s /path/to/libroyale-3.20.0.62-LINUX-x86-64Bit/include .`
1. `python setup.py build_ext`

If the extension was not compiled/setup correctly, you should see the following line:
```log
world - [WARNING] picoflexx.backend: Pico Flexx backend requirements (roypycy) not installed properly
```
If you're getting `ImportError: cannot import name 'roypycy'` even though the library was compiled, a common cause is the python version used to compile the extension differs from that used by Pupil Capture (Python 3.6).

## Usage

### Backend

1. Connect your Pico Flexx to the computer
1. Start Capture
1. Select the `UVC Manager` menu on the right
1. Select `Pico Flexx` from the selector
1. Click `Activate Pico Flexx`

You should see a color map of the camera's depth output.

### Example Plugin

1. Run the backend usage steps above
1. Enable the `Example Picoflexx Plugin` in the `Plugin Manager` menu

See the `example.plugin.py` file on how to access the depth data from the backend.