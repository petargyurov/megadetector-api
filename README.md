# Megadetector API

At its core, this repository is a fork of the Megadetector project that sits within
the [microsoft/CameraTraps](https://github.com/microsoft/CameraTraps) repository.
The aim is to expose a simpler API to get started with the model, for both 
programmers and non-programmers. 

The original CameraTraps repository hosts a number of projects and tools and is 
difficult to navigate. In addition, the code provided for running the 
Megadetector model is designed specifically as a CLI tool, making it difficult 
to use as a script in a pipeline or to build up on and extend.

### Basic Usage

```python
from tf_detector import TFDetector

tf_detector = TFDetector(model_path='md_v4.1.0.pb', output_path='/output')

results = tf_detector.run_detection(input_path='test_imgs/stoats')

```

##### Example of `results.json`
```json
[
  {
    "file": "my_images/img1.JPG",
    "max_detection_conf": 0.981,
    "detections": [
      {
        "category": "1",
        "conf": 0.981,
        "bbox": [
          0.4011,
          0.6796,
          0.07269,
          0.07873
        ]
      }
    ]
  },
  ...
]
```


### Features

- ✔️Simple API to expose the model and run it on images
- ✔️Structural refactoring and code improvements
- ✔️CLI interface
- ✔️For GUI see [megadetector-gui](https://github.com/petargyurov/megadetector-gui) which is powered by this repo
- **TODO:** add basic evaluation metrics to output
- **TODO:** add proper logging functionality
- **TODO:** add different export formats (e.g.: `.csv`)


### What's changed from the original?

For the most part, the functionality of the base `TFDetector` class remains
unchanged.

- abstracted the main class, `TFDetector`, from (almost) any CLI functionality
- the class has its own method to start detections
- class instances can now be initialised with threshold params, amongst other things
- refactored the class`ImagePathUtils` into a simple `utils.py` module; 
there was no reason to have it as a class
- moved some additional methods to `utils.py`