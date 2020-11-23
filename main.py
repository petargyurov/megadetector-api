from tf_detector import TFDetector

model_file = 'md_v4.1.0.pb'
tf_detector = TFDetector(model_file, output_path='/output')

results = tf_detector.run_detection(input_path='test_imgs/animals',
                                    output_file='results.json')
