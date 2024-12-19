# Imports
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor
import cv2

# ** Only needs to be run once to add metadata to the model**
 
ObjectDetectorWriter = object_detector.MetadataWriter
_MODEL_PATH = "Models/UniformModel.tflite"
_LABEL_FILE = "Models/labels.txt"
_SAVE_TO_PATH = "model_metadata.tflite"
# Normalization parameters is required when reprocessing the image. It is
# optional if the image pixel values are in range of [0, 255] and the input
# tensor is quantized to uint8. See the introduction for normalization and
# quantization parameters below for more details.
# https://www.tensorflow.org/lite/models/convert/metadata#normalization_and_quantization_parameters)
_INPUT_NORM_MEAN = 127.5
_INPUT_NORM_STD = 127.5

writer = ObjectDetectorWriter.create_for_inference(
    writer_utils.load_file(_MODEL_PATH), [_INPUT_NORM_MEAN], [_INPUT_NORM_STD],
    [_LABEL_FILE])

print(writer.get_metadata_json())

writer_utils.save_file(writer.populate(), _SAVE_TO_PATH)

# **____________________________________________________________________**

model_path = 'model_metadata.tflite'
base_options = core.BaseOptions(file_name=model_path)
detection_options = processor.DetectionOptions(score_threshold=0.2, max_results=10)
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

def draw_detection_result(frame, detection_result):
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
        color = (0, 255, 0)
        thickness = 2
        
        # Draw the bounding box
        cv2.rectangle(frame, start_point, end_point, color, thickness)
        
        # Draw the label
        if detection.categories:
            category = detection.categories[0]
            label = f"{category.category_name}: {category.score:.2f}"
            cv2.putText(frame, label, (start_point[0], start_point[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    return frame

# Enter the path to your video
# Replace with 0 for webcam
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('detections/output.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    tensor_image = vision.TensorImage.create_from_array(rgb_frame)
    
    detection_result = detector.detect(tensor_image)
    
    annotated_frame = draw_detection_result(frame, detection_result)
    
    out.write(annotated_frame)
    
    cv2.imshow('Detection Output', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()   

cv2.destroyAllWindows()
