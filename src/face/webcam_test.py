from pygrabber.dshow_graph import FilterGraph
graph = FilterGraph()
devices = graph.get_input_devices()
for i, d in enumerate(devices):
    print(i, d)


# import cv2
# import time

# cap = cv2.VideoCapture(0)
# time.sleep(2)  # give camera time to warm up

# ret, frame = cap.read()
# cap.release()

# if not ret:
#     print("FAILED: Could not read from webcam at all. Check if another app is using it.")
# else:
#     cv2.imwrite("webcam_test.jpg", frame)
#     print(f"SUCCESS: Captured frame, shape={frame.shape}. Saved to webcam_test.jpg")



