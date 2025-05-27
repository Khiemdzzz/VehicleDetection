import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import os
import sys
# Import các module tùy chỉnh
from tracker import Tracker 
from vehicle_counter import VehicleCounter 
from config import * 

# --- Hàm Callback chuột để debug tọa độ ---
def RGB(event, x, y, flags, param):

    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        # print(point) # Bỏ comment để xem tọa độ chuột trong console

# --- Khởi tạo và Thiết lập ---
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Kiểm tra sự tồn tại của các file cần thiết trước khi tiếp tục
if not os.path.exists(MODEL_PATH):
    print(f"Lỗi: Không tìm thấy file mô hình tại {MODEL_PATH}")
    sys.exit()
if not os.path.exists(VIDEO_PATH):
    print(f"Lỗi: Không tìm thấy file video tại {VIDEO_PATH}")
    sys.exit()
if not os.path.exists(CLASS_LIST_PATH):
    print(f"Lỗi: Không tìm thấy file danh sách lớp tại {CLASS_LIST_PATH}")
    sys.exit()

# Tải mô hình YOLO
model = YOLO(MODEL_PATH)

# Khởi tạo đối tượng đọc video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Lỗi: Không thể mở video từ đường dẫn {VIDEO_PATH}")
    sys.exit()

# Đọc danh sách các lớp từ file coco.txt
with open(CLASS_LIST_PATH, "r") as my_file:
    class_list = [line.strip() for line in my_file.readlines()]
print(f"Loaded class names from {CLASS_LIST_PATH}: {class_list}")

# Khởi tạo Tracker và VehicleCounter
tracker = Tracker()
# Truyền danh sách các lớp phương tiện cần đếm khi khởi tạo VehicleCounter
vehicle_counter = VehicleCounter(COUNTING_LINE_Y1, VEHICLE_CLASSES) 


# Bộ đếm khung hình để bỏ qua một số khung hình nhằm tăng tốc độ xử lý
frame_count = 0

# Bắt đầu xử lý video từng khung hình
while True:
    ret, frame = cap.read() # Đọc một khung hình từ video
    if not ret: # Nếu không đọc được khung hình nào (cuối video), thoát vòng lặp
        print("Đã kết thúc video hoặc không thể đọc khung hình.")
        break
    
    frame_count += 1
    if frame_count % FRAME_SKIP_INTERVAL != 0:
        continue
    
    # Thay đổi kích thước khung hình để xử lý và hiển thị nhất quán
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Dự đoán đối tượng trong khung hình bằng mô hình YOLO
    results = model.predict(frame)
    detections = results[0].boxes.data # Trích xuất dữ liệu phát hiện
    
    # Chuyển đổi kết quả dự đoán thành pandas DataFrame để dễ dàng thao tác
    px = pd.DataFrame(detections.cpu().numpy()).astype("float")

    # Lọc các phát hiện theo độ tin cậy và các lớp phương tiện quan tâm
    vehicles_for_tracker = []
    for index, row in px.iterrows():
        # Trích xuất tọa độ bounding box, độ tin cậy và ID lớp
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        conf = row[4] # Điểm độ tin cậy
        class_id = int(row[5]) # ID lớp từ mô hình YOLO
        
        # Kiểm tra ID lớp hợp lệ và độ tin cậy
        if class_id >= 0 and class_id < len(class_list):
            class_name = class_list[class_id]
            if class_name in VEHICLE_CLASSES and conf > CONFIDENCE_THRESHOLD:
                vehicles_for_tracker.append([x1, y1, x2, y2, conf, class_name])
    
   
    tracked_vehicles_info = tracker.update(vehicles_for_tracker)

    # Lấy các ID đối tượng đang hoạt động trong khung hình hiện tại để dọn dẹp trạng thái sau
    active_object_ids = set() 

    # --- Logic Đếm và Cập nhật Trạng thái ---
    # Lặp qua tất cả các phương tiện được tracker theo dõi
    for bbox_info in tracked_vehicles_info:
        # Giải nén thông tin từ tracker: x1, y1, x2, y2, ID, tên lớp, độ tin cậy
        x1, y1, x2, y2, object_id, class_name, conf = bbox_info 
        cx = int((x1 + x2) / 2) # Tọa độ X trung tâm của bounding box
        cy = int((y1 + y2) / 2) # Tọa độ Y trung tâm của bounding box

        active_object_ids.add(object_id) # Thêm ID vào set các đối tượng đang hoạt động

        # Cập nhật trạng thái và đếm thông qua VehicleCounter
        # Logic đếm hai chiều được xử lý bên trong class VehicleCounter
        vehicle_counter.update(object_id, class_name, cy) 
        
        # --- Vẽ Bounding Box và Thông tin ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2) # Vẽ bounding box màu tím
        
        # Chuẩn bị văn bản để hiển thị: ID, Tên lớp, Độ tin cậy
        display_text = f'ID:{object_id} {class_name} {conf:.2f}' 

        # Vẽ văn bản với một hình chữ nhật nền bằng cvzone để dễ đọc hơn
        cvzone.putTextRect(frame, display_text, (x1, y1 - 10), scale=0.8, thickness=1, offset=5) 

    # Dọn dẹp trạng thái của các đối tượng không còn được theo dõi bởi tracker
    vehicle_counter.clean_up_states(active_object_ids)

    # --- Vẽ đường kẻ đếm trên khung hình ---
    # Sử dụng COUNTING_LINE_Y1 từ config.py
    cv2.line(frame, (1, COUNTING_LINE_Y1), (FRAME_WIDTH - 2, COUNTING_LINE_Y1), (0, 255, 0), 2) # Đường kẻ màu xanh lá

    # --- Hiển thị tổng số lượng đếm trên khung hình ---
    # Lấy kết quả đếm cho cả hai chiều
    current_counts = vehicle_counter.get_counts()
    counts_down = current_counts['down']
    counts_up = current_counts['up']

    y_offset_display = 50
    cvzone.putTextRect(frame, 'Counts (Down):', (50, y_offset_display), scale=1.5, thickness=2, offset=10, colorR=(0, 255, 0))
    y_offset_display += 40 
    for cls in VEHICLE_CLASSES:
        cvzone.putTextRect(frame, f'{cls.capitalize()}: {counts_down[cls]}', 
                                (50, y_offset_display), scale=1.2, thickness=2, offset=8, 
                                colorR=(0, 200, 0)) # Màu xanh lá đậm
        y_offset_display += 30 
    
    y_offset_display += 20 # Khoảng cách giữa hai nhóm đếm

    cvzone.putTextRect(frame, 'Counts (Up):', (50, y_offset_display), scale=1.5, thickness=2, offset=10, colorR=(0, 0, 255))
    y_offset_display += 40 
    for cls in VEHICLE_CLASSES:
        cvzone.putTextRect(frame, f'{cls.capitalize()}: {counts_up[cls]}', 
                                (50, y_offset_display), scale=1.2, thickness=2, offset=8, 
                                colorR=(0, 0, 200)) # Màu đỏ đậm
        y_offset_display += 30 

    # --- Hiển thị khung hình trong cửa sổ 'RGB' ---
    cv2.imshow("RGB", frame)
    
    # Đợi 1 mili giây và kiểm tra phím 'Esc' (mã 27) để thoát vòng lặp
    if cv2.waitKey(1) & 0xFF == 27: 
        break

# --- Kết thúc và dọn dẹp tài nguyên ---
print("\n--- Final Counts ---")
final_counts = vehicle_counter.get_counts()
print("Counts Down (From above line to below line):")
for cls, count in final_counts['down'].items():
    print(f'  Total {cls} count: {count}')
print("\nCounts Up (From below line to above line):")
for cls, count in final_counts['up'].items():
    print(f'  Total {cls} count: {count}')

cap.release() # Giải phóng đối tượng đọc video
cv2.destroyAllWindows() # Đóng tất cả các cửa sổ OpenCV
