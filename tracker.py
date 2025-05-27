import math

class Tracker:
    def __init__(self):
        # Lưu trữ vị trí trung tâm của các đối tượng đã được theo dõi {id: (cx, cy)}
        self.center_points = {}
        # Bộ đếm ID duy nhất, mỗi khi một đối tượng mới được phát hiện, ID sẽ tăng lên
        self.id_count = 0
        # Lưu trữ thông tin bổ sung của đối tượng {id: {'class': class_name, 'confidence': conf}}
        self.object_info = {}

    def update(self, objects_rect_with_info):
        objects_bbs_ids = [] # Danh sách kết quả chứa bounding box, ID và thông tin đầy đủ

        # Lặp qua từng phát hiện mới trong khung hình hiện tại
        for rect_info in objects_rect_with_info:
            x1, y1, x2, y2, conf, class_name = rect_info # Giải nén thông tin phát hiện
            
            # Tính toán tọa độ trung tâm của bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            same_object_detected = False
            # Kiểm tra xem đối tượng này có phải là đối tượng đã được theo dõi trước đó không
            # bằng cách so sánh khoảng cách với các điểm trung tâm đã lưu
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35: # Ngưỡng khoảng cách để coi là cùng một đối tượng (có thể điều chỉnh)
                    self.center_points[id] = (cx, cy) # Cập nhật vị trí trung tâm của đối tượng
                    # Cập nhật thông tin đối tượng (tên lớp và độ tin cậy)
                    self.object_info[id] = {'class': class_name, 'confidence': conf}
                    # Thêm thông tin đầy đủ của đối tượng vào danh sách kết quả
                    objects_bbs_ids.append([x1, y1, x2, y2, id, class_name, conf]) 
                    same_object_detected = True
                    break

            # Nếu đây là một đối tượng mới (chưa được theo dõi), gán ID mới cho nó
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                self.object_info[self.id_count] = {'class': class_name, 'confidence': conf}
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count, class_name, conf])
                self.id_count += 1 # Tăng bộ đếm ID cho đối tượng tiếp theo

        new_center_points = {}
        new_object_info = {}
        
        # Chỉ giữ lại các đối tượng có ID nằm trong danh sách objects_bbs_ids (tức là đã được phát hiện trong khung hình này)
        for obj_bb_id_full in objects_bbs_ids:
            object_id = obj_bb_id_full[4] # ID của đối tượng
            new_center_points[object_id] = self.center_points[object_id]
            new_object_info[object_id] = self.object_info[object_id]

        self.center_points = new_center_points.copy()
        self.object_info = new_object_info.copy()
        
        return objects_bbs_ids # Trả về danh sách các đối tượng đã được theo dõi với ID
