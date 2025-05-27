# vehicle_counter.py

class VehicleCounter:
    """
    Quản lý việc đếm phương tiện dựa trên trạng thái vượt qua một đường kẻ nhất định.
    Hỗ trợ đếm cả hai chiều: từ trên xuống và từ dưới lên.
    """
    def __init__(self, cy_line, vehicle_classes_to_count):
        """
        Khởi tạo bộ đếm.
        :param cy_line: Tọa độ Y của đường kẻ đếm.
        :param vehicle_classes_to_count: Danh sách các tên lớp phương tiện cần đếm.
        """
        self.cy_line = cy_line
        # Bộ đếm cho phương tiện đi xuống (từ trên xuống dưới vạch)
        self.vehicle_counts_down = {cls: 0 for cls in vehicle_classes_to_count} 
        # Bộ đếm cho phương tiện đi lên (từ dưới lên trên vạch)
        self.vehicle_counts_up = {cls: 0 for cls in vehicle_classes_to_count}   
        # Lưu trạng thái của mỗi đối tượng: 'above_line' (trên vạch), 'below_line' (dưới vạch)
        self.object_crossing_state = {} 

    def update(self, object_id, class_name, current_y_center):
        """
        Cập nhật trạng thái của một đối tượng và kiểm tra xem nó có vượt qua đường kẻ không.
        :param object_id: ID duy nhất của đối tượng (từ tracker).
        :param class_name: Tên lớp của đối tượng (ví dụ: 'car', 'truck').
        :param current_y_center: Tọa độ Y của trung tâm bounding box hiện tại của đối tượng.
        :return: True nếu đối tượng vừa được đếm, False nếu không.
        """
        prev_state = self.object_crossing_state.get(object_id)
        
        current_position_is_above = current_y_center < self.cy_line
        current_position_is_below = current_y_center >= self.cy_line

        # Xác định trạng thái hiện tại của đối tượng so với đường kẻ
        current_state = None
        if current_position_is_above:
            current_state = 'above_line'
        elif current_position_is_below:
            current_state = 'below_line'

        counted_this_pass = False

        # --- Logic đếm cho chiều từ trên xuống dưới ---
        if prev_state == 'above_line' and current_state == 'below_line':
            self.vehicle_counts_down[class_name] += 1
            counted_this_pass = True
            # print(f"ĐÃ ĐẾM (Từ trên xuống): {class_name} với ID {object_id}. Số lượng mới: {self.vehicle_counts_down[class_name]}") # Debugging
        # --- Logic đếm cho chiều từ dưới lên trên ---
        elif prev_state == 'below_line' and current_state == 'above_line':
            self.vehicle_counts_up[class_name] += 1
            counted_this_pass = True
            # print(f"ĐÃ ĐẾM (Từ dưới lên): {class_name} với ID {object_id}. Số lượng mới: {self.vehicle_counts_up[class_name]}") # Debugging

        # Luôn cập nhật trạng thái của đối tượng cho khung hình tiếp theo
        if current_state is not None:
            self.object_crossing_state[object_id] = current_state
        # Nếu đây là đối tượng mới xuất hiện (chưa có trạng thái trước đó), khởi tạo trạng thái của nó
        elif prev_state is None: 
             if current_position_is_above:
                 self.object_crossing_state[object_id] = 'above_line'
             else:
                 self.object_crossing_state[object_id] = 'below_line'

        return counted_this_pass

    def clean_up_states(self, active_object_ids):
        """
        Dọn dẹp trạng thái của các đối tượng không còn được theo dõi trong khung hình hiện tại.
        :param active_object_ids: Set chứa ID của tất cả các đối tượng đang được theo dõi trong khung hình hiện tại.
        """
        objects_to_remove = [
            obj_id for obj_id in self.object_crossing_state
            if obj_id not in active_object_ids
        ]
        for obj_id in objects_to_remove:
            del self.object_crossing_state[obj_id]

    def get_counts(self):
        """
        Trả về dictionary chứa tổng số lượng đếm của từng loại phương tiện cho cả hai chiều.
        Kết quả có dạng {'down': {class: count}, 'up': {class: count}}.
        """
        return {'down': self.vehicle_counts_down, 'up': self.vehicle_counts_up}