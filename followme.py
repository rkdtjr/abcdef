#!/usr/bin/env python3

# 경고 메시지 비활성화
import os
import warnings
os.environ['NNPACK_DISABLE'] = '1'
warnings.filterwarnings('ignore')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO


class PersonFollowerNode(Node):
    def __init__(self):
        super().__init__('person_follower_node')

        # ================= 파라미터 =================
        self.declare_parameter('image_topic', '/image_raw/compressed')
        self.declare_parameter('cmd_vel_topic', '/keyboard/cmd_vel')
        self.declare_parameter('process_scale', 0.5)
        self.declare_parameter('conf_threshold', 0.3)

        self.image_topic = self.get_parameter('image_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.process_scale = self.get_parameter('process_scale').value
        self.conf_threshold = self.get_parameter('conf_threshold').value

        # ================= YOLOv8 모델 =================
        self.get_logger().info('Loading YOLO model...')
        self.model = YOLO('yolov8n.pt')
        self.get_logger().info('YOLO model loaded')

        # ================= ROS2 통신 =================
        self.bridge = CvBridge()
        
        # 이미지 구독
        self.image_sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            10
        )
        
        # cmd_vel 발행
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            10
        )

        # ================= 추적 상태 =================
        self.current_frame = None
        self.current_boxes = None
        self.target_feature = None
        self.tracking = False

        # 히스테리시스 (떨림 방지)
        self.last_linear_cmd = 0.0

        # ================= OpenCV 윈도우 =================
        cv2.namedWindow('Person Follower', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Person Follower', self.mouse_callback)

        self.get_logger().info(f'Subscribed to: {self.image_topic}')
        self.get_logger().info(f'Publishing to: {self.cmd_vel_topic}')
        self.get_logger().info('Person Follower Started!')
        self.get_logger().info('Usage:')
        self.get_logger().info('  - Click on a person to start tracking')
        self.get_logger().info('  - Press "q" to quit')

    # =================================================
    # 특징 추출 (컬러 히스토그램)
    # =================================================
    def extract_feature(self, crop):
        """이미지 crop에서 컬러 히스토그램 특징 추출"""
        if crop is None or crop.size == 0:
            return None
        
        hist = []
        for channel in range(3):  # BGR 채널
            h = cv2.calcHist([crop], [channel], None, [16], [0, 256])
            hist.append(h)
        
        feature = np.concatenate(hist).flatten()
        feature = feature / (np.sum(feature) + 1e-6)  # 정규화
        
        return feature

    def compute_similarity(self, feat1, feat2):
        """두 특징 벡터 간의 코사인 유사도 계산"""
        if feat1 is None or feat2 is None:
            return 0.0
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-6)

    # =================================================
    # 시각화: size_ratio 바
    # =================================================
    def draw_size_ratio_bar(self, img, size_ratio):
        """화면에 size_ratio 시각화 바 그리기"""
        x, y = 20, 60
        w, h = 220, 20

        # 배경
        cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
        
        # 현재 값
        cv2.rectangle(img, (x, y),
                      (x + int(w * min(size_ratio, 1.0)), y + h),
                      (0, 255, 255), -1)

        # 기준선 (녹색: 적정 거리, 빨간색: 너무 가까움)
        cv2.line(img, (x + int(w * 0.7), y),
                 (x + int(w * 0.7), y + h), (0, 255, 0), 2)
        cv2.line(img, (x + int(w * 0.8), y),
                 (x + int(w * 0.8), y + h), (0, 0, 255), 2)

        cv2.putText(img, f"Distance: {size_ratio:.2f}",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

    # =================================================
    # 시각화: offset_x 바
    # =================================================
    def draw_offset_bar(self, img, offset_x):
        """화면에 offset_x 시각화 바 그리기"""
        x, y = 20, 110
        w, h = 220, 20

        # 배경
        cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)

        # 중심선 기준 offset 표시
        center = x + w // 2
        offset_px = int(offset_x * (w // 2))
        cv2.rectangle(img, (center, y),
                      (center + offset_px, y + h),
                      (255, 180, 0), -1)

        # 기준선
        cv2.line(img, (center, y), (center, y + h), (255, 255, 255), 2)
        cv2.line(img, (x + int(w * 0.35), y),
                 (x + int(w * 0.35), y + h), (0, 255, 0), 2)
        cv2.line(img, (x + int(w * 0.65), y),
                 (x + int(w * 0.65), y + h), (0, 255, 0), 2)

        cv2.putText(img, f"Offset: {offset_x:.2f}",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

    # =================================================
    # 제어 명령 계산 (히스테리시스 적용)
    # =================================================
    def compute_control_command(self, bbox, frame_width, frame_height):
        """타겟 위치에 따라 로봇 제어 명령 계산"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        bbox_height = y2 - y1

        # 화면 중심 기준 상대 위치 (-1.0 ~ 1.0)
        offset_x = (center_x - frame_width / 2) / (frame_width / 2)
        
        # 화면 대비 타겟 크기 비율 (0.0 ~ 1.0)
        size_ratio = bbox_height / frame_height

        cmd = Twist()

        sensitivity = 0.1
        sensitivity_margin = 0.3
        # ========== 좌/우 회전 ==========
        if offset_x < -sensitivity:
            cmd.angular.z = 1.5   # 왼쪽 회전
        elif offset_x > sensitivity:
            cmd.angular.z = -1.5  # 오른쪽 회전
        else:
            cmd.angular.z = 0.0   # 중앙 정렬
        if abs(offset_x) > sensitivity_margin:
            self.last_linear_cmd = cmd.linear.x

            return cmd, size_ratio, offset_x

        stop_threshold = 0.80  # 멈춤 임계값
        stop_margin = 0.15
        
        max_distance_ratio = 0.3

        #1~4까지 되려면  0.3 차이에서 4가 되도록
        threshold_diff_ratio = max(min((stop_threshold-size_ratio)/max_distance_ratio * 4,4.0),1.0) # 멈춤 임계값과 현재 비율 비례
        print(threshold_diff_ratio)
        velocity_base = 0.5
        # ========== 전진/후진 (히스테리시스) ==========
        # 이전 명령이 후진(-0.5)이었다면
        if self.last_linear_cmd < 0:
            if size_ratio < stop_threshold:  # 충분히 멀어졌으면
                cmd.linear.x = 0.0  # 정지
            else:
                cmd.linear.x = -velocity_base * threshold_diff_ratio   # 계속 후진
        
        # 이전 명령이 전진(0.5)이었다면
        elif self.last_linear_cmd > 0:
            if size_ratio > stop_threshold:  # 충분히 가까워졌으면
                cmd.linear.x = 0.0  # 정지
            else:
                cmd.linear.x = velocity_base * threshold_diff_ratio   # 계속 전진
        
        # 이전 명령이 정지(0.0)였다면
        else:
            if size_ratio > stop_threshold + stop_margin:    # 너무 가까움
                cmd.linear.x = -velocity_base * threshold_diff_ratio  # 후진
            elif size_ratio < stop_threshold - stop_margin:  # 너무 멀음
                cmd.linear.x = velocity_base * threshold_diff_ratio   # 전진
            else:
                cmd.linear.x = 0.0   # 적절한 거리, 정지

        # 현재 명령 저장 (다음 프레임에서 사용)
        self.last_linear_cmd = cmd.linear.x

        return cmd, size_ratio, offset_x

    # =================================================
    # 마우스 클릭 = 타겟 학습
    # =================================================
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 클릭으로 타겟 선택"""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        
        if self.current_frame is None or self.current_boxes is None:
            return

        # 클릭 좌표를 축소된 이미지 좌표로 변환
        scaled_x = int(x * self.process_scale)
        scaled_y = int(y * self.process_scale)

        # 클릭한 위치에 있는 박스 찾기
        for box in self.current_boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
            
            if bx1 <= scaled_x <= bx2 and by1 <= scaled_y <= by2:
                # 원본 이미지에서 해당 영역 추출
                ox1 = int(bx1 / self.process_scale)
                oy1 = int(by1 / self.process_scale)
                ox2 = int(bx2 / self.process_scale)
                oy2 = int(by2 / self.process_scale)

                crop = self.current_frame[oy1:oy2, ox1:ox2]
                
                # 특징 학습
                self.target_feature = self.extract_feature(crop)
                self.tracking = True
                self.last_linear_cmd = 0.0  # 히스테리시스 초기화
                
                self.get_logger().info('✓ Target learned! Tracking started.')
                return

    # =================================================
    # 메인 이미지 콜백
    # =================================================
    def image_callback(self, msg):
        try:
            # CompressedImage → OpenCV
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            self.current_frame = frame.copy()
            display = frame.copy()


            # 처리용 이미지 축소
            small = cv2.resize(frame, None,
                             fx=self.process_scale,
                             fy=self.process_scale)
            
            # cv2.convertTo(small, -1, 1.5, 50)
            # YOLOv8 사람 탐지
            results = self.model(
                small,
                conf=self.conf_threshold,
                classes=[0],  # person only
                verbose=False
            )

            best_bbox = None
            best_score = 0.0

            # 탐지된 사람들 처리
            if results[0].boxes is not None:
                self.current_boxes = results[0].boxes

                for box in self.current_boxes:
                    # 축소된 좌표를 원본 크기로 변환
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].cpu().numpy() / self.process_scale
                    )

                    # 모든 사람 박스 그리기 (주황색)
                    cv2.rectangle(display, (x1, y1), (x2, y2),
                                (0, 180, 255), 2)

                    # 추적 모드: 타겟과 가장 유사한 사람 찾기
                    if self.tracking:
                        crop = frame[y1:y2, x1:x2]
                        feat = self.extract_feature(crop)
                        score = self.compute_similarity(self.target_feature, feat)

                        if score > best_score:
                            best_score = score
                            best_bbox = (x1, y1, x2, y2)

            # 추적 중이고 타겟을 찾았으면
            if self.tracking and best_bbox is not None:
                # 제어 명령 계산
                cmd, size_ratio, offset_x = self.compute_control_command(
                    best_bbox,
                    frame.shape[1],
                    frame.shape[0]
                )
                
                # cmd_vel 발행
                self.cmd_vel_pub.publish(cmd)

                # 타겟 박스 강조 (녹색)
                x1, y1, x2, y2 = best_bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 4)
                
                # 타겟 레이블
                label = f'TARGET (score: {best_score:.2f})'
                cv2.putText(display, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 중심점
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(display, (cx, cy), 8, (0, 0, 255), -1)

                # 시각화 바 그리기
                self.draw_size_ratio_bar(display, size_ratio)
                self.draw_offset_bar(display, offset_x)

                # 제어 명령 표시
                cmd_text = f'CMD: linear={cmd.linear.x:.1f}, angular={cmd.angular.z:.1f}'
                cv2.putText(display, cmd_text, (20, 160),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            else:
                # 타겟을 잃었거나 추적 안 함 → 정지
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                self.last_linear_cmd = 0.0

                if self.tracking:
                    cv2.putText(display, 'TARGET LOST!', (20, 200),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # 상태 표시
            status = 'TRACKING' if self.tracking else 'CLICK PERSON TO START'
            status_color = (0, 255, 0) if self.tracking else (0, 165, 255)
            cv2.putText(display, status, (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # 화면 표시
            cv2.imshow('Person Follower', display)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Quit requested')
                rclpy.shutdown()
            elif key == ord('r'):
                self.tracking = False
                self.target_feature = None
                self.last_linear_cmd = 0.0
                self.cmd_vel_pub.publish(Twist())
                self.get_logger().info('Tracking reset')

        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {str(e)}')

    def destroy_node(self):
        """노드 종료 시 정지 명령 발행"""
        self.cmd_vel_pub.publish(Twist())
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PersonFollowerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
