from re import I
from ultralytics import YOLO
import math
import cv2
import time
import inspect
import numpy as np
import os
import torch
import random
import ast
import tensorflow as tf
import mediapipe as mp

from PIL import Image
from customer import Customer
from item import Item
from lidar_detector import LidarDetector
from camera_detector import CameraDetector
from transformers import AutoImageProcessor, DPTForDepthEstimation
from transformers import DPTFeatureExtractor, DPTForSemanticSegmentation
from transformers import BlipProcessor, BlipForConditionalGeneration

#from segment_anything import build_sam, SamPredictor, sam_model_registry
from transformers import pipeline
from itertools import combinations


device=  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (f'Device: {device}')

# Binding

# Note
"""
1. 店內所有商品皆整齊擺放在貨架上，每個商品皆以透明隔板分開，或裝在黑色的小籃子裏，能夠清楚分辨每一樣商品，就連小包裝零食或瓶裝飲品都是如此
2.    辨識不易的鮮食商品，在它的包裝上還會加上專用的識別設計
3. 每個貨架上，皆設有重量感測器，能偵測架上商品的重量，並比對影像辨識的結果
4. 客人與電子帳戶綁定
5. 
"""

# Inference relationship between customers and items
"""
1.  1) If customer's body state is 'forwarding', then start to detect intersection between hand 
        and item.
    2) If intersection between hand is true , then detect the move intent is put or get by 
    
2. Body state: {1: in_forwarding, 2:in_event}

"""


class UMS():
    def __init__(self):
        #預先載入模型
        self.model_items= YOLO('weights/ums-5_segmentation.pt')
      #  self.model_items = YOLO("weights/UMS-5.pt")
       
        self.model_pose=YOLO("weights/yolov8s-pose.pt")
        self.model_items.to(device)
        self.model_pose.to(device)
        #骨架和商品圖
        self.anno_poseImg=None
        self.anno_itemsImg=None
        #模型推理結果
        self.pose_results=None
        self.items_results=None
        #顧客
        self.customers={}
        self.distance_threshold=100 #不知道多少比較好，先假設100
        # Item state dict
        self.items_state_dict= {}
        self.items_name_dict= None
        self.depth_dict= {'shelf_depth':None}
        # Event that handle item state
        # items states: 1. None 2. location_changing 3. holding with customer

        # Dpt model 
        self.feature_extractor =None
        self.dpt_model= None
        #self.set_depth_model()

        # img 2 text model
        self.captioner_processor= None
        self.captioner_model= None
        self.set_captioner_model()

        # Frame Variable
        self.ori_frame= None
        self.last_frame_count= 20

        # For deteting intersection
        self.target_hand_index_list= [5, 6, 9, 10, 13, 14]
       
        # Save data of previous frame of hand variable
        self.previous_hand_frames_maxi= 36

        # For detecting motion
        self.motion_state= ['holding', 'puting']
        self.check_motion_count= 24
        self.number_target_check_previous_frames= 3

        # For detecting hand is holding
        self.detect_hand_holding_model= self.load_tf_model(os.path.join(os.getcwd(), 'detect_holding_model_trans'))
        
        # For detecting arm is forwarding
        self.detect_arm_forwarding_model= self.load_tf_model(os.path.join(os.getcwd(), 'detect_motion_model_trans'))

        # For text record
        self.text_end_count= None
        self.target_text= None
    def load_tf_model(self, dir:str):

        return_value= False
        if os.path.exists(dir):
            return_value = tf.keras.models.load_model(dir)
      
        return return_value

    def get_feature_pose_land_mark(self, coors_list: list)-> np.ndarray:
  
        target_nodes= [[5, 7, 9], [6, 8, 10]]            
        feature_array= []
   
        for nodes in target_nodes:
            coor_list= [coors_list[i] for i in nodes]
            perm_list = combinations(coor_list, 2)
            sub_feature_list= []
            for i in perm_list:
                            
                    dist = [i[1][0] - i[0][0], i[1][1] - i[0][1]]
                          
                    norm = np.sqrt(dist[0] ** 2 + dist[1] ** 2)
                    direction = [dist[0] / norm, dist[1] / norm]
                    bullet_vector = [direction[0] * np.sqrt(2), direction[1] * np.sqrt(2)]
                    sub_feature_list+= bullet_vector
            feature_array.append(sub_feature_list)                               
                  
        return np.array(feature_array)

    def get_feature_hand_land_mark(self, coors_list:list)-> np.ndarray:
        feature_array= None
  
        target_node_muscle= [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
                    
        feature_array= []
        for nodes in target_node_muscle:
                      
                        coor_list= [coors_list[i] for i in nodes]

                        perm_list = combinations(coor_list, 2)
                        sub_feature_list= []
                        for i in perm_list:
                         
                        
                            dist = [i[1][0] - i[0][0], i[1][1] - i[0][1]]
                          
                            norm = [np.sqrt(dist[0] ** 2 + dist[1] ** 2)]
                         
                          
                            sub_feature_list+= norm
                        feature_array.append(sub_feature_list)
                       
                  
        return np.array(feature_array) 


    def model_predict(self, model, predict_data:np.ndarray):
  
        prediction= model.predict(
            predict_data,
            batch_size=None,
            verbose="auto",
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )
        return prediction

    def predict_holding(self, feature_array:np.ndarray):
        tf.keras.backend.clear_session()
        time_steps, features= feature_array.shape
        new_feature_array= feature_array.reshape(1, time_steps, features)
        predict= self.model_predict(self.detect_hand_holding_model, new_feature_array)
        return int(predict.argmax(-1).item())

    
    def predict_arm_forwarding(self, feature_array:np.ndarray):
        tf.keras.backend.clear_session()
        time_steps, features= feature_array.shape
        new_feature_array= feature_array.reshape(1, time_steps, features)
        predict= self.model_predict(self.detect_arm_forwarding_model, new_feature_array)
        return int(predict.argmax(-1).item())

    
    def inference(self,img):
        
        self.h, self.w, self.chn= frame.shape
        
        self.frame_count= frame_count
        self.pose_results=self.model_pose.track(img, conf=0.5, iou=0.5)[0].cpu()
        self.items_results=self.model_items(img)[0].cpu()
        
        self.anno_poseImg=self.pose_results.plot()
        
        self.anno_itemsImg=self.items_results.plot()
        self.ori_frame= img
        self.frame= self.anno_itemsImg

    
      #  self.update_items_location() #Update location of each items
      
        self.record_customer()#推理完要去紀錄顧客資訊
        self.update_candidates()#推理完更新顧客的候選商品
        # Update depth of all static items
       # if not self.depth_dict['shelf_depth'] and  len([i for i in self.items_state_dict if self.items_state_dict[i]['static_location']!= None])>= 2 : # Get depth of shelf
          #  dpt_static_items_list, dpt_all_items_dict= self.update_items_depth(-1)
           
         #   self.depth_dict['shelf_depth']= [np.min(dpt_static_items_list), np.max(dpt_static_items_list)]
         #   print ('depth of shelf', self.depth_dict['shelf_depth'])
       
        
        # Put text if item in customer cart
        for customer_id in  self.customers:
            text= f"Cart {[i[:3] for i in self.customers[customer_id].shopping_cart]}"

            cv2.putText(self.frame, text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # Put motion record text 
        if self.target_text!= None:
            cv2.putText(self.frame, self.target_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            if self.frame_count>= self.text_end_count:
                self.target_text= None
                self.text_end_count= None

        if self.has_people():
            return True
        else:
            return False
        
        

    def predict_depth(self, model, img:np.ndarray):
        image= Image.fromarray(img)
    
        # prepare image for the model
        # inputs = image_processor(images=image, return_tensors="pt")
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # visualize the prediction
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
       
        return  formatted

    def set_depth_model(self):
        #image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
        self.dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        self.dpt_model.to(device)

    def set_captioner_model(self):
        self.captioner_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.captioner_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")



    def get_text_from_img(self, img:np.ndarray):
        text=''
        Image.fromarray(img).show()
        with torch.no_grad():
            inputs= self.captioner_processor( Image.fromarray(img), return_tensors="pt").to("cuda")
            out= self.captioner_model.generate(**inputs)
            text_list=self.captioner_processor.decode(out[0], skip_special_tokens=True)
            for i in text_list:
                text+= i
        return text

      
    def get_depth_of_target_area(self, target_pts:list):
        dpeth_img=self.predict_depth(self.dpt_model, self.ori_frame)
        dpt_img= Image.fromarray(dpeth_img)
      
        depth_val_list= [dpt_img.getpixel(tuple(i)) for  i in target_pts]

        return depth_val_list
    


        

    def update_items_depth(self, target_item_index:int):
        dpeth_img=self.predict_depth(self.dpt_model, self.ori_frame)
        dpt_img= Image.fromarray(dpeth_img)
        dpt_img.show()
      
        dpt_static_items_without_target_item_list= [dpt_img.getpixel(tuple(self.items_state_dict[item_index]['static_location'][:2])) for item_index in self.items_state_dict if self.items_state_dict[item_index]['static_location']!= None and item_index!= target_item_index]
        dpt_all_items_list= []
        for item_index in self.items_state_dict:
            time_list= [self.items_state_dict[item_index]['last_frames'][i]['time'] for i in self.items_state_dict[item_index]['last_frames']]
            max_time_index= time_list.index(np.max(time_list))
            target_index= list(self.items_state_dict[item_index]['last_frames'])[max_time_index]
            coor= self.items_state_dict[item_index]['last_frames'][target_index]['location'][:2]
            dpt_all_items_list.append([item_index, dpt_img.getpixel(tuple(coor))])
        dpt_all_items_dict= dict(dpt_all_items_list)


        return dpt_static_items_without_target_item_list, dpt_all_items_dict
      
    def update_items_location(self):
        if self.items_results!= None:
            self.items_name_dict= self.items_results.names
           
            for index, (item_data, cls) in enumerate(zip(self.items_results.boxes.data, self.items_results.boxes.cls)):#再把圖中每個商品的xywh取出來     print ()
                 
                    label_name= self.items_name_dict[int(cls)] # Get item's name
                    x, y, max_x, max_y= item_data[:4].numpy()
                    w, h= int(max_x-x), int(max_y- y)
                  
                    
                    item_id= int(item_data[-1]) # Record items' id
                  
                    if not item_id in self.items_state_dict:
                        sub_dict=  {'label':label_name, 'last_frames':{},'state': None ,'frame_count':None, 'static_location':None}#self.item_dict.copy()
                        last_frame= int(self.frame_count% self.last_frame_count)
                        
                        sub_dict['last_frames'][last_frame]= {'location': [int(x), int(y), int(max_x), int(max_y), w, h], 'time': time.time()}
                        # Update item static location
                        if last_frame== 0 and self.frame_count!=0:
                            sub_dict= self.update_static_location(sub_dict, last_frame)
                            in_static_location= self.is_in_static_location(sub_dict, last_frame)
                            if in_static_location:
                                sub_dict['static_location']==   [int(x), int(y), int(max_x), int(max_y), w, h] 
                                                                                                                            
                        
                        self.items_state_dict[item_id]= sub_dict
                       
                        continue
                    else:
                    
                        sub_dict= self.items_state_dict[item_id]
                        sub_dict['state']= 'in_static_location'
                        last_frame= int(self.frame_count% self.last_frame_count)
                       

                        if last_frame== 0 and len(sub_dict['last_frames'])== self.last_frame_count:
                            sub_dict= self.update_static_location(sub_dict, last_frame)
                      
                            in_static_location= self.is_in_static_location(sub_dict, last_frame)
                            if not in_static_location:
                                
                                sub_dict['state']= 'location_changing'
                                sub_dict['static_location']= None
                                print (f"Item {item_id}'s state is {sub_dict['state']}.")
                               
                                label= sub_dict['label']
                                
                               # cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), -1)
                               # Image.fromarray(frame).show()
                            else:
                                sub_dict['static_location']==   [int(x), int(y), int(max_x), int(max_y), w, h] 

                            
                        last_frame= int(self.frame_count% self.last_frame_count)

                        sub_dict['last_frames'][last_frame]= {'location': [int(x), int(y), int(max_x), int(max_y), w, h], 'time': time.time()}
                      
                        self.items_state_dict[item_id]= sub_dict
        #    for item_id in self.items_state_dict:
                    
        #        print (f"Item {item_id}'s state is {self.items_state_dict[item_id]['state']}.")


  
    def update_static_location(self, sub_dict:dict, last_frame:int):
        """
        Update static location of items
        """
        
        if sub_dict['static_location']== None:
            time_list= [sub_dict['last_frames'][i]['time'] for i in sub_dict['last_frames']]
            time_index= time_list.index(min(time_list))
            min_time_frame_id= list(sub_dict['last_frames'])[time_index]

            if  int(sub_dict['last_frames'][last_frame]['location'][0]+ (sub_dict['last_frames'][last_frame]['location'][4])/2) in np.arange(
            int(sub_dict['last_frames'][min_time_frame_id]['location'][0]), int(sub_dict['last_frames'][min_time_frame_id]['location'][2]),1, dtype=int) and int(sub_dict['last_frames'][last_frame]['location'][1]) in np.arange(
            int(sub_dict['last_frames'][min_time_frame_id]['location'][1]), int(sub_dict['last_frames'][min_time_frame_id]['location'][3]),1, dtype=int):
                sub_dict['static_location']= sub_dict['last_frames'][last_frame]['location']
    

        else:
            if  int(sub_dict['last_frames'][last_frame]['location'][0]+ (sub_dict['last_frames'][last_frame]['location'][4])/2) in np.arange(
            int(sub_dict['static_location'][0]), int(sub_dict['static_location'][2]),1, dtype=int) and int(sub_dict['last_frames'][last_frame]['location'][1]) in np.arange(
            int(sub_dict['static_location'][1]), int(sub_dict['static_location'][3]),1, dtype=int):
                
                sub_dict['static_location']= sub_dict['last_frames'][last_frame]['location']

     
        return sub_dict

   
    def is_in_static_location(self, sub_dict:dict, last_frame:int):
        """
        Check item is in static location.
        """
        item_in_static_location= False
        if sub_dict['static_location']== None:
            return item_in_static_location
        
        if  int(sub_dict['last_frames'][last_frame]['location'][0]+ 
                (sub_dict['last_frames'][last_frame]['location'][4])/2) in np.arange(
            int(sub_dict['static_location'][0]), int(sub_dict['static_location'][2]),1, dtype=int) and int(
                sub_dict['last_frames'][last_frame]['location'][1]+ 
                (sub_dict['last_frames'][last_frame]['location'][5])/2) in np.arange(
            int(sub_dict['static_location'][1]), int(sub_dict['static_location'][3]),1, dtype=int):
            item_in_static_location= True
       
        return item_in_static_location
        

  

    
        
    def has_people(self): #測試OK
        """
        返回有沒有人在該frame中
        """
        if len(self.pose_results.boxes.data)==0:
            return False
        else:
            return True
        
   

    def record_customer(self):
        if self.has_people():
            """
            有人時，要把self.customers設立，確保每位客人都有被偵測並保存，並且更新，想看看如何商品跟人物綁定code要寫在哪
            """

            customers_boxes= self.pose_results.boxes
            customers_keypoints= self.pose_results.keypoints
            customers_boxes_xywh= customers_boxes.xywh

            for customer_box_xywh, customer_box_data, customers_keypoints_data in zip(customers_boxes_xywh, 
                                                                                        customers_boxes.data,
                                                                                        customers_keypoints.data): #self.pose_results.boxes.data=[x,y,w,h,id,conf,class]
                id=int(customer_box_data[4]) #先判斷出這是誰
                print ('Customer id', id)
                if id in self.customers: #如果該id有在customers中則c直接給予
                    c=self.customers[id]
                else:#如果沒有此ID，則創建新顧客
                    c=Customer()
                    c.id=id
              
                   
               
                c.l_hand_xy_with_y_model= (int(customers_keypoints_data[9].tolist()[0]), int(customers_keypoints_data[9].tolist()[1])) #把左手的xy資訊存入
                c.r_hand_xy_with_y_model= (int(customers_keypoints_data[10].tolist()[0]), int(customers_keypoints_data[10].tolist()[1])) #把右手的xy資訊存入
                c.left_arm_xy=customers_keypoints_data[7].tolist()[:2] #把左手肘的xy資訊存入
                c.right_arm_xy=customers_keypoints_data[8].tolist()[:2] #把右手肘的xy資訊存入


                c.box_xywh=customer_box_data[0:4] #把xywh資訊存入' customer_box_xywh
                c.coor_all_nodes_body= [(int(customers_keypoints_data[i].tolist()[0]), int(customers_keypoints_data[i].tolist()[1])) 
                                        for i in range(len(customers_keypoints_data))]

                
              

             
                # Update body state 
                if self.frame_count% 8== 0:
                    body_state_pred= self.predict_arm_forwarding(self.get_feature_pose_land_mark(c.coor_all_nodes_body))

                    if int(body_state_pred)==0:
                        c.body_state= 'arm_forwarding'
                        if c.hands_landmarks!= None:
                            hands_landmarks_dict= c.hands_landmarks
                            # Get motion intent 
                            for hand_index in ['l_hand', 'r_hand']:
                                if hands_landmarks_dict[hand_index]== None:
                                    continue
                                all_coor_landmark_hand_list= [(int(hands_landmarks_dict[hand_index].landmark[i].x*self.w), int(hands_landmarks_dict[hand_index].landmark[i].y* self.h)) 
                                              for i in range(len(hands_landmarks_dict[hand_index].landmark)) ]
                                hand_state_pred= self.predict_holding(self.get_feature_hand_land_mark(all_coor_landmark_hand_list))
                                if int(hand_state_pred)== 1:
                                    c.motion_intent= 'put'
                                else:
                                    c.motion_intent= 'get'

                            #<------------------------>#

                
                        print (f'Current body state: {c.body_state}. frame_count {self.frame_count}.')
                # If body state is in event check it evey 4 frames 
                if c.body_state== 'arm_forwarding' or c.body_state== 'in_event':
                    if self.frame_count% 4== 0:
                        body_state_pred= self.predict_arm_forwarding(self.get_feature_pose_land_mark(c.coor_all_nodes_body))
                         # Set body state 2 None if it's not present arm forwarding.
                        if int(body_state_pred)==1:
                        #   c.body_state=None

                         
                            if c.hands_landmarks!= None:
                                hands_landmarks_dict= c.hands_landmarks
                               
                                for hand_index in ['l_hand', 'r_hand']:
                                    if hands_landmarks_dict[hand_index]== None:
                                        continue
                                    all_coor_landmark_hand_list= [(int(hands_landmarks_dict[hand_index].landmark[i].x*self.w), int(hands_landmarks_dict[hand_index].landmark[i].y* self.h)) 
                                              for i in range(len(hands_landmarks_dict[hand_index].landmark)) ]
                                    hand_state_pred= self.predict_holding(self.get_feature_hand_land_mark(all_coor_landmark_hand_list))
                                    if int(hand_state_pred)== 0:
                                        c.body_state=None
                                        c.hands_state[hand_index]= None
                       


                
                  

                self.customers[id]=c

                # Update land mark hand
                p_x, p_y, p_w, p_h=int(c.box_xywh[0]), int(c.box_xywh[1]), int(c.box_xywh[2]), int(c.box_xywh[3])
                crop_mask= np.zeros(frame.shape, np.uint8)
                cv2.rectangle(crop_mask, (p_x, p_y), (int(p_x+ p_w), int(p_y+ p_h)), (255, 255, 255), -1)
                bitwie_img= cv2.bitwise_and(frame, crop_mask)                  
                self.get_hand_land_mark(bitwie_img, id)





                
    def update_candidates_0(self):
        """
        先去查看候選商品，如果有後選商品，則把time+1，沒有則-1，如果time=0，則刪除，time超過門檻就加入購物車
        """
        for item_index in self.items_state_dict:
            if self.items_state_dict[item_index]['state']== 'location_changing':
                (self.items_state_dict[item_index])

    def check_previous_frames_to_get_motion_intent(self, previous_frames_list:list, previous_landmarks_hand_list:list):
        continue_holding_state_in_previous_frames= False
        for frame, landmark_hand in zip(previous_frames_list, previous_landmarks_hand_list):
          #  Image.fromarray(frame).show()
           
            predict= self.predict_holding(self.get_feature_hand_land_mark(landmark_hand))
           
            print ('holding' if predict==1 else 'not holding')
         
        
            if predict== 1:
                continue_holding_state_in_previous_frames= True
        #    text= self.get_text_from_img(frame)
        #    print ('This is text:', text)
        #    if 'holding' in text:
        #       continue_holding_state_in_previous_frames= True
        return continue_holding_state_in_previous_frames

    def find_not_intersect_previous_frames(self, previous_frames_list:list, item_contour_list:list):
        is_target_frame= False
        target_frames_list= []
        target_landmark_of_hand_list= []
       
        max_frame_count= 4
        for index, previous_frame in enumerate(previous_frames_list):
            previous_frame_with_contours_list= previous_frame[1]

            xy, x0_ind, y_ind = np.intersect1d(np.array([str(i) for i in previous_frame_with_contours_list]), 
                                  np.array([str(i) for i in item_contour_list]), 
                                  return_indices=True)
           
            if not len(x0_ind):
                
              #  Image.fromarray(previous_frame[0]).show()
                for i in range(max_frame_count):
                    target_frames_list.append(previous_frames_list[index+i][0])
                    target_landmark_of_hand_list.append(previous_frames_list[index+i][2])
                break
           
       
        return target_frames_list, target_landmark_of_hand_list

    
    def update_candidates(self):
        """
        先去查看候選商品，如果有候選商品，則把time+1，沒有則-1，如果time=0，則刪除，time超過門檻就加入購物車
        """
       
        customer=Customer()
        if self.has_people():#沒有人就不用更新
            for customer_id in self.customers:#先把顧客取出來
                    
                    customer= self.customers[customer_id]
                    
                    # Pass if body state is not in forwarding or is in event
                    if customer.body_state== None or customer.body_state== "in_event":
                        print (f'None or in event body state {customer.body_state}. frame_count {self.frame_count}' )
                        continue
                
                    p_x, p_y, p_w, p_h=int(customer.box_xywh[0]), int(customer.box_xywh[1]), int(customer.box_xywh[2]), int(customer.box_xywh[3])
                    
                    crop_mask= np.zeros(frame.shape, np.uint8)
                    cv2.rectangle(crop_mask, (p_x, p_y), (int(p_x+ p_w), int(p_y+ p_h)), (255, 255, 255), -1)
                    bitwie_img= cv2.bitwise_and(frame, crop_mask)
                                        
                    self.get_hand_land_mark(bitwie_img, customer_id)
                    hands_landmarks_dict= customer.hands_landmarks
                    if hands_landmarks_dict== None:
                        continue


                    # Update previous hand frames dict  
                    # FOr initiating infer param
                    hands_state_dict= {'l_hand':None, 'r_hand':None }       
                    if self.customers[customer_id].previous_hand_frames_count== None:
                        self.customers[customer_id].previous_hand_frames_count= 0
                    else:
                        self.customers[customer_id].previous_hand_frames_count+= 1

                    scale_rate= 0.1
                    for hand_index in ['l_hand', 'r_hand']:
                        if hands_landmarks_dict[hand_index]== None:
                            continue
                        
                        l_min_x, l_min_y, l_max_x, l_max_y= int(customer.l_hand_xy_with_y_model[0]-customer.box_xywh[2]*scale_rate),int(customer.l_hand_xy_with_y_model[1]-customer.box_xywh[2]*scale_rate), int(customer.l_hand_xy_with_y_model[0]+customer.box_xywh[2]*scale_rate), int(customer.l_hand_xy_with_y_model[1]+ customer.box_xywh[2]*scale_rate)
                        r_min_x, r_min_y, r_max_x, r_max_y= int(customer.r_hand_xy_with_y_model[0]-customer.box_xywh[2]*scale_rate),int(customer.r_hand_xy_with_y_model[1]-customer.box_xywh[2]*scale_rate), int(customer.r_hand_xy_with_y_model[0]+customer.box_xywh[2]*scale_rate), int(customer.r_hand_xy_with_y_model[1]+ customer.box_xywh[2]*scale_rate)
                        crop_mask= np.zeros(frame.shape, np.uint8)
                        if hand_index== 'l_hand':
                            box_hand=self.ori_frame[l_min_y: l_max_y,l_min_x: l_max_x]
                            cv2.rectangle(crop_mask, (l_min_x, l_min_y), (l_max_x, l_max_y), (255, 255, 255), -1)

                        else:
                            box_hand=self.ori_frame[r_min_y: r_max_y,r_min_x: r_max_x]
                            cv2.rectangle(crop_mask, (r_min_x, r_min_y), (r_max_x, r_max_y), (255, 255, 255), -1)
                        
                        bitwie_img= cv2.bitwise_and(frame, crop_mask)  
                     
                        contours= np.where(bitwie_img!= (0, 0, 0))
                        contours_hand_list= [(x, y) for y, x in zip(contours[0], contours[1])]
                        all_coor_landmark_hand_list= [(int(hands_landmarks_dict[hand_index].landmark[i].x*self.w), int(hands_landmarks_dict[hand_index].landmark[i].y* self.h)) 
                                              for i in range(len(hands_landmarks_dict[hand_index].landmark)) ]
                
                        # Save [box of hand, contours of box, landmark of hand] into previous hand frame dict
                        self.customers[customer_id].previous_hand_frames_dict[hand_index][customer.previous_hand_frames_count]= [box_hand, contours_hand_list, all_coor_landmark_hand_list]
                        # Pop oldest frame while previous frame count exceed a given number
                        if len(self.customers[customer_id].previous_hand_frames_dict[hand_index])> self.previous_hand_frames_maxi:
                               [self.customers[customer_id].previous_hand_frames_dict[hand_index].pop(i0) 
                                for i0 in list(self.customers[customer_id].previous_hand_frames_dict[hand_index])[:1]
                              ]
                               
                        # Init infer if hand is in holding state
                        hands_state_dict[hand_index]= self.predict_holding(self.get_feature_hand_land_mark(all_coor_landmark_hand_list))
                    # Pass if do not exist state of hand is holding
                    print ('hands_state_dict', hands_state_dict)
                    
                    if  not len([hands_state_dict[i] for i in hands_state_dict if hands_state_dict[i]==1]):
                        continue

                    #==========================#
 
                  #  print ('item class attr',inspect.getmembers(self.items_results, lambda a:not(inspect.isroutine(a))))

   
                  #  for hand_xywh in customer.hand_xywhs:#先把手的xywh取出來
              
                    self.items_name_dict= self.items_results.names
                    inter_dict= {'r_hand':{}, 'l_hand': {}}
                 
                    for item_index, item_mask_coor in enumerate(
                        self.items_results.masks.xy):#再把圖中每個商品的xywh取出來)
                   
                        cls= self.items_results.boxes.cls[item_index]
                        contour_array= np.array([[(int(i[0]), int(i[1])) for i in item_mask_coor]])
                     #   x, y, w, h= int(item_xywh[0]- item_xywh[2]*0.5), int(item_xywh[1]- item_xywh[3]*0.5), int(item_xywh[2]), int(item_xywh[3])
                        mask= np.zeros(frame.shape, np.uint8)
                        cv2.drawContours(mask,contour_array, -1,(255, 255, 255), -1)
                        contours= np.where(mask== (255, 255, 255))                       
                        contours_list= [(x, y) for y, x in zip(contours[0], contours[1])]
                       
                        for hand_index in hands_landmarks_dict:
                            if hands_state_dict[hand_index]== None:
                                continue
                          
                            if hands_landmarks_dict[hand_index]== None:
                                continue


                            specific_coor_landmark_hand_list= [(int(hands_landmarks_dict[hand_index].landmark[i].x*self.w), int(hands_landmarks_dict[hand_index].landmark[i].y* self.h)) 
                                              for i in range(len(hands_landmarks_dict[hand_index].landmark)) 
                                              if i in self.target_hand_index_list]
                            all_coor_landmark_hand_list= [(int(hands_landmarks_dict[hand_index].landmark[i].x*self.w), int(hands_landmarks_dict[hand_index].landmark[i].y* self.h)) 
                                              for i in range(len(hands_landmarks_dict[hand_index].landmark)) ]
                            
                            xy, x_ind, y_ind = np.intersect1d(np.array([str(i) for i in specific_coor_landmark_hand_list]), np.array([str(i) for i in contours_list]), return_indices=True)
                            inter_list= [contours_list[i] for i in y_ind] # find the intersection between hand's land mark mask and item's mask
                         
                            contours_item_and_hand_list= contours_list+ specific_coor_landmark_hand_list
                           

                            # Update inter dict save [intersection coordinate between hand and item, union coordinate between hand and item, coordinate of hand]
                            if len(inter_list)> 0:
                                                       
                                inter_dict[hand_index][item_index]= [inter_list, 
                                                                     contours_item_and_hand_list,
                                                                     all_coor_landmark_hand_list]
                
                    # Save mask of item result

                    #Image.fromarray(frame).show()
                    #Image.fromarray(frame).save(os.path.join(os.getcwd(), 'items_mask.jpg'))
                   
                    target_items_inter_dict= {}
                    for i, hand_index in enumerate(inter_dict):
                        if not len(inter_dict[hand_index]):
                            continue
                        color= (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                        inter_list= [inter_dict[hand_index][i][0] for i in inter_dict[hand_index]]
                        contours_item_and_hand_list=  [inter_dict[hand_index][i][1] for i in inter_dict[hand_index]]
                        landmark_hand_list= [inter_dict[hand_index][i][2] for i in inter_dict[hand_index]]
                        # ====================
                        if not len(inter_list):
                        
                            continue
                        # Do not initiate inference if lift time is not none and up.
                     #   elif len(inter_list)!= 0:
                     #       if self.customers[customer_id].lift_time_dict[hand_index]!= None:
                     #           if self.customers[customer_id].lift_time_dict[hand_index]< self.customers[customer_id].previous_hand_frames_count:
                     #               continue
                        # ====================


                        # Get  index of target item
                        target_item_index= list(inter_dict[hand_index])[inter_list.index(max(inter_list))]
                        cls= int(self.items_results.boxes.cls[target_item_index])
                    
                        print ('inter true', hand_index, self.items_name_dict[cls])
                      
                 

                        # Update intersection between hand and item dict
                        if not customer_id in target_items_inter_dict:
                            target_items_inter_dict[customer_id]= {hand_index:self.items_name_dict[cls] }
                        else:
                            target_items_inter_dict[customer_id][hand_index]= self.items_name_dict[cls] 
                        
                   
                        # Predict is holding 
                        if  not len(contours_item_and_hand_list):
                            continue
                        
                    #    predict= self.predict_holding(self.get_feature_hand_land_mark(landmark_hand_list[0]))
                  
                     #   print ('Is holding' if int(predict)==1 else 'Is not holding')

                    #    if not 'holding' in text:
                     
                        target_previous_frames_list, target_previous_landmarks_hand_list= self.find_not_intersect_previous_frames([self.customers[customer_id].previous_hand_frames_dict[hand_index][i] for i 
                                                           in list(self.customers[customer_id].previous_hand_frames_dict[hand_index])],
                                                          contours_item_and_hand_list[0])
                        
                        is_continue_holding_previously=  self.check_previous_frames_to_get_motion_intent(target_previous_frames_list, 
                                                                                           target_previous_landmarks_hand_list)
                        
                        if is_continue_holding_previously:
                            target_motion='put'
                        else:
                            target_motion= 'get'
                    #    target_motion= customer.motion_intent
                      
                        # Pass if the intersect object do not in cart
                        print ('Target_motion:', target_motion)
                        text= f"Motion: {target_motion}"

                        if self.text_end_count== None:
                            self.text_end_count= self.frame_count+30
                            self.target_text= text
                        if self.frame_count>= self.text_end_count:
                            self.target_text= None
                            self.text_end_count= None
                        if target_motion== 'put' and not self.items_name_dict[cls] in  self.customers[customer_id].shopping_cart:
                            continue
                        

                        # Update state of items
                        self.update_item_cart( customer_id, hand_index, target_item_index
                                               , self.items_name_dict[cls], target_motion
                                                )
                      #  text= f"Cust ID: {customer_id}'s motion {target_motion} sc {self.customers[customer_id].shopping_cart}"
                        self.frame_count+= 1
                       
                   
                    #  if self.is_near(hand_xywh,thing_xywh): 
                        #    thing=Item() 
                 

    def update_item_cart(self,  customer_id:int, hand_index:str, item_id:int, item_name:str, motion:str):
        # Item existing in shelf
        
       # if item_id in self.items_state_dict:
            # Check item align the z index of hand 
        #    print (f'Item {item_id} existing in shelf')
            
            
            if  motion== 'get':
                self.customers[customer_id].hands_state[hand_index]= 'holding'
                self.customers[customer_id].body_state= 'in_event'
                self.customers[customer_id].add_item_2_cart(item_name)
            elif motion== 'put':
                self.customers[customer_id].hands_state[hand_index]= 'puting'
                self.customers[customer_id].body_state= 'in_evenet'
                self.customers[customer_id].remove_item_from_cart(item_name)
               
           # elif self.customers[customer_id].hands_state[hand_index]== 'holding':
           #     self.hands_state(customer_id, hand_index)
            
            print (f'Customer {customer_id} cart items : {self.customers[customer_id].shopping_cart}')
              
        
        # Item do not exist in shelf
       # else:
       #     print (f'Item {item_id} do not exist in shelf')
            
    def check_holding_state(self, customer_id:int, hand_index:str):
        print ('Check holding state.')
        holding= False
        hands_landmarks_dict= self.customers[customer_id].hands_landmarks
        coord_hand_list= [(int(hands_landmarks_dict[hand_index].landmark[i].x*self.w), int(hands_landmarks_dict[hand_index].landmark[i].y* self.h)) 
                          for i in range(len(hands_landmarks_dict[hand_index].landmark))]
        
        target_area_array= np.array(coord_hand_list)
                           
        x,y,w,h = cv2.boundingRect(target_area_array)
        roi=self.ori_frame[y:y+h,x:x+w]
      #  Image.fromarray(roi).show()
        text= self.get_text_from_img(roi)
        print (f'Image to text for holding state: {text}')
        if 'holding' in text:
            holding= True


    
    def is_near(self,hand_xywh,thing_xywh):
        #計算兩個xywh的距離
        distance=math.sqrt(math.pow(hand_xywh[0] - thing_xywh[0], 2) + math.pow(hand_xywh[1] - thing_xywh[1], 2))
        return self.distance_threshold >distance
    

    
    def get_hand_land_mark(self, image:np.ndarray, customer_id:int):
        h, w, chn= image.shape
       # mp_drawing = mp.solutions.drawing_utils
       # mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands
        num_seconds= 0.01

        left_hand_xy= (int(self.customers[customer_id].l_hand_xy_with_y_model[0]),int(self.customers[customer_id].l_hand_xy_with_y_model[1]))
        right_hand_xy= (int(self.customers[customer_id].r_hand_xy_with_y_model[0]),int(self.customers[customer_id].r_hand_xy_with_y_model[1]))
        hands_coor_list= [left_hand_xy, right_hand_xy]
        hands_coor_dict= {'l_hand': left_hand_xy, 'r_hand': right_hand_xy}
        
        scale_rate= 0.3
       

     
        while True:
            hands_landmarks_dict= {'l_hand': None, 'r_hand': None }
            with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5) as hands:
            #coor_hand= hands_coor_dict[hand_index]      
            #mask= np.ones(image.shape, np.uint8)* 255
            #cv2.rectangle(mask, (int(coor_hand[0]), int(coor_hand[1])), 
            #              (int(coor_hand[0]+ w*scale_rate), int(coor_hand[1]- w*scale_rate)), (0 ,0 ,0), -1)
            #image=  cv2.bitwise_or(image.copy(), mask)
            #bitwise_img= image.copy()
           # Image.fromarray(bitwise_img).show()
 
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                h, w, chn= image.shape

                if results.multi_hand_landmarks:
                    hand_landmarks= results.multi_hand_landmarks[0] 
                    coord_list= [[(int(hand_landmarks.landmark[i].x*w), int(hand_landmarks.landmark[i].y* h)) for i in range(len(hand_landmarks.landmark))]]

                    wrist_rel_x= int(hand_landmarks.landmark[0].x* w)
                    wrist_rel_y= int(hand_landmarks.landmark[0].y* h)
                    dist_list=[math.dist((wrist_rel_x, wrist_rel_y), i)  for i in hands_coor_list]
                    target_hand_index= dist_list.index(min(dist_list))

                    if target_hand_index== 0:
                      
                        hands_landmarks_dict['l_hand']= hand_landmarks
                       # hands_landmarks_dict['l_hand']['coord_list']= coord_list
                        
                        mask= np.zeros(image.shape, np.uint8)
                        b_x,b_y,b_w,b_h = cv2.boundingRect( np.array(coord_list[0]))
                        cv2.rectangle(mask, (b_x, b_y), (b_x+ b_w, b_y+ b_h), (255, 255, 255), -1)
                        image= bitwise_img= cv2.bitwise_or(image, mask)

                        self.customers[customer_id].hands_landmarks= hands_landmarks_dict # Save hand land mark
                      
                        continue
                        
                    else:
                        hands_landmarks_dict['r_hand']= hand_landmarks
                       
                     #   hands_landmarks_dict['r_hand']['coord_list']= coord_list

                        mask= np.zeros(image.shape, np.uint8)
                        b_x,b_y,b_w,b_h = cv2.boundingRect( np.array(coord_list[0]))
                        cv2.rectangle(mask, (b_x, b_y), (b_x+ b_w, b_y+ b_h), (255, 255, 255), -1)
                        image= bitwise_img= cv2.bitwise_or(image, mask)

                        self.customers[customer_id].hands_landmarks= hands_landmarks_dict # Save hand land mark

                        continue

                  
                 
                else:
                   
                    break
              
    

if __name__=="__main__":
    frame_count= 0
    ums=UMS()
    test_img= False

    if test_img:
    # Image test
        img_file_dir= 'resource/test_img.jpg'
        frame= cv2.imread(img_file_dir)
    
        val= ums.inference(frame)
        Image.fromarray(ums.anno_poseImg).show()
        Image.fromarray(ums.anno_itemsImg).show()
    else:

        # video test
        test_file="resource/test.mov"
        cap=cv2.VideoCapture(test_file)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (960,540))
   
        while cap.isOpened():
            ret,frame=cap.read()
            if ret:
                frame=cv2.resize(frame,(960,540))
                val= ums.inference(frame)
                frame_count+= 1
              #  if not val:
              #      continue
               # cv2.imshow("pose",ums.anno_poseImg)
                out.write(ums.frame)
                cv2.imshow("things",ums.frame)
                cv2.waitKey(1)
            
            else:
                break
       
        out.release()
        cv2.destroyAllWindows()
