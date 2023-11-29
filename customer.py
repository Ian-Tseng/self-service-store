from item import Item

class Customer():
    def __init__(self):
        #紀錄ID
        self.id=None
        #紀錄購買了啥
        self.shopping_cart=[]
        #記錄軌跡資訊
        self.body_xywh=None
        #紀錄候選商品
        self.candidates=[]
        # Position of node of body
        self.l_hand_xy_with_y_model=None
        self.r_hand_xy_with_y_model=None
        self.left_arm_xy=None
        self.right_arm_xy=None
        # Box of customer
        self.box_xywh= None
        self.hands_landmarks= None
        self.in_store=False
        # State of hands
        self.hands_state= {'l_hand':None, 'r_hand':None}
        # State of body
        self.body_state= None
        self.previous_hand_frames_dict= {'l_hand': {}, 'r_hand':{}}
        self.previous_hand_frames_count= None
        # Motion intent
        self.motion_intent= None
        # Coordinate of all nodes of body
        self.coor_all_nodes_body= None
       
        self.name=None
    
    def get_id(self):
        """
        返回顧客ID
        """
        return self.id
    
    def add_item_2_cart(self,item:str):
        """
        說明:
        ---------
        要加入購物車的商品

        參數：
        ----------
        item : Item class object

        返回：
        ----------
        None 
        """
        self.shopping_cart.append(item)
    def remove_item_from_cart(self,item:str):
        """
        說明:
        ---------
        要加入購物車的商品

        參數：
        ----------
        item : Item class object

        返回：
        ----------
        None 
        """
        if item in self.shopping_cart:
            self.shopping_cart.remove(item)

    def add_item2candidate(self,item:Item):
        """
        說明:
        ---------
        要加入候選名單的商品

        參數：
        ----------
        item : Item class object

        返回：
        ----------
        None 
        """
        self.candidates.append(item)

    def get_candidate(self):
        """
        說明:
        ---------
        取得所有候選名單

        參數：
        ----------
        None

        返回：
        ----------
        array of Item class object
        """
        return self.candidates
    
    def get_shopping_cart(self):
        """
        說明:
        ---------
        得到所有購物車內的商品

        參數：
        ----------
        None

        返回：
        ----------
        array of Item class object
        """
        return self.shopping_cart
    
    def remove_item4candidate(self,item):
        """
        說明:
        ---------
        刪除候選名單內的商品

        參數：
        ----------
        item : Item class object

        返回：
        ----------
        None 
        """
        self.candidates.remove(item)
    
    def remove_item4shopping_cart(self,item):
        """
        說明:
        ---------
        刪除購物車內的商品

        參數：
        ----------
        item : Item class object

        返回：
        ----------
        Array of Item class object
        """
        self.shopping_cart.remove(item)