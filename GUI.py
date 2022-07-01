import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from active_contours_v1 import act_contours
from Grabcut import Grabcut
import cv2
from lesion_contours_detect import lesion_contours

'''
class ActiveContoursApp(tk.Tk):
    def __init__(self, pad=5):
        super().__init__()

        # im.show()
        # print(im)
        self.filename = '腎臟.png'
        im = Image.open(self.filename)
        width, height = im.size
        self.imTK = ImageTk.PhotoImage(im)
        self.title('active contours sample')
        align_mode = 'nswe'
        self.div1 = tk.Canvas(self, width=width, height=height, bg='blue')
        self.div2 = tk.Frame(self, width=100, height=height // 2, bg='orange')
        self.div3 = tk.Frame(self, width=100, height=height // 2, bg='green')
        self.div1.grid(row=0, column=0, padx=pad, pady=pad, rowspan=2, sticky=align_mode)
        self.div2.grid(column=1, row=0, padx=pad, pady=pad, sticky=align_mode)
        self.div3.grid(column=1, row=1, padx=pad, pady=pad, sticky=align_mode)
        # self.define_layout(self, rows=2, cols=2)
        # self.define_layout([div1, div2, div3])
        # self.div1.pack()
        self.div1.create_image(0, 0, anchor='nw', image=self.imTK)

        # img.grid(row=0, column=0, padx=pad, pady=pad, rowspan=2, sticky=align_mode)
        # self.div1.bind("<B1-Motion>", self.print_mouse_status)
        self.div1.bind("<Button>", self.print_mouse_click)
        self.state = 0
        self.coor_list = []
    def define_layout(self, obj, rows=1, cols=1):
        def method(trg, row, col):
            for r in range(row):
                trg.rowconfigure(r, weight=1)
            for c in range(col):
                trg.columnconfigure(c, weight=1)
        if type(obj) == list:
            [method(trg, rows, cols) for trg in obj]
        else:
            trg = obj
            method(trg, rows, cols)

    def print_mouse_status(self, event):
        print("滑鼠狀態", event.type)
        print("滑鼠位置", event.x, event.y)

    def print_mouse_click(self, event):

        key_dict = {1: '左', 2: '中', 3: '右'}
        print(event.type, "單擊了滑鼠{}鍵".format(key_dict[event.num]))
        print("滑鼠位置", event.x, event.y)
        self.div1.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill='red')
        if self.state == 0:
            self.coor_list.append([event.x, event.y])
            self.state = 1
            # self.div1.create_oval(event.x - 2, event.y - 2, event.x + 2, event.y + 2, fill='red')
        elif self.state == 1:
            self.coor_list.append([event.x, event.y])
            act_contours(self.filename, self.coor_list, 20)
            self.coor_list = []
            # self.div1.create_oval(event.x - 2, event.y - 2, event.x + 2, event.y + 2, fill='blue')
'''

'''
class GrabcutApp(tk.Tk):
    def __init__(self, pad=5):
        super().__init__()

        # im.show()
        # print(im)
        self.filename = 'img.jpg'
        im = Image.open(self.filename)
        width, height = im.size
        self.imTK = ImageTk.PhotoImage(im)
        self.title('active contours sample')
        align_mode = 'nswe'
        self.div1 = tk.Canvas(self, width=width, height=height, bg='blue')
        self.div2 = tk.Frame(self, width=width, height=100, bg='green')

        self.div1.grid(row=0, column=0, padx=pad, pady=pad, sticky=align_mode)
        self.div2.grid(row=1, column=0, padx=pad, pady=pad, sticky=align_mode)


        self.div1.create_image(0, 0, anchor='nw', image=self.imTK)
        # img.grid(row=0, column=0, padx=pad, pady=pad, rowspan=2, sticky=align_mode)
        # self.div1.bind("<B1-Motion>", self.print_mouse_status)
        self.div1.bind("<Button>", self.print_mouse_click)
        self.state = 0
        self.coor_list = []

    def define_layout(self, obj, rows=1, cols=1):
        def method(trg, row, col):
            for r in range(row):
                trg.rowconfigure(r, weight=1)
            for c in range(col):
                trg.columnconfigure(c, weight=1)
        if type(obj) == list:
            [method(trg, rows, cols) for trg in obj]
        else:
            trg = obj
            method(trg, rows, cols)

    def print_mouse_status(self, event):
        print("滑鼠狀態", event.type)
        print("滑鼠位置", event.x, event.y)

    def print_mouse_click(self, event):

        key_dict = {1: '左', 2: '中', 3: '右'}
        print(event.type, "單擊了滑鼠{}鍵".format(key_dict[event.num]))
        print("滑鼠位置", event.x, event.y)
        if self.state == 0:
            self.coor_list.append([event.x, event.y])
            self.state = 1
            self.div1.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, fill='red')
        elif self.state == 1:
            self.coor_list.append([event.x, event.y])
            # act_contours(self.filename, self.coor_list, 20)
            Grabcut(self.filename, self.coor_list)
            self.coor_list = []
            self.state = 0
            self.div1.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, fill='blue')
'''

class AnnoToolsApp(tk.Tk):

    def __init__(self, pad=5):
        super().__init__()
        align_mode = 'nswe'
        self.div1 = tk.Canvas(self, width=200, height=200, bg='blue')
        self.div2 = tk.Frame(self, width=200, height=200, bg='green')
        self.div1.grid(row=0, column=0, padx=pad, pady=pad, sticky=align_mode)
        self.B1 = tk.Button(self, text='choose image', fg='black', command=self.FileOpen) #choose image
        self.B1.grid(row=1, column=0)
        self.div1.bind('<Button-1>', self.draw_contours)

    def FileOpen(self):
        fname = filedialog.askopenfilename(title='choose', filetypes=[("jpeg files", "*.jpg"), ("png files", "*.png")])
        im = cv2.imread(fname)
        self.im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        self.contours_dropped, self.hierarchy_dropped, self.mapping_id = lesion_contours(im)
        h, w, l = im.shape
        print('im.shape', im.shape)
        im_PIL = Image.fromarray(self.im_rgb)
        self.imTK = ImageTk.PhotoImage(image=im_PIL)
        self.div1.config(width=w, height=h)
        self.div1.update()
        self.canvas_img = self.div1.create_image(0, 0, anchor='nw', image=self.imTK)

    def draw_contours(self, event):
        point = (event.x, event.y)
        print('len(self.contours_dropped)', len(self.contours_dropped))
        for i, cnt in enumerate(self.contours_dropped):
            dst = cv2.pointPolygonTest(cnt, point, True)
            if dst >= 0 and self.hierarchy_dropped[0, i, 3] == -1:
                # parent = self.mapping_id[i]
                # print('this is parent', parent, self.hierarchy_dropped[0, i, :], cv2.contourArea(cnt), len(cnt))
                # print('type(cnt)', type(cnt))
                pick_cnt = cnt
                break
        im_rgb_2 = cv2.drawContours(self.im_rgb, [pick_cnt], -1, (255, 0, 0), 1)
        # self.div1.configure(image=im_rgb_2)
        self.img2 = ImageTk.PhotoImage(image=Image.fromarray(im_rgb_2))
        self.div1.itemconfig(self.canvas_img, image=self.img2)
        self.div1.imTK = im_rgb_2

if __name__ =='__main__':
    # app = ActiveContoursApp()
    # app = GrabcutApp(pad=0)
    app = AnnoToolsApp()
    app.mainloop()

