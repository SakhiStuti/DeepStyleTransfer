from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
from torchvision.utils import save_image
from inference import stylize


class main:
    def __init__(self, master):
        self.master = master
        self.frame1 = Frame(self.master)
        self.b_content = Button(self.frame1, text="Import  Content  Image", height=2, width=36, command = self.browse_content)
        self.c_content = Canvas(self.frame1, width=256, height=256, bg='white')
        self.b_style = Button(self.frame1, text="Import  Style  Image", height=2, width=36, command = self.browse_style)
        self.c_style = Canvas(self.frame1, width=256, height=256, bg='white')
        
        self.frame2 = Frame(self.master)
        self.b_output = Button(self.frame2, text="Generate  Stylized  Image!" ,height=2, width=72, command =self.style)
        self.c_output = Canvas(self.frame2, width=512, height=512, bg='white')
        
        self.frame3 = Frame(self.frame2)
        self.b_save = Button(self.frame3, text = "SAVE IMAGE", height=2, width=35, command = self.save)
        self.b_restart = Button(self.frame3, text = "RESTART", height=2, width=35, command = self.reset)
        
        self.frame1.grid(row=0, column=0, sticky="nsew")
        self.b_content.pack(side = "top")
        self.c_content.pack(side = "top")
        self.b_style.pack(side = "top")
        self.c_style.pack(side = "top")
        
        self.frame2.grid(row=0, column=1, sticky="nsew")
        self.b_output.pack(side = "top")
        img_output = Image.open('painting_gui.png')
        img_output = ImageTk.PhotoImage(img_output)
        self.master.dp = img_output
        self.c_output.pack(side = "top")
        self.c_output.create_image(0, 0, image = img_output, anchor = 'nw')
        self.frame3.pack(side = "top")
        
        self.b_save.pack(side = "left")
        self.b_restart.pack(side = "left")
        
        self.style_image = None
        self.content_image = None
        self.output_image = None

        self.setup()
        
    def setup(self):
        self.b_output.config(state="disabled")
        self.b_save.config(state="disabled")
    
    def browse_content(self):
        content_input = filedialog.askopenfilename(filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png")))
        if content_input:
            self.content_image = content_input
            content_img = Image.open(self.content_image)
            content_img = content_img.resize((256, 256))
            content_img = ImageTk.PhotoImage(content_img)
            self.master.content_img = content_img
            self.c_content.create_image(0, 0, image = content_img, anchor = 'nw')
            self.enable_style()
    
    def browse_style(self):
      
        style_input = filedialog.askopenfilename(filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png")))
        if style_input:
            self.style_image = style_input
            style_img = Image.open(self.style_image)
            style_img = style_img.resize((256, 256))
            style_img = ImageTk.PhotoImage(style_img)
            self.master.style_img = style_img
            self.c_style.create_image(0, 0, image = style_img, anchor = 'nw')
            self.enable_style()
           

    def enable_style(self):
        if self.content_image and self.style_image:
            self.b_output.config(state="normal")
    
    def style(self):
        print(self.content_image)
        self.output_image = stylize(self.content_image, self.style_image)
        #self.output_image = Image.open('3.jpg')
        output_img = self.output_image.resize((512, 512))
        output_img = ImageTk.PhotoImage(output_img)
        self.master.output_img = output_img
        self.c_output.create_image(0, 0, image = output_img, anchor = 'nw')
        self.b_save.config(state="normal")
        
    def save(self):
        content_name = os.path.basename(self.content_image).split('.')[0]
        style_name = os.path.basename(self.style_image).split('.')[0]
        self.output_image.save('./RESULTS_gui/{}_{}.jpg'.format(content_name, style_name))
        
    def reset(self):
        self.c_content.delete('all')
        self.c_style.delete('all')
        self.c_output.create_image(0, 0, image = self.master.dp, anchor = 'nw')
        self.b_output.config(state="disabled")
        self.b_save.config(state="disabled")
        self.style_image = None
        self.content_image = None
        self.output_image = None

if __name__ == "__main__":
    root = Tk()
    main(root)
    root.title('STYLE TRANSFER')
    root.resizable(0, 0)
    root.mainloop()