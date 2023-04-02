import subprocess
import json
import platform
import os
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageDraw, ImageFont
# import pytesseract
import numpy as np
import time
import subprocess
from rich.progress import track
from .chatgpt_caller import chatgpt_caller

basic_path = os.path.join(os.path.dirname(__file__), "..")

model_name_to_dict = {"lcnet" : "resources/pretrained_model/picodet_lcnet_x1_0_fgd_layout_infer"}


class pdf_reader():
    def __init__(self, pdf_file, output_dir, model_name, device, api_key) -> None:
        self.pdf_file = pdf_file
        self.pdf_page_num = 12 # TODO: get pdf page num
        self.output_dir = os.path.join(output_dir, pdf_file.split("/")[-1].split(".")[0])
        self.model_name = model_name
        self.device = device
        self.ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False)
        self.imgName2pageInfo = {}
        self.imgName2imgs = {}
        self.bbox = None
        self.if_split_page = True

        self.pdfPicPath = os.path.join(self.output_dir, "pdf_pics")
        self.create_folder(self.pdfPicPath)
        self.structurePath = os.path.join(self.output_dir, "structured_pics")
        self.create_folder(self.structurePath)
        self.textPath = os.path.join(self.output_dir, "texted_pics")
        self.create_folder(self.textPath)
        self.gptPath = os.path.join(self.output_dir, "gpt_results")
        self.create_folder(self.gptPath)

        self.api_key = api_key
        self.chatgpt_caller = chatgpt_caller(self.gptPath, self.api_key)

    def create_folder(self, folder_path):
        # create path if not exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def generate_pics(self):
        start = time.time()
        command = ""
    
        # judge os is windows or linux or mac
        if platform.system() == 'Windows':
            command = [os.path.join(basic_path, "src/pdftopic/pdftopng.exe"), "-q", self.pdf_file, self.pdfPicPath+"/pics"]
        elif platform.system() == 'Linux':
            command = [os.path.join(basic_path, "src/pdftopic/pdftopng_linux"), "-q", self.pdf_file, self.pdfPicPath+"/pics"]
        elif platform.system() == 'Darwin':
            command = [os.path.join(basic_path, "src/pdftopic/pdftopng_mac"), "-q", self.pdf_file, self.pdfPicPath+"/pics"]
        subprocess.run(command)

        for file_name in os.listdir(self.pdfPicPath):
            self.imgName2pageInfo[file_name] = {"pageNum": int(file_name.split("-")[-1].split(".")[0].strip()), "bbox": [], "text": []}
            self.imgName2imgs[file_name] = Image.open(os.path.join(self.pdfPicPath, file_name))

        end = time.time()
        print("generate pics time: ", end-start)

    # def generate_txt_with_tes(self):
    #     start = time.time()

    #     self.contents = ""
    #     for file_name in track(sorted(list(self.imgName2imgs.keys()))):
    #         result = pytesseract.image_to_string(self.imgName2imgs[file_name])

    #         # replace all single \n with space and replace \n\n with \n
    #         result = result.replace("\n\n", "?!?").replace("\n", " ").replace("?!?", "\n").replace("- ", "")
    #         self.contents += result + "\n"
        
    #     with open(os.path.join(self.textPath, "text_tes.txt"), "w") as f:
    #         f.write(self.contents)

    #     end = time.time()
    #     print("generate pics time: ", end-start)

    def generate_structured_pics(self):
        start = time.time()
        command = ["python", os.path.join(basic_path, "src/structurer/infer.py"),
                "--model_dir=" + os.path.join(basic_path, model_name_to_dict[self.model_name]),
                "--image_dir=" + self.pdfPicPath,
                "--device=" + self.device,
                "--output_dir=" + self.structurePath,
                "--save_results"]
        subprocess.run(command)

        end = time.time()
        print("generate structured pics time: ", end-start)

    def add_n_in_text(self, text, font, max_width):
        """
        将文本添加 \n 以使文本适合宽度为 max_width 的框中
        """
        text_length = font.getlength(text)
        # 如果文本本身已经适合框中，则直接返回原始文本
        if text_length <= max_width:
            return text
        
        # 否则，添加换行符，将文本分成多行
        words = text.split(' ')
        lines = []
        current_line = words[0]
        
        for word in words[1:]:
            if font.getlength(current_line + ' ' + word) <= max_width:
                current_line += ' ' + word
            else:
                lines.append(current_line)
                current_line = word
        
        # 将最后一行添加到行列表中
        lines.append(current_line)
        
        # 将行列表连接成一个字符串，并在每行末尾添加换行符
        result = '\n'.join(lines)
        return result

    def generate_text(self):
        '''
        bbox [0,1,2,3]
        (0,1)-----------------
          |                   |
          |                   |
          |                   |
          |                   |
           ---------------(0+2,1+3)
        '''
        start = time.time()
        with open(os.path.join(self.structurePath, "bbox.json"), "r") as f:
            self.bbox = json.load(f)

        for box in self.bbox:
            pageNum = self.imgName2pageInfo[box["file_name"]]["pageNum"]
            self.imgName2pageInfo[box["file_name"]]["bbox"].append(box["bbox"])
            x_min = box['bbox'][0]-5
            y_min = box['bbox'][1]
            x_max = box['bbox'][0]+box['bbox'][2]+5
            y_max = box['bbox'][1]+box['bbox'][3]

            cropped_img = self.imgName2imgs[box["file_name"]].crop((x_min, y_min, x_max, y_max))
            cropped_img = np.asarray(cropped_img)
            ocrResult = self.ocr.ocr(cropped_img, cls=False)
            str = ""
            if ocrResult is None:
                print("ocrResult is None with category: " + box["category_id"])
                continue
            for line in ocrResult:
                str += line[1][0] + " "
            self.imgName2pageInfo[box["file_name"]]["text"].append(str)

        with open(os.path.join(self.textPath, "text.json"), "w") as f:
            json.dump(self.imgName2pageInfo, f)
        
        end = time.time()
        print("generate text time: ", end-start)

    def generate_txt(self, readTextFile = False):
        start = time.time()
        if readTextFile:
            with open(os.path.join(self.textPath, "text.json"), "r") as f:
                self.imgName2pageInfo = json.load(f)
        
        all_x_min = []
        self.max_x_max = 0
        for name in self.imgName2pageInfo:
            for box in self.imgName2pageInfo[name]["bbox"]:
                x_min = box[0]-5
                x_max = box[0]+box[2]+5

                all_x_min.append(x_min)
                if x_max > self.max_x_max:
                    self.max_x_max = x_max

        self.if_split_page = (sum(all_x_min)/len(all_x_min) > self.max_x_max/4) # 用于判断分栏
        print(f"split page: {self.if_split_page}")

        self.contents = ""
        for name in sorted(list(self.imgName2pageInfo.keys())):
            if not self.if_split_page:
                all_bbox_y = [i[1] for i in self.imgName2pageInfo[name]["bbox"]]
                sorted_index =  np.argsort(all_bbox_y)
                for i in sorted_index:
                    self.contents += self.imgName2pageInfo[name]["text"][i] + "\n"
            else:
                all_bbox_x = [i[0] for i in self.imgName2pageInfo[name]["bbox"]]

                bboxs = [[],[]]
                for i in range(len(all_bbox_x)):
                    bbox_index = 0
                    if all_bbox_x[i] > self.max_x_max/2:
                        bbox_index = 1
                    bboxs[bbox_index].append((self.imgName2pageInfo[name]["bbox"][i][1], self.imgName2pageInfo[name]["text"][i]))

                for i in bboxs:
                    sorted_bbox = sorted(i, key=lambda bbox: bbox[0])
                    for text_box in sorted_bbox:
                        self.contents += text_box[1] + "\n"
                        
        with open(os.path.join(self.textPath, "text.txt"), "w") as f:
            f.write(self.contents)

        end = time.time()
        print("generate txt time: ", end-start)

    def draw_ocr_result(self, readTextFile = False):
        start = time.time()
        if readTextFile:
            with open(os.path.join(self.textPath, "text.json"), "r") as f:
                self.imgName2pageInfo = json.load(f)
        
        for imgName in self.imgName2pageInfo:
            imgFile = self.imgName2imgs[imgName].convert('RGB')
            img = self.imgName2pageInfo[imgName]
            boxes = []
            for i, box in enumerate(img["bbox"]):
                x_min = box[0]-5
                y_min = box[1]
                x_max = box[0]+box[2]+5
                y_max = box[1]+box[3]
                boxes.append([x_min, y_min, x_max, y_max])
            texts = img["text"]

            draw = ImageDraw.Draw(imgFile)
            for box in boxes:
                draw.rectangle(box, outline="red")

            # 创建一个与原图大小相同的画布，用于放置右边的区域
            canvas = Image.new("RGB", (imgFile.width * 2, imgFile.height), (255, 255, 255))

            # 在画布上画出对应框的位置，并在每个框里写入文字
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.truetype(os.path.join(basic_path, "resources/fonts/latin.ttf"), size=16)  # 定义字体

            for i, box in enumerate(boxes):
                if len(texts[i]) != 0:
                    x1, y1, x2, y2 = box
                    draw.rectangle((x1 + imgFile.width, y1, x2 + imgFile.width, y2), outline="red")  # 在右边画出对应框的位置

                    # 定义框的宽度和高度
                    box_width = x2 - x1
                    box_height = y2 - y1

                    letter_width = font.getlength(texts[i])/len(texts[i])  # 计算文字的宽度
                    letter_height = font.getbbox("G")[3]  # 计算文字的高度

                    res_text = self.add_n_in_text(texts[i], font, box_width)

                    draw.text((x1 + imgFile.width + 2, y1 + 1), res_text, font=font, fill="black")

            # 将左右两块区域拼接在一起
            result = Image.new("RGB", (imgFile.width * 2, imgFile.height))
            result.paste(canvas, (0, 0))
            result.paste(imgFile, (0, 0))

            result.save(os.path.join(self.textPath, imgName.split(".")[0]+"-text.pdf"))

        end = time.time()
        print("draw ocr result time: ", end-start)

    def chatgpt_embeding(self, readTextFile = False):
        start = time.time()
        with open(os.path.join(self.textPath, "text.txt"), "r") as f:
            self.contents = f.read()

        self.chatgpt_caller.file2embedding(self.contents)

        end = time.time()
        print("chatgpt embeding time: ", end-start)

    def pdf_to_txt(self):
        self.generate_pics()
        self.generate_structured_pics()
        self.generate_text()
        self.generate_txt()
        
    def ask(self, question):
        [prompt,answer] = self.chatgpt_caller.ask(question)
        # print("question: ", question)
        # print("prompt: ", prompt)
        print("answer: ", answer)





            
