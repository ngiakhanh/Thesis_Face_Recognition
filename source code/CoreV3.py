from tkinter import *
from tkinter import messagebox, filedialog
import CoreV2 as core 
import os
from tkinter.scrolledtext import ScrolledText
import time

HEIGHT = 430
WIDTH = 640
title = "Face recognition algorithms"

class Details:
    def __init__(self, master, k, m, threshold, rights, total, success_rate, results, positions_list, method, ignore, mean_within_class_distance, mean_between_class_distance, database, checking, data_dict, stop_signal, image_dictionary, mean_vector, eigen_vectors_eig, time_diff):
        self.master = master
        self.k = k
        self.m = m
        self.threshold = threshold
        self.rights = rights
        self.total = total
        self.success_rate = success_rate
        self.results = results
        self.positions_list = positions_list
        self.method = method
        self.ignore = ignore
        self.mean_within = mean_within_class_distance
        self.mean_between = mean_between_class_distance
        self.database = database
        self.checking = checking
        self.data_dict = data_dict
        self.stop_signal = stop_signal
        self.image_dictionary = image_dictionary
        self.mean_vector = mean_vector
        self.eigen_vectors = eigen_vectors_eig
        self.time_diff = time_diff

        self.master.title('Details')
        positionRight = int(root.winfo_screenwidth()/2 - WIDTH/2)
        positionDown = int(root.winfo_screenheight()/2 - HEIGHT/2) 
        self.master.geometry("{}x{}+{}+{}".format(WIDTH, 450, positionRight, positionDown))

        self.summ = Label(self.master, text='SUMMARY:')
        self.summ.place(x=25, y=25)

        self.info = ScrolledText(self.master, undo=True, height=12, width=72)
        self.info.place(x=25, y=50)
        self.info.insert(INSERT, "METHOD: " + str(self.method) + "\n")
        if (ignore > -1): 
            self.info.insert(INSERT, "Ignored first " + str(self.ignore) + " component(s)\n")
        self.info.insert(INSERT, "K: " + str(self.k) + "\n")
        if (m > -1):
            self.info.insert(INSERT, "m: " + str(self.m) + "\n")
        self.info.insert(INSERT, "Threshold: "+ str(self.threshold) + "\n")
        self.info.insert(INSERT, "Right: " + str(self.rights) + "/" + str(self.total) + "\n")
        self.info.insert(INSERT, "Success rate: "+ str(self.success_rate) + "%\n")
        self.info.insert(INSERT, "Mean within class distance: " + str(self.mean_within) + "\n")
        self.info.insert(INSERT, "Mean between class distance: " + str(self.mean_between) + "\n")
        self.info.insert(INSERT, "Within/Between ratio: " + str(self.mean_within/self.mean_between) + "\n")
        self.info.insert(INSERT, "Execution time: " + str(self.time_diff/1000) + "(s)\n")
        self.info.insert(INSERT, "---------------------\n")

        self.data = Button(self.master, text = "Face\nDatabase", command=lambda:self.Result(self.database), height=3, width=10)
        self.data.place(x = 25, y = 270)

        self.test = Button(self.master, text = "Checking\nFaces", command=lambda:self.Result(self.checking), height=3, width=8)
        self.test.place(x = 165, y = 270)

        self.graph = Button(self.master, text = "Graph", command=self.Graph, height=3, width=8)
        self.graph.place(x = 290, y = 270)

        self.res = Button(self.master, text = "Results", command=lambda:self.Result(self.results), height=3, width=8)
        self.res.place(x = 415, y = 270)

        self.pos = Button(self.master, text = "Detailed\nPositions", command=lambda:self.Result(self.positions_list), height=3, width=8)
        self.pos.place(x = 540, y = 270)

        self.mean = Button(self.master, text = "Mean\nVector", command=self.Mean_vector, height=3, width=10)
        self.mean.place(x = 125, y = 350)

        self.eig = Button(self.master, text = "Eigenvector", command=self.Eigenvector, height=3, width=10)
        self.eig.place(x = 283, y = 380)

        self.eigs = Button(self.master, text = "All\nEigenvectors", command=self.All, height=3, width=10)
        self.eigs.place(x = 441, y = 350)

        self.num_eig = Text(self.master, height=1, width=5)
        self.num_eig.insert(INSERT, "1")
        self.num_eig.place(x = 303, y = 350)
        # (x = 283, y = 350)
        # text3.get("1.0",'end-1c')

    def Mean_vector(self):
        core.plot_mean_vector(self.image_dictionary, self.mean_vector)

    def Eigenvector(self):
        number = int(self.num_eig.get("1.0",'end-1c'))-1
        if (number < 0 or number > self.eigen_vectors.shape[1]):
            messagebox.showinfo("Warning", "Invalid eigenvector!")
        else:
            core.plot_eigen_vector(self.image_dictionary, self.eigen_vectors, number)

    def All(self):
        core.plot_eigen_vectors(self.image_dictionary, self.eigen_vectors)

    def Graph(self):
        title=self.method
        if (ignore>0):
            title += ' ignored first ' + str(ignore) + 'component(s)'
        core.plot_scatter_UI(self.data_dict, title, self.stop_signal)

    def Result(self,a):
        os.startfile(a)

    def close_windows(self):
        self.master.destroy()

root = Tk()
root.title(title)

positionRight = int(root.winfo_screenwidth()/2 - WIDTH/2)
positionDown = int(root.winfo_screenheight()/2 - HEIGHT/2) 
root.geometry("{}x{}+{}+{}".format(WIDTH, HEIGHT, positionRight, positionDown))

def filepath1():

    path = filedialog.askdirectory()
    text1.delete(1.0,END)
    text1.insert(INSERT, path)
    print(path)

def filepath2():

    path = filedialog.askdirectory()
    text2.delete(1.0,END)
    text2.insert(INSERT, path)
    print(path)

def sel():
   selection = "You selected the option " + str(var.get())
   print(selection)

#Line1
label1 = Label(root, text="FACE DATABASE:")
label1.place(x = 25, y = 50)

text1 = Text(root, height=1, width=50)
text1.insert(INSERT, "...")
text1.place(x = 130, y = 50)

B1 = Button(root, text = "Browse", command=filepath1)
B1.place(x = 550, y = 46)

#Line2
label2 = Label(root, text="CHECKING FACES:")
label2.place(x = 25, y = 100)

text2 = Text(root, height=1, width=50)
text2.insert(INSERT, "...")
text2.place(x = 130, y = 100)

B2 = Button(root, text = "Browse", command=filepath2)
B2.place(x = 550, y = 96)

#Line3
label3 = Label(root, text="METHOD:")
label3.place(x = 25, y = 150)

var = StringVar()
var.set('Eigenface')
R1 = Radiobutton(root, text = "Eigenface", variable = var, value = 'Eigenface', command = sel, anchor=W)
R1.place(x = 125, y = 149)

R2 = Radiobutton(root, text = "Fisherface", variable = var, value = 'Fisherface', command = sel, anchor=W)
R2.place(x = 230, y = 149)

#Line4
label4 = Label(root, text="STOP SIGNAL:")
label4.place(x = 25, y = 200)

text3 = Text(root, height=1, width=5)
text3.insert(INSERT, "_")
text3.place(x = 110, y = 200)

label5 = Label(root, text="DESIRED HEIGHT:")
label5.place(x = 250, y = 200)

text4 = Text(root, height=1, width=5)
text4.insert(INSERT, "128")
text4.place(x = 355, y = 200)

label6 = Label(root, text="DESIRED WIDTH:")
label6.place(x = 450, y = 200)

text5 = Text(root, height=1, width=5)
text5.insert(INSERT, "128")
text5.place(x = 550, y = 200)

#Line5
label7 = Label(root, text="THRESHOLD:")
label7.place(x = 25, y = 250)

text6 = Text(root, height=1, width=5)
text6.insert(INSERT, "500")
text6.place(x = 110, y = 250)

label8 = Label(root, text="K:")
label8.place(x = 205, y = 250)

text7 = Text(root, height=1, width=7)
text7.insert(INSERT, "40")
text7.place(x = 225, y = 250)

label10 = Label(root, text="IGNORE:")
label10.place(x = 325, y = 250)

text9 = Text(root, height=1, width=5)
text9.insert(INSERT, "0")
text9.place(x = 380, y = 250)

label9 = Label(root, text="M:")
label9.place(x = 470, y = 250)

text8 = Text(root, height=1, width=7)
text8.insert(INSERT, "default")
text8.place(x = 495, y = 250)

#Line6
CheckVar1 = StringVar()
CheckVar1.set('False')
C1 = Checkbutton(root, text = "SELF-LEARNING", variable = CheckVar1, onvalue = 'True', offvalue = 'False', height=5, width = 20)
C1.place(x = -3, y = 270)

#Bottom
def run(image_dir, stop_signal, check_image_dir, sizeA, sizeB, threshold, k=-1, clear=True, ignore=0):
    if (text8.get("1.0",'end-1c') == 'default'):  
        m = -2
    else:
        m = int(text8.get("1.0",'end-1c'))

    if (CheckVar1.get() == 'True'):
        copy = True
    elif (CheckVar1.get() == 'False'):
        copy = False

    if (var.get() == 'Eigenface'):
        # k, threshold, rights, total, success_rate, results, positions_list, mean_within_class_distance, mean_between_class_distance, data_dict, image_dictionary, mean_vector, eigen_vectors_eig =40, 500, 11, 15, 73, 'C:/Users/desktop.ini', 'C:/Users/desktop.ini', 4, 5, 6, 0, 1, 2
        start_time = int(round(time.time() * 1000))
        k, threshold, rights, total, success_rate, results, positions_list, mean_within_class_distance, mean_between_class_distance, data_dict, image_dictionary, mean_vector, eigen_vectors_eig = core.run_eigenface(os.path.abspath(image_dir), stop_signal, os.path.abspath(check_image_dir), [sizeA, sizeB], threshold=[threshold], k=k, clear=clear, copy=copy, ignore_first=ignore, title='EigenFace', results='eigenface_results.txt', positions_list='eigenface_positions.txt')
        time_diff = int(round(time.time() * 1000)) - start_time

        new = Tk()
        Details(new, k, -1, threshold, rights, total, success_rate, results, positions_list, 'Eigenface', ignore, mean_within_class_distance, mean_between_class_distance, image_dir, check_image_dir, data_dict, stop_signal, image_dictionary, mean_vector, eigen_vectors_eig, time_diff)  
        

    elif (var.get() == 'Fisherface'):
        start_time = int(round(time.time() * 1000))
        k, m, threshold, rights, total, success_rate, results, positions_list, mean_within_class_distance, mean_between_class_distance, data_dict, image_dictionary, mean_vector, eigen_vectors_eig = core.run_fisherface(os.path.abspath(image_dir), stop_signal, os.path.abspath(check_image_dir), [sizeA, sizeB], threshold=[threshold], k=k, m=m, clear=clear, copy=copy, title='Fisherface', results='fisherface_results.txt', positions_list='fisherface_positions.txt')
        time_diff = int(round(time.time() * 1000)) - start_time

        new = Tk()
        Details(new, k, m, threshold, rights, total, success_rate, results, positions_list, 'Fisherface', -1, mean_within_class_distance, mean_between_class_distance, image_dir, check_image_dir, data_dict, stop_signal, image_dictionary, mean_vector, eigen_vectors_eig, time_diff)
    
    else:
        messagebox.showinfo("Warning", "You didn't choose method!")
source = text1.get("1.0",'end-1c')
stop = text3.get("1.0",'end-1c')
test = text2.get("1.0",'end-1c')
sizeA = int(text4.get("1.0",'end-1c'))
sizeB = int(text5.get("1.0",'end-1c'))
threshold = int(text6.get("1.0",'end-1c'))
k = int(text7.get("1.0",'end-1c'))
clear=True

ignore = int(text9.get("1.0",'end-1c'))

B3 = Button(root, text = "Start", command=lambda:run(text1.get("1.0",'end-1c'), text3.get("1.0",'end-1c'), text2.get("1.0",'end-1c'), int(text4.get("1.0",'end-1c')), int(text5.get("1.0",'end-1c')), int(text6.get("1.0",'end-1c')), int(text7.get("1.0",'end-1c')), True, int(text9.get("1.0",'end-1c'))), height=2, width=10)
B3.place(x = 280, y = 365)

root.mainloop()



