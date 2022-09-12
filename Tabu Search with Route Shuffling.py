import math
import numpy
from random import shuffle
import tkinter
import tkinter.filedialog
from scipy.spatial.distance import pdist, squareform
import ntpath
import random


class TSPuiWindow(tkinter.Frame):
    def __init__(self, root):
        tkinter.Frame.__init__(self, root)
        self.Numfiles = 0
        self.Filearray = []

        # options for buttons
        button_opt = {'fill': tkinter.constants.BOTH, 'padx': 5, 'pady': 5}

        # define buttons
        tkinter.Button(self, text='Add files', command=self.addfile).pack(**button_opt)
        tkinter.Button(self, text='Export results', command=self.exportresults).pack(**button_opt)

        # define options for opening or saving a file
        self.open_opt = options = {}
        options['defaultextension'] = '.txt'
        options['filetypes'] = [('all files', '.*'), ('text files', '.txt')]
        options['initialdir'] = 'C:\\'
        options['initialfile'] = 'Output-NN.txt'
        options['parent'] = root
        options['title'] = 'Open input file'

        self.save_opt = options = {}
        options['defaultextension'] = '.txt'
        options['filetypes'] = [('all files', '.*'), ('text files', '.txt')]
        options['initialdir'] = 'C:\\'
        options['initialfile'] = 'Output-NN.txt'
        options['parent'] = root
        options['title'] = 'Save output file'


    def addfile(self):
        # get filename
        filelist = tkinter.filedialog.askopenfilenames(**self.open_opt)
        print(filelist)

        # process filename
        for i in range (0,len(filelist)):
            filename = filelist[i]
            if filename:
                filearray = TSP("", filename) #using empty string for path because filename includes the full path
                self.Filearray.append(filearray)
                self.Numfiles += 1

    def exportresults(self):
        savefile = tkinter.filedialog.asksaveasfile(mode='w', **self.save_opt)

        myoutput = open(savefile.name, "a")
        for i in range(0,self.Numfiles):
            filename  = ntpath.basename(self.Filearray[i].filename)
            sequence  = str(self.Filearray[i].Sequence)
            totaldist = str(self.Filearray[i].TotalDist)
            nndist    = str(self.Filearray[i].NNDist)
            distances = str(self.Filearray[i].Distances)
            average = str(self.Filearray[i].Average)
            myoutput.write('Filename:       ' + filename + '\n')
            myoutput.write('NN distance:    ' + nndist + '\n')
            myoutput.write('List of computed distances:    ' + distances + '\n')
            myoutput.write('Average computed distance:     ' + average + '\n')
            myoutput.write('Best distance: ' + totaldist + '\n')
            myoutput.write('Best Sequence:\n')
            myoutput.write(sequence + '\n')
            myoutput.write('----------------\n')


        myoutput.close()



class TSP:
    def __init__(self, path, filename):
        coord = []
        self.path = path
        self.filename = filename
        with open(path+filename, "r") as iFile:
            self.InstanceName = iFile.readline().strip()
            iFile.readline()
            iFile.readline()
            iFile.readline()
            iFile.readline()
            iFile.readline()
            iFile.readline()
            iFile.readline()
            iFile.readline()
            line = iFile.readline().strip()
            while line != "":
                vals = line.split()
                coord.append([float(vals[1]), float(vals[2])])
                line = iFile.readline().strip()

        self.NumCust = len(coord) - 1
        self.Coord = numpy.array(coord)
        self.DistMatrix = squareform(pdist(self.Coord, "euclidean"))
        self.Sequence = []  # Will be a vector of the indices of nearest neighbours
        self.TotalDist = []
        self.Distances = []
        self.apply_nearest_neighbor()
        self.calculate_total_distance()
        self.NNDist = self.TotalDist
        self.tabu_search_random()


    def apply_nearest_neighbor(self):
        dist_matrix = self.DistMatrix.copy()
        for i in range( 0,self.NumCust+1):
            dist_matrix[i][i] = math.inf

        row   = 0
        index = 0
        for i in range (0,self.NumCust): # i denominates the row
            min_distance = math.inf
            for j in range (0,self.NumCust+1):
                if dist_matrix[row][j] < min_distance:
                    min_distance = dist_matrix[row][j]
                    index = j
                dist_matrix[j][row] = math.inf
            row = index
            self.Sequence.append(index)
        self.Sequence = numpy.concatenate(([0],self.Sequence, [0]), axis=0)

    def calculate_total_distance(self):
        self.TotalDist = 0
        for i in range(0, len(self.Sequence) - 1):
            self.TotalDist += self.DistMatrix[self.Sequence[i]][self.Sequence[i + 1]]

    def tabu_search_random(self):
        times = 0
        maxiter = 10
        seq_cur_sol = self.Sequence.copy()
        tabu_list   = []
        seq_best    = self.Sequence.copy()
        dist_best   = math.inf
        run = 1
        distances_list = []

        while run <= 6: #number of repetitions of the whole tabu_search algorithm
            tabulistlength = random.randint(1,20) #tabu tenure
            while times <= maxiter:
                dist_old = math.inf
                dist_cur_sol = 0
                for j in range(0, self.NumCust + 1):
                    dist_cur_sol += self.DistMatrix[seq_cur_sol[j]][seq_cur_sol[j + 1]]

                while dist_old > dist_cur_sol:
                    seq_old = seq_cur_sol.copy()
                    for i in range(1, self.NumCust + 1):
                        for j in range (1, self.NumCust + 1):
                            seq_new = seq_old.copy()
                            seq_new[i], seq_new[j] = seq_new[j], seq_new[i]
                            dist_old = 0
                            dist_new = 0
                            for k in range(0, self.NumCust + 1):
                                dist_old += self.DistMatrix[seq_old[k]][seq_old[k + 1]]
                                dist_new += self.DistMatrix[seq_new[k]][seq_new[k + 1]]
                            if (any((seq_new == x).all() for x in tabu_list)) == False and dist_new < dist_cur_sol: #Chechking whether solution is tabu
                                dist_cur_sol = dist_new.copy()
                                seq_cur_sol = seq_new.copy()


                tabu_list.append(seq_cur_sol)  # Using tabu tenure to keep tabu_list within predefined range
                if len(tabu_list) > tabulistlength:
                    tabu_list.remove(tabu_list[0])


                if dist_best > dist_cur_sol:  # aspiration criterion
                    dist_best = dist_cur_sol.copy()
                    seq_best  = seq_cur_sol.copy()

                seq_cur_sol = seq_cur_sol[1:len(seq_cur_sol) - 1]
                shuffle(seq_cur_sol) # Diversification element in the solution
                seq_cur_sol = numpy.concatenate(([0], seq_cur_sol, [0]), axis=0)
                times += 1
            times = 0
            distances_list.append(dist_best)
            run += 1
        self.Sequence  = seq_best
        self.TotalDist = dist_best
        self.Distances = distances_list
        self.Average = sum(distances_list)/float(len(distances_list))
        print ("A file is processed")


if __name__ == '__main__':
    root = tkinter.Tk()
    TSPuiWindow(root).pack()
    root.mainloop()