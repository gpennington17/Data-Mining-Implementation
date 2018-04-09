import math
import numpy as np
import Nodes

def get_count_dic(X_a, y, labels, dist_vals=None):
    dic = {}
    #counting labels of each distict val of attv
    if dist_vals is not None:
        for val in dist_vals:
            total=0
            cnt_dic = {}
            for label in labels:
                cnt_dic[label]=0
            for ndx in range(len(X_a)):
                if val == X_a[ndx]:
                    for label in labels:
                        if label == y[ndx]:
                            cnt_dic[label]+=1
                            total+=1
            cnt_dic['tot']=total
            dic[val] = cnt_dic
        return dic
    
    #only counting number of labels
    else:
        for label in labels:
            dic[label]=0

        for ndx in range(len(X_a)):
            for label in labels:
                if label == y[ndx]:
                    dic[label]+=1
        
        return dic

def Info_D(X,y,attb_ndx,dist_vals,labels):
    dic = get_count_dic(X,y,labels)
    Info_D=0
    for label in labels:
        if dic[label] <= 0:
            continue
        pi=dic[label]/len(X)
        Info_D-=pi*math.log(pi,2)
    return Info_D
    
def Info_gain(X,y,attb_ndx,dist_vals,labels,type_='cat'):
    #Step 1. Find info(D)
    ID = Info_D(X,y,attb_ndx,dist_vals,labels)
    
    #Step 2. Find info(D_A)
    if type_ == 'cat':
        
        X_a=X[:,attb_ndx]
        count_dic = get_count_dic(X_a,y,labels,dist_vals=dist_vals)
        ID_A=0
        for item in count_dic:
            W=count_dic[item]['tot']/len(X_a)
            temp=0
            for label in labels:
                if count_dic[item][label] <= 0:
                    continue
                pi=count_dic[item][label]/count_dic[item]['tot']
                temp-=pi*math.log(pi,2)
            ID_A+=W*temp
        return ID-ID_A
		
		
def part_X(X,y,attb_ndx):
    data_new = np.concatenate((X,y[:,None]),axis=1)
    unis = np.unique(X[:,attb_ndx])
    n_parts = len(unis)
    New_X=[]
    New_Y=[]
    temp3=[]
    for part in range(n_parts):
        temp1=[]
        temp2=[]
        for row in data_new:
            if(row[attb_ndx] == unis[part]):
                temp1.append(row)
                temp2.append(row[-1])
        temp1=np.array(temp1)
        temp2=np.array(temp2)
        New_X.append(np.delete(temp1,[attb_ndx,len(row)-1],axis=1))
        New_Y.append(temp2)
        temp3.append(unis[part])
    return New_X, New_Y, temp3
	
	
class DCT_classifer:
    def __init__(self,attbs,labels):
        self.attb_list=list(attbs)
        self.all_attbs=list(attbs)
        self.tree=Nodes.DCT_Tree()
        self.labels=list(labels)
        self.level=0
    
    def fit(self,X,y):
        return self.build(X,y)
    
    def clear(self):
        self.attb_list=list(self.all_attbs)
        self.tree=Nodes.DCT_Tree()
        self.level=0
    
    def build(self,X,y,parent=None,end_name=None):
        self.level+=1
        #Step 1. if all same class than we are done
        if len(np.unique(y)) == 1:
            #return "val="+end_name+"  ->  label="+y[0]
            return tuple((end_name,y[0]))
            
        #Step 2. if there are no attributes then return N
        if len(self.attb_list) == 0:
            dic = get_count_dic(X,y,np.unique(y))
            Max=-1
            majority='None'
            for item in dic:
                if dic[item] > Max:
                    majority = item
            #return "val="+end_name+"  ->  label="+y[0]
            return ((end_name,majority))
        
        #Step 3. find best splitting method (default using IG)
        best_val, best_attb = -1, 'None' 
        for ndx in range(len(self.attb_list)):
            distinct = np.unique(X[:,ndx])
            labels = np.unique(y)
            IG=Info_gain(X,y,ndx,distinct,labels)
            if IG > best_val:
                best_val = IG
                best_attb = ndx
        split_attb = self.attb_list[best_attb]
        self.attb_list.remove(split_attb)
        if parent == None:
            self.root=split_attb
            self.tree.add_node(split_attb)
        else:
            #split_attb="val="+end_name+" -> "+split_attb
            split_attb=tuple((end_name,split_attb))
            self.tree.add_node(split_attb,parent)
        #Step 4. partition
        X_parts, Y_parts, names = part_X(X,y,best_attb)
        for p in range(len(X_parts)):
            self.tree.add_node(self.build(X_parts[p],Y_parts[p],parent=split_attb,end_name=names[p]),parent=split_attb)
        return 'END'
    
    def show(self):
        self.tree.display(self.root,0)
        
    def predict(self,X):
        def p_row(row,root=-1):
            y='NF'
            if root == -1:
                root = self.root
                r_ndx=self.all_attbs.index(root)
            else:
                r_ndx=self.all_attbs.index(root[1])
            childs= self.tree.nodes[root].children
            r_attb = row[r_ndx]
            for tup in childs:
                if tup[0] == r_attb:
                    if tup[1] in  self.labels:
                        y = tup[1]
                        break
                    else:
                        y = p_row(row,root=tup)
                        break
            return y
        y=[]
        X=np.array(X)
        if len(X.shape) > 1:
            for row in X:
                y.append(p_row(row))
        else:
            return p_row(X)
        return y