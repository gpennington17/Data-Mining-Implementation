#Nodes to be used by decision tree alg
class DCT_Node:
    def __init__(self, ID):
        self.ID = ID  #the label will be splitting criterion
        self.children = []
    def label(self):
        return self.ID
    def add_child(self,ID):
        self.children.append(ID)

class DCT_Tree:
    def __init__(self):
        self.nodes={}
    
    def add_node(self, ID, parent=None):
        if(ID == 'END'):
            return None
        node=DCT_Node(ID)
        self.nodes[ID]=node
        if parent is not None:
            if( parent not in self.nodes):
                print("Failed making",ID,"have parent",parent,"\n")
                return node
            self.nodes[parent].add_child(ID)
        return node
    
    def display(self,ID,depth=0):
        children = self.nodes[ID].children
        if depth == 0:
            print("{0}".format(ID))
        else:
            print("\t"*depth, "{0}".format(ID))
        
        depth += 1
        for child in children:
            self.display(child,depth)