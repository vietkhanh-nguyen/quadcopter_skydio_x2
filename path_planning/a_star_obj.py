import math
from collections import deque, namedtuple
import numpy as np

# 3D node: z (slice), row (y index), col (x index)
NodeGrid3D = namedtuple('Node', ['z', 'row', 'col'])

class AStarNode:
    def __init__(self, node_pose: NodeGrid3D, start_node: NodeGrid3D, end_node: NodeGrid3D, parent_node = None):
        self.node_pose = node_pose
        self.start_node = start_node
        self.end_node = end_node
        self.parent_node = parent_node
        self.g = self.calculate_g(self.parent_node)
        self.h = self.calculate_h()
        self.f = self.calculate_f()
    
    def __eq__(self, other):
        if isinstance(other, NodeGrid3D):
            return (self.node_pose.z == other.z and
                    self.node_pose.row == other.row and
                    self.node_pose.col == other.col)
        if isinstance(other, AStarNode):
            return (self.node_pose.z == other.node_pose.z and
                    self.node_pose.row == other.node_pose.row and
                    self.node_pose.col == other.node_pose.col)
        return False

    def calculate_g(self, parent_node): 
        # g cost is the distance from starting node (cumulative)
        if parent_node is None:
            return 0.0
        dz = (self.node_pose.z - parent_node.node_pose.z)
        dy = (self.node_pose.row - parent_node.node_pose.row)
        dx = (self.node_pose.col - parent_node.node_pose.col)
        return parent_node.g + math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def calculate_h(self):
        # heuristic = Euclidean distance to end (3D)
        dz = (self.node_pose.z - self.end_node.z)
        dy = (self.node_pose.row - self.end_node.row)
        dx = (self.node_pose.col - self.end_node.col)
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def calculate_f(self):
        return self.g + self.h
    
    def recalculate(self, parent_node):
        new_g = parent_node.g + math.sqrt(
            (self.node_pose.col - parent_node.node_pose.col) ** 2 +
            (self.node_pose.row - parent_node.node_pose.row) ** 2 +
            (self.node_pose.z - parent_node.node_pose.z) ** 2
        )
        if new_g < self.g:
            self.parent_node = parent_node
            self.g = new_g
            self.f = self.calculate_f()
            return True
        else:
            return False

class OpenList:
    def __init__(self):
        self.open_list = set()         # set of (z,row,col)
        self.open_list_data = []       # ordered list of nodes by f
        self.arr = []                  # parallel list of f values for binary insertion
    
    def add_node(self, node: AStarNode):
        index = self.binary_search(node.f)
        self.open_list.add((node.node_pose.z, node.node_pose.row, node.node_pose.col))
        self.open_list_data.insert(index, node)
        self.arr.insert(index, node.f)
    
    def binary_search(self, target):
        if not self.arr:
            return 0
        left, right = 0, len(self.arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.arr[mid] == target:
                return mid
            elif self.arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left
    
    def is_not_in_open(self, node: AStarNode):
        key = (node.node_pose.z, node.node_pose.row, node.node_pose.col)
        if key not in self.open_list:
            self.add_node(node)
            return 1
        else:
            return 0

    def pop_node(self):
        node = self.open_list_data.pop(0)
        # remove only the first occurrence of f (consistent data)
        try:
            self.arr.pop(0)
        except ValueError:
            pass
        self.open_list.remove((node.node_pose.z, node.node_pose.row, node.node_pose.col))
        return node


class ClosedList:
    def __init__(self):
        self.closed_set = set()

    def add_node(self, node: AStarNode):
        self.closed_set.add((node.node_pose.z, node.node_pose.row, node.node_pose.col))

    def is_in_closed(self, node: AStarNode):
        return (node.node_pose.z, node.node_pose.row, node.node_pose.col) in self.closed_set
    
    def __len__(self):
        return len(self.closed_set)


class Stack:
    def __init__(self):
        self.stack = deque()

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        return self.stack.pop() if self.stack else None

    def peek(self):
        return self.stack[-1] if self.stack else None

    def is_empty(self):
        return not self.stack

    def size(self):
        return len(self.stack)
