import numpy as np
import heapq as hq
import heapq
import math

#Question1

_DIVISORS = [180.0 / 2 ** n for n in range(32)]

def floaterleave_latlng(lat, lng):
    if not isinstance(lat, float) or not isinstance(lng, float):
        print('Usage: floaterleave_latlng(float, float)')
        raise ValueError("Supplied arguments must be of type float!")

    if (lng > 180):
        x = (lng % 180) + 180.0
    elif (lng < -180):
        x = (-((-lng) % 180)) + 180.0
    else:
        x = lng + 180.0
    if (lat > 90):
        y = (lat % 90) + 90.0
    elif (lat < -90):
        y = (-((-lat) % 90)) + 90.0
    else:
        y = lat + 90.0

    morton_code = ""
    for dx in _DIVISORS:
        digit = 0
        if (y >= dx):
            digit |= 2
            y -= dx
        if (x >= dx):
            digit |= 1
            x -= dx
        morton_code += str(digit)

    return morton_code

def generate_sorted_objects(MBR_list):
    MBR_list.sort(key=lambda x: x[2])
    for i in range(0,len(MBR_list)):
        MBR_list[i].pop(2)
    return MBR_list

print("Please input coords and offsets file:")
coords=[]
f1=input()
f2=input()
fr=open(f1, 'r')
all_lines=fr.readlines()
for line in all_lines:
    line=line.strip().split(',')
    coords.append(list(map(float,line)))

offsets=[]
#fr=open('d:/download/offsets.txt', 'r')
fr=open(f2, 'r')
all_lines=fr.readlines()
for line in all_lines:
    line=line.strip().split(',')
    offsets.append(list(map(float,line)))

print("\nPart 1: index development")
MBR=[[i for j in range(4)]for i in range(10000)]

for i in range(len(offsets)):
	start = int(offsets[i][1])
	end = int(offsets[i][2])
	num_coords = end - start + 1

	x_coords = []
	y_coords = []

	for j in range(num_coords):
		x_coords.append(coords[start+j][0])
		y_coords.append(coords[start+j][1])

	MBR[i][0] = min(x_coords)
	MBR[i][1] = max(x_coords)
	MBR[i][2] = min(y_coords)
	MBR[i][3] = max(y_coords)

center_coords =[[i for j in range(2)]for i in range(10000)]

for i in range(len(MBR)):
	center_coords[i][0] = round(np.mean((MBR[i][0],MBR[i][1])),7)
	center_coords[i][1] = round(np.mean((MBR[i][2],MBR[i][3])),7)

#print(center_coords)
x = np.array(center_coords)[:,0]
y = np.array(center_coords)[:,1]


z_order=[]
for i in range(len(MBR)):
	z_order.append(floaterleave_latlng(y[i],x[i]))


new_list1=[]
id=np.array(offsets)[:,0]
id=list(id)

for i in range(len(MBR)):
    new_list =[id[i],MBR[i],z_order[i]]
    new_list1=new_list1+[new_list]

new_list1 = generate_sorted_objects(new_list1)

def construct(collection):
    level = [] # list of levels(of nodes) of nodes(of mbrs) of mbrs(of points)
    node_id = 0
    rank = 0
    t = 0
    # Until we reach the root node
    while len(collection)>1:
        if (len(collection) % 20 == 0):
            l = int(len(collection) / 20)
        else:
            l = int(len(collection) / 20) + 1

        # Split the mbr collection into nodes of 20
        nodes = [collection[x:x+20] for x in range(0, len(collection), 20)]
        # If the last node has less than 8 mbrs, fill with mbrs of the previous
        balance = 8 - len(nodes[-1])
        if balance > 0 and len(nodes) > 1:
            migrate = len(nodes[-2])
            nodes[-1] = nodes[-2][migrate-balance:] + nodes[-1]
            nodes[-2] = nodes[-2][:migrate-balance]
        # Make the new MBRs based on the corners of each 20-piece
        collection = []
        for i in range(l):
            x_low = []
            x_high = []
            y_low = []
            y_high = []
            for j in range(len(nodes[i])):
                x_low.append(nodes[i][j][1][0])
                x_high.append(nodes[i][j][1][1])
                y_low.append(nodes[i][j][1][2])
                y_high.append(nodes[i][j][1][3])
                node_id += 1
            xlow = min(x_low)
            xhigh = max(x_high)
            ylow = min(y_low)
            yhigh = max(y_high)
            t+=1
            new = [nodes[i][j][0], [xlow, xhigh, ylow, yhigh]]
            collection = collection + [new]
        level.append(nodes)
        print("{} nodes at level {}".format(len(nodes), rank))
        rank += 1
    return level

new_list2 = construct(new_list1)
t=0
for i in range(25):
    for j in range(20):
        new_list2[1][i][j][0] = t
        t+=1

t=500
for i in range(2):
    for j in range(len(new_list2[2][i])):
        new_list2[2][i][j][0] = t
        t+=1

new_list2[3][0][0][0]=525
new_list2[3][0][1][0]=526

is_leaf = []
node_id = []
t=0
for i in range(len(new_list2)):
    for j in range(len(new_list2[i])):
        node_id.append(t)
        if(t<500):
            is_leaf.append(0)
        else:
            is_leaf.append(1)
        t+=1

new_list3=[]
t=0
for i in range(len(new_list2)):
    for j in range(len(new_list2[i])):
        t_list =[is_leaf[t],node_id[t],new_list2[i][j]]
        new_list3.append(t_list)
        t+=1


def write_data_matrix(filename, in_list):
    outfile = open(filename, "w")

    for listitem in in_list:
        outfile.write('[')
        outfile.write(f"{','.join(list(map(str, listitem)))}")  # Works
        outfile.write(']')
        outfile.write('\n')
    outfile.close()

write_data_matrix('Rtree.txt', new_list3)


#############################################################3
#Question2

def range_query(rtree, window, results, node=None):
    # At the root node
    if node == None:
        node = rtree[-1]
        for n in node[2]:
            if intersects(n[1],window) or intersects(window,n[1]):
                range_query(rtree, window, results, rtree[n[0]])
    # At an intermediate node
    elif node[0] == 1:
        for n in node[2]:
            if intersects(n[1], window) or intersects(window, n[1]):
                range_query(rtree, window, results, rtree[n[0]])
    # At a leaf node
    else:
        for n in node[2]:
            if intersects(n[1], window) or intersects(window, n[1]):
                results.append(n[0])
                #print(n[0])

def intersects(mbr, window):
    if mbr[0] > window[1] or mbr[1] < window[0]:
        return False
    if mbr[2] > window[3] or mbr[3] < window[2]:
        return False
    return True

print("\nPart 2: range queries")
print("Please input Rqueies file:")
fname = input()
fr=open(fname, 'r')
queries=[]
all_lines=fr.readlines()
for line in all_lines:
    line=line.strip().split(' ')
    queries.append(list(map(float,line)))
queries = np.array(queries)
queries[:,[1,2]] = queries[:,[2,1]]
queries=queries.tolist()


for i in range(len(queries)):
    results = []
    range_query(new_list3, queries[i], results, node=None)
    print("{}({}):".format(i,len(results))," ".join(str(j) for j in results))

###########################################################3
#Question3

print('\n')
print("Part 3: kNN queries")

def find_knn(rtree, qp, k, num):

    results = []
    q = []

    for i in range(500):
        for mbr in rtree[i][2]:
            dist = distance(mbr,qp)
            hq.heappush(q, [dist,mbr[0]])
    small_to_big = [heapq.heappop(q) for i in range(len(q))]

    while len(small_to_big) > 0 and len(results) < k:
        result = hq.heappop(small_to_big)
        if result[0]==0:
            result = hq.heappop(small_to_big)
        else:
            results.append(result[1])

    print("{}({}):".format(num,k), end=' ')
    for r in range(k-1):
        l = hq.heappop(results)
        print("{},".format(l), end='')
    l = hq.heappop(results)
    print(l)


def distance(mbr, p):
    x = 0
    y = 0
    if p[0] < mbr[1][0]:
        x = mbr[1][0] - p[0]
    elif p[0] > mbr[1][1]:
        x = p[0] - mbr[1][1]
    else:
        x = 0

    if p[1] < mbr[1][2]:
        y = mbr[1][2] - p[1]
    elif p[1] > mbr[1][3]:
        y = p[1] - mbr[1][3]
    else:
        y = 0

    return math.sqrt(x**2 + y**2)


print("Please input NNqueries file:")
fname = input()
fr = open(fname,'r')
#fr=open('data/NNqueries.txt', 'r')
knn_queries=[]
all_lines=fr.readlines()
for line in all_lines:
    line=line.strip().split(' ')
    knn_queries.append(list(map(float,line)))

print("Please input k:")
k = int(input())
num = 0
for query in knn_queries:
    find_knn(new_list3,query,k,num)
    num+=1;


