#!/usr/bin/python2.7

from numpy import *
import random
import matplotlib.pyplot as plt
from pylab import *
from networkx import *
from matplotlib.animation import FuncAnimation
import pylab
from matplotlib import rc
import matplotlib.patches as patches
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})


######################################################################################
#############  1.- Izhikevich model in Hierarchical network #############
#######################################################################################

######### General variables ################################################ 
g_inhibition=0.2 # percentage of inhibitory units (no hubs)
kappa=0.75 # hub interconnectivity
eta=0.75 # percentage of inhibitory local hubs
case=1# case 1: global hubs are inhibitory, case 2: global hubs are excitatory 
step=3  # step of the Ravast-Barabasi network
clusters=5 # number of replicas of R-B network
time=3000 # time steps
h=0.1 # time resolution in miliseconds
mul=40 # synaptic weight multiplier in mV
s=1 # seed

seed(s)
size=clusters*5**step
G=nx.empty_graph(size)
degree_i=zeros(size)
neighbors=zeros((size,size))


################# Hierarchical network ##############################
def small_clusters(count):
 for i in range(0,4):
  for j in range(i+1,5):
   if(i!=j):
    neighbors[int(count+i),int(degree_i[count+i])]=count+j
    neighbors[int(count+j),int(degree_i[count+j])]=count+i
    degree_i[int(count+i)]+=1
    degree_i[int(count+j)]+=1
    G.add_edge(count+i,count+j)
 count+=5
 return count


def hub(count,evol):
 a=random()
 for j in range(count-5**evol,count-5**(evol-1),5**(evol-1)): 
   for i in range(j,j+5**(evol-1)-5**(evol-2)):
    if(random()<1.0):
     neighbors[int(i),int(degree_i[i])]=count-1
     degree_i[i]+=1
    if(random()<1.0):
     neighbors[int(count-1),int(degree_i[count-1])]=i
     degree_i[count-1]+=1
    G.add_edge(count-1,i)
 return count

count=0 
for i in range(0,size/5):
 count=small_clusters(count)

start=2
while(start<=step):
 for i in range((5**start)-1,size,(5**start)):
  count=hub(i+1,start)
 start+=1


################# Rich club connectivity ######################

for i in range(24,size,25):
 for j in range(24,size,25):
  if(random()<kappa and i!=j):
   neighbors[int(i),int(degree_i[i])]=j
   degree_i[i]+=1
   G.add_edge(i,j)

############## synaptic weights of regular neurons  #############


weights=ones(size)*mul

for i in range(0,size,25):
 for j in range(i,i+24):
  if(random()<g_inhibition):
   weights[j]=-mul#weights1[randint(size/25+1, size-size/25)]*mul
  else:
   weights[j]=mul#weights1[randint(2*size/25, size)]*mul

############ synaptic weights of Hub neurons  ####################

for i in range(24,size,25):
 if(degree_i[i]>50):
  if(case==1):
   weights[i]=-mul
  else:
   weights[i]=mul
 else:
  if(random()<eta):
   weights[i]=-mul
  else:
   weights[i]=mul

########  initialization of variables of izhikevich model  ##########

u=zeros(size)
v=zeros(size)
a=zeros(size)
b=zeros(size)
c=zeros(size)
d=zeros(size)

for i in range(0,size):
 if(weights[i]<0):   
  a[i]=0.02+0.08*random()
  b[i]=0.25-0.05*random()
  c[i]=-65
  d[i]=2
 else:
  a[i]=0.02
  b[i]=0.2
  c[i]=-65+15*(random())**2
  d[i]=8-6*(random())**2
 v[i]=c[i]
 u[i]=2

aux=ones((time,size))
fired=zeros((size,2))
l=0
trans=500 # time steps for transitory phase
serie=[]

fileNameTemplate = r'./serie_case{0:02d}_w{1:02d}_u{2:02d}_rich{3:02d}_inhi{4:02d}_h{5:02d}_s{6:02d}.dat' #series name
archi=open(fileNameTemplate.format(case,mul,size,int(kappa*100),int(eta*100),int(h*10),s),'w') 

############## time evolution of Izhikevich model #################################
for j in range(0,time+trans):
 for i in range(0,size):
  aporte=0
  for k in range(0,int(degree_i[i])):
   if(fired[int(neighbors[i,k]),int(l)]==1):
    aporte+=weights[int(neighbors[i,k])]
  if(weights[i]>0):
   I=5*random()
  else:
   I=2*random()
  A1=h*(0.04*pow(v[i],2)+5*v[i]+140-u[i]+I+aporte)  #//second order Runge-kutta integration
  B1=h*(a[i]*(b[i]*v[i]-u[i]))
  A2=h*(0.04*pow((v[i]+A1/2.0),2)+5*(v[i]+A1/2.0)+140-(u[i]+B1/2.0)+I+aporte)
  B2=h*(a[i]*(b[i]*(v[i]+A1/2.0)-(u[i]+B1/2.0)))
  v[i]+=A2;
  u[i]+=B2;
  if(v[i]>=30.0):
   fired[i,1-l]=1
   if(j>=trans):
    aux[j-trans,i]=30
   v[i]=c[i]
   u[i]=u[i]+d[i]
  elif(v[i]<c[i]):
   fired[i,1-l]=0
   v[i]=c[i]
   if(j>=trans):
    aux[j-trans,i]=c[i]-10
  else:
   if(j>=trans):
    aux[j-trans,i]=v[i]
 l=1-l
 if(j>=trans):
  serie.append(float(sum(v)/size))
  list_=[float(sum(v)/size)]
  archi.write(" ".join(map(lambda x: str(x), list_))+"\n")
archi.close


##########################################################################################################
##### 2.- Animation  #####
##########################################################################################################
"""
fig = plt.figure(figsize=(12,12))
fig.patch.set_facecolor('black')
plt.axis('off')

pos=nx.spring_layout(G)
nsizes=ones(size)*100
labels=[]
exci=[]
inhi=[]
for i in range(0,size):
 if(weights[i]<0 and degree_i[i]<20):
  if(len(inhi)<1):
   inhi.append(i)
 if(weights[i]>0 and degree_i[i]<20):
  if(len(exci)<1):
   exci.append(i)

x=zeros((size))
x3=zeros((size))
for i in range(0,size):
 if(weights[i]>0):
  x3[i]=1
 else:
  x3[i]=-1

for i in range(24,size,25):
 nsizes[i]=pow(G.degree(i)*2000,0.5)*1.5

nodes = nx.draw_networkx_nodes(G,pos, nodelist=G.nodes(), node_size=nsizes, cmap = plt.get_cmap('seismic'), node_color=x3, 
alpha=0.8)

nodes = nx.draw_networkx_nodes(G,pos, nodelist=exci,  node_color='darkred', node_size=pow(5*2000,0.5)*1.5, label='excitatory', alpha=0.8, scatterpoints=1)
nodes = nx.draw_networkx_nodes(G,pos, nodelist=inhi, node_color='darkblue', node_size=pow(5*2000,0.5)*1.5, label='inhibitory', alpha=0.8, scatterpoints=1)

dim=G.number_of_edges()
colors=ones(dim)*0.5
edges = nx.draw_networkx_edges(G,pos=pos, edgelist=G.edges(),  style='solid', width=colors, edge_color='blue', arrows=True) 
plt.legend(numpoints = 1)
nodes = nx.draw_networkx_nodes(G,pos, nodelist=G.nodes(), node_size=nsizes, cmap = plt.get_cmap('seismic'), node_color=x3)

def update(n):
  for j in range(0,size):
   if(aux[n,j]>-30.0):
    x[j]=0
   else:
    x[j]=x3[j]#aux[n,j]
  nodes.set_array(x)
  return nodes,

fileNameTemplate = r'./animation_case{0:02d}_w{1:02d}_u{2:02d}_rich{3:02d}_inhi{4:02d}_s{5:02d}_1.mp4'
anim = FuncAnimation(fig, update, frames=time, interval=10, blit=True,save_count=0,repeat=False)
anim.save(fileNameTemplate.format(case,mul,size,int(kappa*100),int(eta*100),s), fps=200)#writer = 'mencoder', fps=40)

"""
##########################################################################################################
#####  3.-1 raster plot  #####
##########################################################################################################

fig = plt.figure(figsize=(17.7,14.2))
ax1 = plt.subplot2grid((15, 30), (0, 0), colspan=30, rowspan=9)
ax2 = plt.subplot2grid((15, 30), (9, 0), colspan=30, rowspan=2)
ax3 = plt.subplot2grid((15, 30), (12, 10), colspan=10, rowspan=3)

ax1.set_xlim(0,time)
ax1.set_ylim(0,size)
ax1.set_ylabel("neurons", fontsize=32)
ax1.tick_params(labelsize=18)
ax1.text(time/20.0,size*0.9, r'a', fontsize=32, color='white', ha='center')

for i in range(24,size,25):
 if(weights[i]<0):
  rect = patches.Ellipse((4,i),time/70,6,linewidth=0.5,edgecolor='b',facecolor='b')#lightgreen 'lemonchiffon'
 else:
  rect = patches.Ellipse((4,i),time/70,6,linewidth=0.5,edgecolor='r',facecolor='r')#lightgreen 'lemonchiffon'
 ax1.add_patch(rect)

g=time/100
j=time/100

im=transpose(aux)
def funcion(j):
 f=0
 for i in range(0,time,g):
  for k in range(i,i+g):
   if(f==0):
    im[j,k]=-4.8
  if(f==0):
   f=1
  else:
   f=0 

if(size>125): ## threshold for dotted lines
 th=50
else:
 th=10

for i in range(0,len(degree_i)):
 if(degree_i[i]>th):
  if(i==len(degree_i)-1):
   funcion(i-1)
  else:
   funcion(i-1)
   funcion(i+1)

nuevoy=[]
list_y=[]
for i in range(0,time+time/5,time/5):
 list_y.append(int(i))
 nuevoy.append(int(i*h))

fileNameTemplate = r'./evolution_case{0:02d}_w{1:02d}_u{2:02d}_rich{3:02d}_inhi{4:02d}_h{5:02d}_s{6:02d}.png'
ax1.pcolor(im,cmap='gist_heat')      #subplot 1, time evolution
ax1.axes.get_xaxis().set_visible(False)

                                 # b) State of the system
ax2.set_xticks(list_y)
ax2.set_xticklabels(nuevoy, fontsize=22)
ax2.set_xlabel("time (ms)", fontsize=32)
ax2.plot(range(1,time+1), serie)
ax2.set_ylabel("S(t)", fontsize=32)
ax2.tick_params(labelsize=18)
ax2.text(time/20.0,0.6*(max(serie)-min(serie))+min(serie), r'b', fontsize=32, color='k', ha='center')

#################  DFA method   ###########################
rc('text', usetex=True)
x=zeros(time)
xn=zeros(time)
mean=sum(serie)/time

b=0
for i in range(0,time):
	b+=serie[i]-mean
	x[i]=b	

ns = []
fluctuations = []
bs=4 #box size
while (bs<=time*pow(4,-1)):
    n=int(time/bs) # caja 
    for g in range(0,int(bs)):
	s1=s2=r=t=j=0
	for i in range(0+int(g*time/bs),int(g*time/bs+time/bs)):	
		r+=x[i]*j
		t+=pow(j,2)
	 	s1+=x[i]
		s2+=j
		j+=1
	ss=s1*s2
	w=pow(s2,2)
	a=(n*r-ss)/(n*t-w)
	b=(s1-a*s2)/n
	j=f=0	
	for i in range(0+int(g*time/bs),int(g*time/bs+time/bs)):
		xn[i]=a*j+b
		j+=1
    for i in range(0,time):
	f+=pow(x[i]-xn[i],2)
    f=pow(f/time,0.5)
    ns.append(float(n))
    fluctuations.append(float(f))
    bs+=int(ceil(bs*(0.15)))
   
ax3.plot(ns, fluctuations, color='r')
ax3.text(4,max(fluctuations)*1.3, r'c', fontsize=32, color='k', ha='center')
ax3.set_xlim(1,time/2)
ax3.set_ylim(min(fluctuations)*0.25,max(fluctuations)*4.0)

plt.xscale('log')
plt.yscale('log')
ax3.set_ylabel("F(n)", fontsize=32)
ax3.set_xlabel("n", fontsize=32)

############### fit of F(n) vs n, and legends #######################

N=len(fluctuations)
s1=s2=r=t=0
for i in range(0,N):
	r+=log10(fluctuations[i])*log10(ns[i])
	t+=pow(log10(ns[i]),2)
	s1+=log10(fluctuations[i])
	s2+=log10(ns[i])
ss=s1*s2
w=pow(s2,2)
a=(N*r-ss)/(N*t-w)
b=(s1-a*s2)/N

minx=[]
miny=[]

listx=linspace(log10(4),log10(time/4),50)
for i in listx:
 minx.append(pow(10,i))
 miny.append(pow(10,a*i+b+0.2))

ax3.plot(minx, miny,'blue')

start=[minx[10],minx[10]]
end=[miny[10],miny[28]]
ax3.plot(start, end,'blue')

start=[minx[10],minx[28]-0.5]
end=[miny[28],miny[28]]
ax3.plot(start, end,'blue')

label=r'$\alpha={:.02f}$'
ax3.text(minx[10],miny[28]*1.4,  label.format(a), fontsize=37, ha='center')

ax3.tick_params(labelsize=24)
ax = savefig(fileNameTemplate.format(case,int(mul),size,int(kappa*100),int(eta*100),int(h*10),s), bbox_inches='tight')
