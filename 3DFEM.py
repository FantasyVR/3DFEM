import taichi as ti
import numpy as np
from Camera import *
ti.init(arch=ti.cpu)


@ti.data_oriented
class SoftBody():
    def __init__(self, nodefile='cube.node', tetfile='cube.ele'):
        self.v = self.readNodefile(nodefile)
        self.t = self.readTetfile(tetfile)

        self.nv = len(self.v)
        self.nt = len(self.t)
        self.dim = 3
        # 存储节点位置
        self.nodes = ti.Vector(3, dt=ti.f32, shape=self.nv)
        self.scaleNodes = ti.Vector(3, dt=ti.f32, shape=self.nv)
        # 存储四面体的索引
        self.tets = ti.Vector(4, dt=ti.i32, shape=self.nt)

        self.m = ti.var(dt=ti.f32, shape=())
        self.vel = ti.Vector(3, dt=ti.f32, shape=self.nv)
        self.force = ti.Vector(3, dt=ti.f32, shape=self.nv)
        self.acc = ti.Vector(3, dt=ti.f32, shape=self.nv)
        self.restVolume = ti.var(dt=ti.f32, shape=(self.nt))
        #变形梯度相关
        self.F = ti.Matrix(3, 3, dt=ti.f32, shape=(self.nt))
        self.B = ti.Matrix(3, 3, dt=ti.f32, shape=(self.nt))
        self.D_s = ti.Matrix(3, 3, dt=ti.f32, shape=(self.nt))

        # Energy
        self.E = ti.var(dt=ti.f32, shape=(self.nt))

        # First Piola-Kirchoff stress tensor
        self.P = ti.Matrix(3, 3, dt=ti.f32, shape=(self.nt))

        #刚度矩阵
        self.K = ti.Matrix(3, 3, dt=ti.f32, shape=(self.nv, self.nv))

        # 仿真步长
        self.dt = ti.var(dt=ti.f32, shape=())
        # 重力加速度
        self.gravity = ti.var(dt=ti.f32, shape=())
        # Lame 常数
        self.mu = ti.var(dt=ti.f32, shape=())
        self.l = ti.var(dt=ti.f32, shape=())

    def readNodefile(self, nodefile=''):
        nodes = list()
        nFile = open(nodefile, 'r')
        line = int(nFile.readline().strip()[0])
        for l in nFile.readlines():
            node = tuple(map(float, l.strip().split()[1:]))
            nodes.append(node)
        nFile.close()
        return nodes

    def readTetfile(self, tetfile=''):
        tets = list()
        tFile = open(tetfile, 'r')
        line = int(tFile.readline().strip()[0])
        for l in tFile.readlines():
            tet = tuple(map(int, l.strip().split()[1:]))
            tets.append(tet)
        tFile.close()
        return tets

    @ti.kernel
    def init_TaichiMesh(self):
        for vertI in ti.static(range(self.nv)):
            self.nodes[vertI] = ti.Vector(list(self.v[vertI]))
        for tetI in ti.static(range(self.nt)):
            self.tets[tetI] = ti.Vector(list(self.t[tetI]))

        self.gravity[None] = -9.80
        self.m[None] = 1.0
        self.dt[None] = 1.0e-6
        self.mu[None] = 1.0e4
        self.l[None] = 1.0e4

        for i in self.nodes:
            self.acc[i] = ti.Vector([0,0,0])
            self.vel[i] = ti.Vector([0,0,0])
            self.force[i] = ti.Vector([0,0,0])


        self.computeB()

    def print_meshCPU(self):
        print("Print Mesh info in CPU")
        for vi in range(len(self.v)):
            print(list(self.v[vi]))
        for tetI in range(len(self.t)):
            print(list(self.t[tetI]))

    @ti.kernel
    def print_mesh(self):
        print("\n")
        print("Print Mesh info in Kernel")
        print("--------nodes-----------")
        for node in self.nodes:
            print(self.nodes[node])
        #print("--------tets-----------")
        #for tet in self.tets:
        #    print(self.tets[tet])
        print("\n")

    @ti.func
    def computeB(self):
        for tet in self.tets:
            tetIdx = self.tets[tet]
            p0 = self.nodes[tetIdx[0]]
            p1 = self.nodes[tetIdx[1]]
            p2 = self.nodes[tetIdx[2]]
            p3 = self.nodes[tetIdx[3]]
            self.B[tet] = ti.Matrix.cols([p1 - p0, p2 - p0,
                                          p3 - p0]).inverse()
            self.restVolume[tet] = ti.abs(ti.Matrix.cols(
                [p1 - p0, p2 - p0, p3 - p0]).determinant() / 6.0)
            #print("Rest Volume....")
            #print(self.restVolume[tet])

    @ti.func
    def computeD_s(self):
        for tet in self.tets:
            tetIdx = self.tets[tet]
            p0 = self.nodes[tetIdx[0]]
            p1 = self.nodes[tetIdx[1]]
            p2 = self.nodes[tetIdx[2]]
            p3 = self.nodes[tetIdx[3]]
            self.D_s[tet] = ti.Matrix.cols([p1 - p0, p2 - p0, p3 - p0])

    @ti.func
    def computeF(self):
        self.computeD_s()
        for tet in self.tets:
            self.F[tet] = self.D_s[tet] @ self.B[tet]
            #print("Deformation matrix for tet ", tet, " is: ", self.F[tet])

    @ti.func
    def computeEnergy(self):
        for tet in self.tets:
            mu, l, F = ti.static(self.mu[None], self.l[None], self.F[Tet])
            U, sigma, V = ti.svd(F)
            J = sigma[0] * sigma[1] * sigma[2]
            J = max(0.1, J)
            sumSqureSigma = sigma[0]**2 + sigma[1]**2 + sigma[2]**2
            self.E[tet] = 0.5 * mu * (sumSqureSigma - 3) - (
                mu - 0.5 * l * ti.log(J)) * ti.log(J)

    @ti.func
    def computeP(self):
        self.computeF()
        for tet in self.tets:
            U, sigma, V = ti.svd(self.F[tet])
            #print("Sigma of ", tet, " is: ", sigma[0,0],",",sigma[1,1],",",sigma[2,2])
            J = sigma[0,0] * sigma[1,1] * sigma[2,2]
            #print("Determinate: ", tet, " is : ", J)
            #J = self.F[tet].determinant()
            #J = max(0.2, J)
            FInvT = self.F[tet].transpose().inverse()
            self.P[tet] = self.mu[None] * (self.F[tet] - FInvT) + self.l[None] * ti.log(J) * FInvT

    @ti.func
    def computeForce(self):
        self.computeP()
        for tet in self.tets:
            H = self.restVolume[tet] * self.P[tet] * self.B[tet].transpose() # H = v * P * B^T, H = [f_1, f_2, f_3], f_0 = - sum_1^3{f_i}
            tetIdx = self.tets[tet]
            self.force[tetIdx[1]] += ti.Vector([H[0, 0], H[1, 0], H[2, 0]])
            self.force[tetIdx[2]] += ti.Vector([H[0, 1], H[1, 1], H[2, 1]])
            self.force[tetIdx[3]] += ti.Vector([H[0, 2], H[1, 2], H[2, 2]])
            self.force[tetIdx[0]] += -ti.Vector(
                [H[0, 0], H[1, 0], H[2, 0]]) - ti.Vector([
                    H[0, 1], H[1, 1], H[2, 1]
                ]) - ti.Vector([H[0, 2], H[1, 2], H[2, 2]])

    @ti.func
    def resetForce(self):
        for f in self.force:
            self.force[f] = ti.Vector([0.0, 0.0, 0.0])

    @ti.func
    def integrate(self):
        for n in self.nodes:
            #print("force for ", n , " node is: ", self.force[n])
            self.acc[n] = self.force[n] / self.m[None] + [0.0,self.gravity[None],0.0]
            self.vel[n] = (self.vel[n] + self.acc[n] * self.dt[None]) * math.exp(1.0e-6)
            self.nodes[n] += self.vel[n] * self.dt[None]

            # 处理碰撞问题
            if self.nodes[n].y < -1.0:
                self.nodes[n].y = -1.0

    @ti.kernel
    def step(self):
        self.resetForce()
        self.computeForce()  
        self.integrate()

    def displayTriangles(self, gui, mvp, points, indices):
        for i in range(indices.shape[0]):
            tempV = np.zeros((3,3))
            for j in range(indices.shape[1]):
                tv  = np.matmul(mvp, np.append(points[indices[i,j]],1.0))
                tv  = [x/tv[3] for x in tv][:3]
                tv  = [(x+1)/2.0 for x in tv]
                tempV[j] = tv 
            gui.line([tempV[0,0],tempV[0,1]],[tempV[1,0],tempV[1,1]], color=0x0000FF)
            gui.line([tempV[2,0],tempV[2,1]],[tempV[1,0],tempV[1,1]], color=0x0000FF)
            gui.line([tempV[0,0],tempV[0,1]],[tempV[2,0],tempV[2,1]], color=0x0000FF)

            #gui.triangle([tempV[0,0],tempV[0,1]],[tempV[1,0],tempV[1,1]],[tempV[2,0],tempV[2,1]],color=0x00FF00)

    def displayTets(self, gui, mvp):
        for i in range(self.nt):
            tIdx = self.tets[i]
            pL = [self.nodes[tIdx[i]] for i in range(4)]
            pL = [np.array([p[0],p[1],p[2],1.0]) for p in pL]
            pL = [np.matmul(mvp,p) for p in pL]
            pL = [p/p[3] for p in pL]
            pL = [(p+1)/2.0 for p in pL]
            for i in range(3):
                for j in range(i+1,4):
                    p1 = [pL[i][0],pL[i][1]]
                    p2 = [pL[j][0],pL[j][1]]
                    gui.line(p1,p2,radius=2,color=0x00FF00)

    def display(self, gui, mvp):
        #画出四面体网格
        #p = [np.array([self.nodes[i][0],self.nodes[i][1],self.nodes[i][2], 1.0]) for i in range(self.nv)] # p.xyz -> p.xyzw
        #pp = [np.matmul(mvp, p[i]) for i in range(self.nv) ] # mvp * p.xyzw
        #pp = [pp[i] / pp[i][3] for i in range(self.nv)] # p.xyz / p.w -> ndc coordinate
        #pp = [(pp[i][j] + 1)/2.0 for i in range(self.nv) for j in range(2)] # [-1,1] -> [0,1]
        #for i in range(self.nv):
        #    gui.circle([pp[2*i],pp[2*i+1]],color=0xFF0000, radius=5)

        self.displayTets(gui,mvp)

        # 画出Triangles
        triangleArray = np.array([[-1,-1.0,0.0],[1,-1.0,0],[-1,-1.0,1],[1,-1.0,1]])
        indices =np.array([[0,1,2],[1,2,3]],dtype=np.int32)
        self.displayTriangles(gui,mvp,triangleArray,indices)

if __name__ == "__main__":
    softCube = SoftBody()
    softCube.init_TaichiMesh()
    gui = ti.GUI("FEM")
    pause = True


    trans = [0,0,0]
    radius = 8
    t = time.time()
    trans[0] = radius * math.cos(2)
    trans[1] = 0.5
    trans[2] = radius * math.sin(2)
    
    p = projective()
    v = view(pos= trans)
    m = np.identity(4)
    mvp = np.matmul(p, np.matmul(v,m))

    while gui.running:
        for e in gui.get_events():
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE and e.type == ti.GUI.PRESS:
                pause = not pause
                print("Press SPACE")
                #softCube.step()
                #softCube.print_mesh()
                #trans = [0,0,0]
                #radius = 8
                #t = time.time()
                #trans[0] = radius * math.cos(2)
                #trans[2] = radius * math.sin(2)
                #p = projective()
                #v = view(pos= trans)
                #m = np.identity(4)
                #mvp = np.matmul(p, np.matmul(v,m))
        if not pause:
            softCube.step()

        softCube.display(gui, mvp)
        gui.show()