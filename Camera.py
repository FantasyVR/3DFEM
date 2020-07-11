import math
import numpy as np
from numpy import linalg as LA
import time

# quternion to Rotation matrix
# see link: https://www.andre-gaschler.com/rotationconverter/
# and https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
def qToR(q=[0.0, 0.0, 0.0, 1.0], R=np.identity(3)):
    # normalize
    q = np.asmatrix(q)
    if LA.norm(q) < 1.0e-6:
        q = np.matrix([0, 0, 0, 1.0])
    else:
        q = (q / LA.norm(q))
    qx, qy, qz, qw = q[0, 0], q[0, 1], q[0, 2], q[0, 3]
    R = np.matrix([[
        1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw,
        2 * qx * qz + 2 * qy * qw
    ],
                   [
                       2 * qz * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2,
                       2 * qy * qz - 2 * qx * qw
                   ],
                   [
                       2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw,
                       1 - 2 * qx**2 - 2 * qy**2
                   ]])
    return R


def aToq(axis=[1.0, 0.0, 0.0], angle=40):
    sinAngle = math.sin(angle * math.pi / 360.0)
    cosAngle = math.cos(angle * math.pi / 360.0)
    axis = np.array(axis)
    axis = axis / LA.norm(axis)
    axis = [x * sinAngle for x in axis]
    axis.append(cosAngle)
    return np.matrix(axis)


def model(translate=[1.0, 2.0, 3.0],
          rotateQ=[1.0, 0.0, 0.0, 1.0],
          scale=[1.0, 1.0, 1.0]):
    scale.append(1.0)
    S = np.diag(scale)
    trans = np.asmatrix(translate)
    R = qToR(rotateQ)
    t = np.matrix([0.0, 0.0, 0.0])
    R = np.block([[R, trans.T], [t, 1.0]])
    return R * S

#http://www.songho.ca/opengl/gl_camera.html#lookat
def view(pos=[0.0, 0.0, 5.0], target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0]):
    pos = np.array(pos)
    target = np.array(target)
    up = np.array(up)

    viewDir = pos - target
    if LA.norm(viewDir) < 1.0e-6:
        viewDir = np.array([0.0, 0.0, -1.0])
    else:
        viewDir = viewDir / LA.norm(viewDir)

    right = np.cross(up, viewDir)
    right = right / LA.norm(right)

    up = np.cross(viewDir, right)
    trans = -pos
    x = np.dot(right, trans)
    y = np.dot(up, trans)
    z = np.dot(viewDir, trans)
    view = np.block([[right, x], [up, y], [viewDir, z], [0.0, 0.0, 0.0, 1.0]])
    return view

#http://www.songho.ca/opengl/gl_projectionmatrix.html
def projective(eye_fov=45, aspect_ration=1, zNear=0.1, zFar=50):
    height = 2 * zNear * math.tan(eye_fov * math.pi / 360.0)
    width = height * aspect_ration

    r = 0.5 * width
    t = 0.5 * height
    projection = np.array([[zNear / r, 0, 0, 0], [0, zNear / t, 0, 0],
                            [
                                0, 0, (zNear + zFar) / (zNear - zFar),
                                2 * zFar * zNear / (zNear - zFar)
                            ], [0, 0, -1, 0]])
    return projection

def display(gui, pos, mvp):

    p = [np.array([pos[i][0],pos[i][1],pos[i][2], 1.0]) for i in range(3)] # p.xyz -> p.xyzw
    pp = [np.matmul(mvp, p[i]) for i in range(3) ] # mvp * p.xyzw
    pp = [pp[i] / pp[i][3] for i in range(3)] # p.xyz / p.w -> ndc coordinate
    pp = [(pp[i][j] + 1)/2.0 for i in range(3) for j in range(2)] # [-1,1] -> [0,1]
    gui.triangle([pp[0],pp[1]],[pp[2],pp[3]],[pp[4],pp[5]],color=0xFF0000)
    #print(pp)
    #for i in range(3):
        #p = np.array([pos[i][0],pos[i][1],pos[i][2], 1.0])
        #pp = np.matmul(mvp, p)
        #pp = pp / pp[3]
        #pp = [(x+1)/2 for x in pp]
        #print(pp)
       # gui.circle([pp[i][0], pp[i][1]], color=0xFF0000, radius=5)
    
if __name__ == "__main__":
    
    import taichi as ti 
    ti.init(arch=ti.cpu)
    height = 521
    width = height

    pos = ti.Vector(3,dt=ti.f32, shape=3)


    pos[0] = [1.0,0,0]
    pos[1] = [0.0,1,0]
    pos[2] = [-1.0,0,0]
    gui = ti.GUI("Hello Triangle",res=(width,height))
    pause = True
    trans = [0.0,0.0,5.0]
    while gui.running:
        for e in gui.get_events():
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE and e.type == ti.GUI.PRESS:
                pause = not pause
                print("Press SPACE")
            elif e.key == 'w' and e.type == ti.GUI.PRESS:
                trans[2] -= 0.5
            elif e.key == 's' and e.type == ti.GUI.PRESS:
                trans[2] += 0.5
            elif e.key == 'd' and e.type == ti.GUI.PRESS:
                trans[0] += 0.5
            elif e.key == 'a' and e.type == ti.GUI.PRESS:
                trans[0] -= 0.5
        radius = 5
        t = time.time()
        trans[0] = radius * math.cos(t)
        trans[2] = radius * math.sin(t)
        
        p = projective()
        v = view(pos= trans)
        m = np.identity(4)
        mvp = np.matmul(p, np.matmul(v,m))
        display(gui, pos, mvp)
        gui.show()