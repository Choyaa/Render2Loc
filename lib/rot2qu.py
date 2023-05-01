import numpy as np

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

if __name__=="__main__":
    
    # input: rotation matrix(c2w)
    rotate = [] 
    R = rotate[:3, :3] 
    t = rotate[:3, 3]
    # output: quaternion, translation(w2c)
    new_qvec = rotmat2qvec(R.T)
    new_t = -R.T @ t
    print("qwxyz:", new_qvec)
    print("t:", new_t)
        
    